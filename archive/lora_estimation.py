
import pandas as pd
import numpy as np
from peft import LoraModel, LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import DataCollatorWithPadding
from datasets import Value
# === Load Datasets ===




def load_lora_bert_surrogate(model_name="distilbert-base-uncased", num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    for name, param in base_model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    
    lora_config = LoraConfig(
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # works with DistilBERT
        r=4,                       # more expressive LoRA
        lora_alpha=16,            # keep alpha same as before
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Optional debug output

    return tokenizer, model


def tokenize_batch(batch, tokenizer):
    tokenized = tokenizer(
        batch['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=128
    )
    return tokenized

def df_map(dataset, tokenizer, surrogate):
    df = dataset.copy()
    if surrogate:
        df['labels'] = df['bb_score']  # float in [0,1]
    else: 
        df['labels'] = df['true_label']
 
    df_mapped = Dataset.from_pandas(df)
    df_mapped = df_mapped.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    return df, df_mapped


def predict_with_model(texts, tokenizer, model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

@torch.no_grad()
def compute_group_auc_difference(model, inputs, dataset, group1='white', group2='black', batch_size=128):
    device = next(model.parameters()).device
    model.eval()

    # Construct a DataLoader
    ds = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    dl = DataLoader(ds, batch_size=batch_size)

    all_scores = []
    all_ids = []
    all_labels = []
    true_ids = inputs["id"]
    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(dl):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().tolist())
            batch_ids = true_ids[i * batch_size: (i + 1) * batch_size]
            if torch.is_tensor(batch_ids):
                batch_ids = batch_ids.detach().cpu().tolist()
            else:
                batch_ids = [
                    i.detach().cpu().item() if torch.is_tensor(i) else i
                    for i in batch_ids
                ]

            all_ids.extend(batch_ids)

    # Convert scores to pandas Series with ID index
   
    scores_series = pd.Series(all_scores, index=all_ids)

    # Filter dataset to those we just scored
    df_eval = dataset[dataset['id'].isin(all_ids)].copy()

    
    df_eval['score'] = df_eval['id'].map(scores_series)

    try:
        auc1 = roc_auc_score(df_eval[df_eval['group'] == group1]['true_label'],
                             df_eval[df_eval['group'] == group1]['score'])
    except ValueError:
        auc1 = 0

    try:
        auc2 = roc_auc_score(df_eval[df_eval['group'] == group2]['true_label'],
                             df_eval[df_eval['group'] == group2]['score'])
    except ValueError:
        auc2 = 0

    return torch.tensor(auc1 - auc2), scores_series



def custom_loss(model, inputs, dataset_T, constraint_pred, dataset, lambda_penalty=5.0, maximize = True):
    device = next(model.parameters()).device  # Get model's device
    # === Loss Constraint ===
    inputs_T = {
        "input_ids": torch.tensor(dataset_T['input_ids']).long(),  # Long is fine for input_ids
        "attention_mask": torch.tensor(dataset_T['attention_mask']).long(),  # Long is fine for attention_mask
        "labels": torch.tensor(dataset_T['labels']).long(),  # Long for classification labels
        "id": dataset_T['id']
    }
    
    constraint_logits = model(**{k: v.to(device) for k, v in inputs_T.items() if k != "id"}).logits.view(-1)  # h(T)
    constraint_probs = torch.sigmoid(constraint_logits)

    # The conctraint preds will in reality be given by the loop before, here symbolised with np.random.random
    if isinstance(constraint_pred, torch.Tensor):
        constraint_targets = constraint_pred.clone().detach().to(dtype=torch.float32, device=device)
    else:
        constraint_targets = torch.tensor(constraint_pred, dtype=torch.float32, device=device)


    loss_constraint = torch.nn.functional.mse_loss(
        constraint_probs, 
        constraint_targets  # Now Float
    )

    # === AUC loss on D ===
    with torch.no_grad():
        loss_auc, _ = compute_group_auc_difference(model, inputs, dataset)
        loss_auc = loss_auc.float().to(device)

    # === Total loss ===
    return (loss_constraint - lambda_penalty * loss_auc) if maximize else (loss_constraint + lambda_penalty * loss_auc)

def train_surrogate(model, df_S, df_S_mapped, epochs, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def custom_loss_surrogate(model, inputs):
    
        device = next(model.parameters()).device

        logits = model(**inputs).logits.view(-1)
        probs = torch.sigmoid(logits)

        labels = inputs["labels"].float().to(device)
       
        loss = torch.nn.functional.mse_loss(probs, labels)

        return loss

    training_args = TrainingArguments(
        output_dir="./tmp_cerm",
        label_names = ['labels'],
        per_device_train_batch_size=batch_size,
        dataloader_pin_memory=False,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        disable_tqdm=True
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
            outputs = model(**inputs)
            loss = custom_loss_surrogate(model, inputs)
            
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=df_S_mapped,
        
    )

    trainer.train()
    return model


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        preserved = {
            key: [f[key] for f in features]
            for key in ["id", "group", "text"] if key in features[0]
        }

        # Remove preserved keys from features
        for f in features:
            for key in preserved:
                f.pop(key, None)

        # Standard collation
        batch = super().__call__(features)

        # Re-inject preserved fields
        for key, values in preserved.items():
            if key == "id":
                # Convert to string → int if needed
                try:
                    values = [int(v) for v in values]
                except ValueError:
                    # fallback: encode string ids to index
                    unique = {v: i for i, v in enumerate(sorted(set(values)))}
                    values = [unique[v] for v in values]
                batch[key] = torch.tensor(values)
            
            else:  # e.g., text
                batch[key] = values  # leave as list of str
        
        
        return batch
    
def train_cerm_pairwise(model, df_D, df_D_mapped, df_T_mapped, constraint_pred, epochs, batch_size, lambda_penalty, tokenizer, maximize=True):
    import gc
    df_D_mapped = df_D_mapped.cast_column("id", Value("int64"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if next(model.parameters()).device != device:
        model.to(device)

    # Move constraint_pred and df_T_mapped to CPU to avoid unnecessary CUDA use
    if isinstance(constraint_pred, torch.Tensor):
        constraint_pred = constraint_pred.detach().cpu()
    elif isinstance(constraint_pred, np.ndarray):
        constraint_pred = torch.tensor(constraint_pred, device="cpu")

    if hasattr(df_T_mapped, "features"):
        df_T_dict = df_T_mapped.with_format(None).to_dict()
    else:
        df_T_dict = df_T_mapped

    for k in df_T_dict:
        val = df_T_dict[k]
        if isinstance(val, torch.Tensor):
            df_T_dict[k] = val.detach().cpu()
        elif isinstance(val, np.ndarray):
            df_T_dict[k] = torch.tensor(val, device="cpu")

    # Overwrite df_T_mapped with safe, detached version
    df_T_mapped = Dataset.from_dict(df_T_dict)

    training_args = TrainingArguments(
        output_dir="./tmp_cerm",
        label_names=['labels'],
        per_device_train_batch_size=batch_size,
        dataloader_pin_memory=False,
        num_train_epochs=epochs,
        logging_steps=100,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=1,
        fp16=torch.cuda.is_available(),
        disable_tqdm=True
    )
    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            
            loss = custom_loss(model, inputs, df_T_mapped, constraint_pred, dataset=df_D,
                               lambda_penalty=lambda_penalty, maximize=maximize)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=df_D_mapped,
    data_collator=CustomDataCollator(tokenizer)
    )

    trainer.train()

    # Free Trainer memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return model
