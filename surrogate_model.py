# surrogate_model.py

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics.functional import auroc
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def load_lora_bert_surrogate(model_name="bert-base-uncased", num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    for param in base_model.parameters():
        param.requires_grad = False
    for name, param in base_model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    lora_modules = (
        ["query", "value"] if "bert" in model_name else ["q_lin", "k_lin", "v_lin", "out_lin"]
    )
    lora_config = LoraConfig(
        target_modules=lora_modules,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    return tokenizer, model


def train_surrogate(model, df_S, df_S_mapped, epochs, batch_size):
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    def custom_loss_surrogate(model, inputs):
        logits = model(**inputs).logits.view(-1)
        probs = torch.sigmoid(logits)
        labels = inputs["labels"].float()
        return torch.nn.functional.mse_loss(probs, labels)

    training_args = TrainingArguments(
        output_dir="./tmp_cerm",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = custom_loss_surrogate(model, inputs)
            return (loss, model(**inputs)) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=df_S_mapped,
    )
    trainer.train()
    return model


@torch.no_grad()
def compute_group_auc_difference(model, inputs, dataset, group1=0, group2=1, batch_size=128):
    device = next(model.parameters()).device
    model.eval()

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)
    groups = torch.tensor(dataset["group"].values).to(device)

    logits = []
    for i in range(0, len(input_ids), batch_size):
        batch_logits = model(
            input_ids=input_ids[i : i + batch_size],
            attention_mask=attention_mask[i : i + batch_size],
        ).logits.view(-1)
        logits.append(batch_logits)

    logits = torch.cat(logits)
    probs = torch.sigmoid(logits)

    mask1 = groups == group1
    mask2 = groups == group2
    auc1 = auroc(probs[mask1], labels[mask1].long(), task="binary").item() if mask1.any() else 0.0
    auc2 = auroc(probs[mask2], labels[mask2].long(), task="binary").item() if mask2.any() else 0.0

    return torch.tensor(auc1 - auc2, device=device), probs.cpu()


def predict_with_model(texts, tokenizer, model, batch_size=16):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

        torch.cuda.empty_cache()  # optional but safe

    return np.array(all_probs)
