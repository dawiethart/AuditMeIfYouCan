import gc

import torch
from cooper import (CMPState, ConstrainedMinimizationProblem, Constraint,
                    ConstraintState, ConstraintType)
from cooper.multipliers import DenseMultiplier
from cooper.optim import SimultaneousOptimizer
from datasets import Value
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def eval_h(
    base_model,
    df_D,
    df_D_mapped,
    inputs_D,
    df_T_mapped,
    constraint_pred,
    epochs_opt,
    batch_size,
    lambda_penalty,
    tokenizer,
    Maximize,
    compute_group_auc_diff_fn,
):
    h = train_cerm_pairwise(
        base_model,
        df_D,
        df_D_mapped,
        df_T_mapped,
        constraint_pred,
        epochs_opt,
        batch_size,
        lambda_penalty,
        tokenizer,
        maximize=Maximize,
    )
    print("Start Calculation of AUC on whole D -> might take a while")
    with torch.no_grad():
        pred_h = compute_group_auc_diff_fn(h, inputs_D, df_D)
    print("Done with calculation whole D")
    eval_scores_h = pred_h[1].detach().to(torch.float32).cpu()
    delta = pred_h[0]

    h.to("cpu")
    del h
    gc.collect()
    torch.cuda.empty_cache()

    return delta, eval_scores_h


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        preserved = {
            key: [f[key] for f in features] for key in ["id", "group", "text"] if key in features[0]
        }
        for f in features:
            for key in preserved:
                f.pop(key, None)
        batch = super().__call__(features)

        for key, values in preserved.items():
            if key == "id":
                try:
                    values = [int(v) for v in values]
                except ValueError:
                    unique = {v: i for i, v in enumerate(sorted(set(values)))}
                    values = [unique[v] for v in values]
                batch[key] = torch.tensor(values)
            elif key == "group":
                try:
                    batch[key] = torch.tensor(values)
                except Exception:
                    pass
            else:
                batch[key] = values

        return batch


class PairwiseCERMProblem(ConstrainedMinimizationProblem):
    def __init__(self, model, inputs_T, constraint_pred, lambda_penalty, maximize, device):
        super().__init__()
        self.model = model
        self.inputs_T = inputs_T
        self.constraint_pred = constraint_pred
        self.lambda_penalty = lambda_penalty
        self.maximize = maximize
        self.device = device

        self.multiplier = DenseMultiplier(num_constraints=1, device=device)
        self.constraint = Constraint(
            multiplier=self.multiplier, constraint_type=ConstraintType.INEQUALITY
        )

    def compute_auc_surrogate(self, logits_pos, logits_neg):
        margin = 1.0
        pairwise_diff = logits_pos.view(-1, 1) - logits_neg.view(1, -1)
        return 1.0 - torch.sigmoid(margin * pairwise_diff).mean()

    def compute_cmp_state(self, model, inputs, targets):
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        targets = targets.to(self.device)
        logits = model(**model_inputs).logits.squeeze()
        logits = torch.clamp(logits, min=-20, max=20)

        groups = inputs["group"].to(self.device)
        logits_0 = logits[groups == 0]
        logits_1 = logits[groups == 1]
        labels_0 = targets[groups == 0]
        labels_1 = targets[groups == 1]

        auc_0 = self.compute_auc_surrogate(logits_0[labels_0 == 1], logits_0[labels_0 == 0])
        auc_1 = self.compute_auc_surrogate(logits_1[labels_1 == 1], logits_1[labels_1 == 0])

        auc_gap = torch.abs(auc_0 - auc_1)
        loss = -auc_gap if self.maximize else auc_gap

        logits_T = model(
            **{k: v.to(self.device) for k, v in self.inputs_T.items() if k != "group"}
        ).logits.squeeze()
        logits_T = torch.clamp(logits_T, min=-20, max=20)
        probs_T = torch.sigmoid(logits_T)
        
        constraint_val = torch.nn.functional.mse_loss(
            probs_T, self.constraint_pred.to(self.device), reduction="mean"
        )
        violation = constraint_val - self.lambda_penalty

        constraint_state = ConstraintState(violation=violation.unsqueeze(0))
        observed_constraints = {self.constraint: constraint_state}

        return CMPState(loss=loss, observed_constraints=observed_constraints)


def train_cerm_pairwise(
    model,
    df_D,
    df_D_mapped,
    df_T_mapped,
    dic_constraint,
    epochs,
    batch_size,
    lambda_penalty,
    tokenizer,
    maximize=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df_D_mapped = df_D_mapped.cast_column("id", Value("int64"))
    data_collator = CustomDataCollator(tokenizer)
    dataloader = DataLoader(
        df_D_mapped.with_format("torch"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    constraint_pred = [dic_constraint[key] for key in dic_constraint]
    constraint_pred = torch.tensor(constraint_pred, device="cpu")

    inputs_T = {
        "input_ids": torch.tensor(df_T_mapped["input_ids"]).long(),
        "attention_mask": torch.tensor(df_T_mapped["attention_mask"]).long(),
        "labels": torch.tensor(df_T_mapped["labels"]).float(),
    }

    cmp = PairwiseCERMProblem(model, inputs_T, constraint_pred, lambda_penalty, maximize, device)

    primal_optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=5e-2, maximize=True)

    optimizer = SimultaneousOptimizer(
        cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
    )

    model.train()
    losses = []
    violations = []
    
    for epoch in range(epochs):
        for batch_i, batch in enumerate(dataloader):
            inputs_D = {k: batch[k].to(device) for k in ["input_ids", "attention_mask", "group"]}
            targets_D = batch["labels"].to(device)

            rollout = optimizer.roll(
                compute_cmp_state_kwargs={
                    "model": model,
                    "inputs": inputs_D,
                    "targets": targets_D,
                }
            )

            cmp_state = rollout.cmp_state
            loss = cmp_state.loss.item()
            violation = list(cmp_state.observed_constraints.values())[0].violation.item()

            losses.append(loss)
            violations.append(violation)

            if batch_i % 198 == 0 or batch_i == len(dataloader) - 1:
                multiplier_val = cmp.multiplier().detach().cpu().item()
                print(f"Lambda: {multiplier_val:.4f}, Violation: {violation:.4f}")
                print(
                    f"[Epoch {epoch+1}, Batch {batch_i+1}/{len(dataloader)}] "
                    f"Loss: {loss:.4f}, Violation: {violation:.4f}"
                )

    print("Training done for C-ERM")
    return model
