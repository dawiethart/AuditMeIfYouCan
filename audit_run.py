# audit_runner.py
import time

import pandas as pd
import torch
import numpy as np

import wandb
from evaluation import compute_blackbox_auc_difference
from optimization import eval_h
from utils import (delta_progress, df_map, select_topk_stratified_disagreement,
                   stratified_sampling)


class AuditRunner:
    def __init__(
        self,
        dataset_D,
        black_box_api_fn,
        surrogate_model_loader,
        train_surrogate_fn,
        predict_fn,
        compute_group_auc_diff_fn,
        config,
    ):
        self.D = dataset_D
        self.api = black_box_api_fn
        self.load_surrogate = surrogate_model_loader
        self.train_surrogate = train_surrogate_fn
        self.predict = predict_fn
        self.compute_auc_diff = compute_group_auc_diff_fn
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.delta_auc_blackbox = compute_blackbox_auc_difference(
            labels=self.D["true_label"].astype(int),
            groups=self.D["group"],
            scores=self.api(self.D["text"].tolist()),
        )

    def initialize_dataset_S(self, size):
        S = stratified_sampling(size, self.D)
        S["bb_score"] = self.api(S["text"].tolist())
        return S

    def refine_until_converged(self, surrogate, tokenizer, base_model, inputs_D, df_D_mapped, S):
        weights = pd.Series(1.0 / len(self.D), index=self.D.index)
        lambda_param = torch.log(torch.tensor(1e6 * 2 ** 100 / 0.05))  # Example H, M, delta
        thresholds = pd.Series(
            torch.distributions.Exponential(lambda_param).sample([len(self.D)]).numpy(),
            index=self.D.index,
        )

        surrogate_preds = self.predict(
                self.D["text"].tolist(), tokenizer, surrogate, self.config.batch_size
            )
        self.D["surrogate_score"] = surrogate_preds

        T = S.drop(columns = "bb_score") # first random stratified sample
 
        for i in range(50):  # Max 50 inner iterations to avoid infinite loops

            constraint_ids = T["id"] #Use subset T of surrogate_preds as Constraint
            constraint_preds = surrogate_preds #Careful: has size D and not size T
            constraint_dict = dict(zip(constraint_ids, surrogate_preds))

            df_T, df_T_mapped = df_map(T, tokenizer, False)
           
            delta1, eval_h1 = eval_h(
                base_model=base_model,
                df_D=self.D,
                df_D_mapped=df_D_mapped,
                inputs_D=inputs_D,
                df_T_mapped=df_T_mapped,
                constraint_pred=constraint_dict,
                epochs_opt=self.config.epochs_opt,
                batch_size=self.config.batch_size,
                lambda_penalty=self.config.lambda_penalty,
                tokenizer=tokenizer,
                Maximize=True,
                compute_group_auc_diff_fn=self.compute_auc_diff,
            )

            delta2, eval_h2 = eval_h(
                base_model=base_model,
                df_D=self.D,
                df_D_mapped=df_D_mapped,
                inputs_D=inputs_D,
                df_T_mapped=df_T_mapped,
                constraint_pred=constraint_dict,
                epochs_opt=self.config.epochs_opt,
                batch_size=self.config.batch_size,
                lambda_penalty=self.config.lambda_penalty,
                tokenizer=tokenizer,
                Maximize=False,
                compute_group_auc_diff_fn=self.compute_auc_diff,
            )

            if abs(delta1 - delta2) <= 2 * self.config.epsilon:
                print(f"Stopping: ΔAUC(h1-h2) = {abs(delta1 - delta2):.4f} within 2ε tolerance")
                break

            disagreement_mask = ((eval_h1 - constraint_preds).abs() > self.config.epsilon) | (
                (eval_h2 - constraint_preds).abs() > self.config.epsilon
            ) #Use full dataset D to compare h1/h2(D) with h^(D)

            disagreement_ids = self.D[np.array(disagreement_mask)].index 
            while sum(weights.loc[disagreement_ids]) > 1: #Hier wurde nur einmal verdoppelt und nicht, bis die Abbruchbedingung erreicht ist
                weights.loc[disagreement_ids] *= 2 

            newly_covered = weights[weights >= thresholds]
            new_points = self.D.loc[newly_covered.index].copy()

            print(f"Added {len(new_points)} samples to T")
            print(f"ΔAUC(h1-h2) = {abs(delta1 - delta2):.4f}, epsilon = {self.config.epsilon}")

            T = new_points.copy()

            torch.cuda.empty_cache()

        return T

    def run(self):
        tokenizer, base_model = self.load_surrogate()

        self.D, df_D_mapped = df_map(self.D, tokenizer, False)
        inputs_D = {
            "input_ids": torch.tensor(df_D_mapped["input_ids"]).long(),
            "attention_mask": torch.tensor(df_D_mapped["attention_mask"]).long(),
            "labels": torch.tensor(df_D_mapped["labels"]).long(),
            "id": df_D_mapped["id"],
        }

        S = self.initialize_dataset_S(self.config.size_T)
        df_S, df_S_mapped = df_map(S, tokenizer, True)

        surrogate = self.train_surrogate(
            base_model, df_S, df_S_mapped, self.config.epochs_sur, self.config.batch_size
        )
        surrogate = surrogate.to("cpu")
        torch.cuda.empty_cache()

        for iteration in range(self.config.iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            start_time = time.time()

            T = self.refine_until_converged(surrogate, tokenizer, base_model, inputs_D, df_D_mapped, S)

            T["bb_score"] = self.api(T["text"].tolist())

            df_T, df_T_mapped = df_map(T, tokenizer, True)
            surrogate = self.train_surrogate(
                base_model, df_T, df_T_mapped, self.config.epochs_sur, self.config.batch_size
            )
            surrogate = surrogate.to("cpu")

            S = pd.concat([S, T], ignore_index=True).drop_duplicates(subset="id")

            df_S, df_S_mapped = df_map(S, tokenizer, True)
            surrogate = self.train_surrogate(
                base_model, df_S, df_S_mapped, self.config.epochs_sur, self.config.batch_size
            )
            surrogate = surrogate.to("cpu")
            torch.cuda.empty_cache()

            _, probs = self.compute_auc_diff(surrogate.to(self.device), inputs_D, self.D)
            delta_final = compute_blackbox_auc_difference(
                labels=self.D["true_label"].astype(int), groups=self.D["group"], scores=probs
            )

            delta_progress_vals = delta_progress(
                df_new=T,
                df_old=S,
                iteration=iteration,
                delta_auc_blackbox=self.delta_auc_blackbox,
                compute_auc_fn=compute_blackbox_auc_difference
            )

            wandb.log(
                {
                    "iteration": iteration,
                    "query_budget": len(S),
                    "delta_auc_estimate": float(delta_final),
                    "delta_auc_true": float(self.delta_auc_blackbox),
                    "delta_auc_abs_error": abs(delta_final - self.delta_auc_blackbox),
                    f"delta_auc_progress_iter_{iteration}": delta_progress_vals,
                    "duration_min": (time.time() - start_time) / 60,
                }
            )

            print(f"[Iteration {iteration+1}] Final surrogate ΔAUC: {delta_final:.4f}")
            print(f"[Iteration {iteration+1}] Duration: {(time.time() - start_time) / 60:.2f} min")

        return delta_final
