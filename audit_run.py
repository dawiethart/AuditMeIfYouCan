# audit_runner.py
import time

import pandas as pd
import torch
import numpy as np

import wandb
from evaluation import compute_blackbox_auc_difference, evaluate_inner_loop, evaluate_outer_loop
from optimization import eval_h
from utils import (delta_progress, df_map, select_topk_stratified_disagreement,
                   stratified_sampling, plot_weight_evolution, random_ordered_sampling, stratified_ordered_sampling)


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
    def test(self, surrogate_preds, eval_h1, eval_h2, save_path="disagreement_snapshot.csv"):
        """
        Save predictions, binarized labels, and disagreement flags to a CSV.

        Parameters:
            surrogate_preds (np.ndarray): Scores from the surrogate model.
            eval_h1 (np.ndarray): Scores from h1 (maximizing ΔAUC).
            eval_h2 (np.ndarray): Scores from h2 (minimizing ΔAUC).
            save_path (str): File path to save the CSV.
        """

        # Convert scores to tensors
        sur_t = torch.from_numpy(surrogate_preds)
        h1_t  = torch.as_tensor(eval_h1)
        h2_t  = torch.as_tensor(eval_h2)

        # Binarize predictions
        sur_lbls = (sur_t >= 0.5).to(torch.int)
        h1_lbls  = (h1_t  >= 0.5).to(torch.int)
        h2_lbls  = (h2_t  >= 0.5).to(torch.int)

        # Convert to numpy for saving
        disagreement_df = pd.DataFrame({
            "id": self.D["id"].values,
            "text": self.D["text"].values,
            "surrogate_score": sur_t.cpu().numpy(),
            "h1_score": h1_t.cpu().numpy(),
            "h2_score": h2_t.cpu().numpy(),
            "surrogate_label": sur_lbls.cpu().numpy(),
            "h1_label": h1_lbls.cpu().numpy(),
            "h2_label": h2_lbls.cpu().numpy(),
            "disagrees": ((h1_lbls != sur_lbls) | (h2_lbls != sur_lbls)).cpu().numpy()
        })

        # Save to CSV
        disagreement_df.to_csv(save_path, index=False)
        print(f"[INFO] Disagreement snapshot saved to: {save_path}")
    
    def refine_until_converged(
        self,
        surrogate,          # surrogate model h^ already trained on S
        tokenizer,
        base_model,
        inputs_D,
        df_D_mapped,
        S,                  # your current surrogate training set
    ):
        # 0) initialize uniform weights over D
        weights = pd.Series(1.0 / len(self.D), index=self.D.index)
        weight_history = []
        selected_ids_history = []

        # 0.5) compute h^(x) for all x in D once
        surrogate_preds = self.predict(
            self.D["text"].tolist(),
            tokenizer,
            surrogate,
            self.config.batch_size
        )
        
        sur_labels = (surrogate_preds >= 0.5).astype(int)   # array of 0s and 1s

        # 2a) sample fresh thresholds for every x in D
        λ = torch.log(torch.tensor(1e6 * 2 ** 100 / 0.004))
        λ = λ*0.1
        thresholds = pd.Series(
            torch.distributions.Exponential(λ)
                    .sample([len(self.D)])
                    .numpy(),
            index=self.D.index,
        )

        # 1) start with an empty constraint set T
        T = pd.DataFrame(columns=self.D.columns)

        # 2) inner refinement loop
        for inner_it in range(50):
          
            # Take at least one positive and one negative from each group (0,1)
            safe_T = []
            for group in [0, 1]:
                group_df = self.D[self.D["group"] == group]
                group_df_labels = (self.predict(
                        group_df["text"].tolist(),
                        tokenizer,
                        surrogate,
                        self.config.batch_size
                    ) >= 0.5)
               
                for label in [0, 1]:
                    subset = group_df[group_df_labels == label]
                    
                    if not subset.empty:
                        safe_T.append(subset.sample(1, random_state=inner_it))  # or .iloc[[0]]
            if len(T) >= 4:
                T = pd.concat([T, pd.concat(safe_T)], ignore_index=True).drop_duplicates()
            else:
                
                T = pd.concat(safe_T) 
        

            print(f"T:{len(T)}")

            # 2b) build constraint_dict only on the current T
            #     enforce h(x)=h^(x) ∀ x ∈ T
                   # 3) build your constraint_dict over T
            constraint_dict = {
                _id: float(sur_labels[idx])      # cast to float if your solver expects floats
                for idx, _id in zip(T.index, T["id"])
            }
            print(f"Number of constraints: {len(constraint_dict)}")
            print(f"constraints: {constraint_dict}")
            # 2c) prepare T for eval_h
            df_T, df_T_mapped = df_map(T, tokenizer, False)

            # 2d) fit the two penalized models h₁,h₂ over D
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

            # 2e) record current weights and log ΔAUC, stats, IDs
            weight_history.append(weights.copy())
            print("EVALUTATING INNER LOOP")
            delta_inner = evaluate_inner_loop(
                D=self.D,
                weights_history=weight_history,
                thresholds=thresholds,
                delta1=delta1,
                delta2=delta2,
                epsilon=self.config.epsilon,
            )
            # convergence?
            print("EVALUATING CONVERGENCE")
            if delta_inner <= 2 * self.config.epsilon and inner_it > 1:
                print(f"Converged @ inner it={inner_it}: ΔAUC(h1,h2)={delta_inner:.4f}")
                break

            # 2f) find disagreements on D vs surrogate_preds
            omega = 1e-4
            sur_t = torch.from_numpy(surrogate_preds)      # dtype inferred, probably float
            h1_t  = torch.as_tensor(eval_h1)
            h2_t  = torch.as_tensor(eval_h2)
            
            # binarize into int tensors
            sur_lbls = (sur_t >= 0.5).to(torch.int)
            h1_lbls  = (h1_t  >= 0.5).to(torch.int)
            h2_lbls  = (h2_t  >= 0.5).to(torch.int)

            disagree_mask = (h1_lbls != sur_lbls) | (h2_lbls != sur_lbls)
            mask_np      = disagree_mask.cpu().numpy()
            disagree_ids = self.D.index[mask_np]
            #self.test(surrogate_preds, eval_h1, eval_h2, save_path=f"disagreement_iter_{inner_it}.csv")
            if len(disagree_ids) == 0:
                
                break
            # 2g) bump those weights until their total ≤1
            while weights.loc[disagree_ids].sum() <= 1:
                print(weights.loc[disagree_ids].sum())
                weights.loc[disagree_ids] = weights.loc[disagree_ids] * 2
            print(weights.max)

            # 2h) define new T = { x : weight[x] ≥ threshold[x] }
            keep_mask = weights >= thresholds
            selected_ids_history.append(weights[keep_mask].copy())
            T = self.D.loc[keep_mask].copy()
            print(f"T has now {len(T)} samples")

            
            selected_ids_history.append(T.copy())
            #S = self.D.loc[T.index].copy()

           
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

       
            # Precompute baselines once
        random_D, _ = random_ordered_sampling(self.D, self.api, seed=42)
        random_D["bb_score"] = self.api(random_D["text"].tolist())
        strat_D, _ = stratified_ordered_sampling(self.D, self.api,
                                                group_col="group",
                                                group1=0, group2=1)
        strat_D["bb_score"] = self.api(strat_D["text"].tolist())

        for it in range(self.config.iterations):
            start = time.time()
            print(f"\n=== Iteration {it+1} ===")

            # 1) Active query & retrain on T then S
            T = self.refine_until_converged(surrogate, tokenizer, base_model,
                                            inputs_D, df_D_mapped, S)
            T["bb_score"] = self.api(T["text"].tolist())
            S = pd.concat([S, T], ignore_index=True).drop_duplicates("id")

            df_S, df_S_mapped = df_map(S, tokenizer, True)
            surrogate = self.train_surrogate(base_model, df_S, df_S_mapped,
                                            self.config.epochs_sur,
                                            self.config.batch_size).to("cpu")
            torch.cuda.empty_cache()

            # 2) Evaluate surrogate ΔAUC
            _, probs = self.compute_auc_diff(surrogate.to(self.device),
                                            inputs_D, self.D)
            delta_final = compute_blackbox_auc_difference(
                self.D["true_label"].astype(int),
                self.D["group"],
                probs
            )

            # 3) Slice baselines to current budget
            budget = len(S)
            rand_S = random_D.iloc[:budget]
            strat_S = strat_D.iloc[:budget]

            # 4) Log outer‐loop metrics
            evals = evaluate_outer_loop(
                ground_truth_delta=self.delta_auc_blackbox,
                D=self.D,
                sample_S=S,      sample_scores=S["bb_score"].values,
                random_S=rand_S, random_scores=rand_S["bb_score"].values,
                strat_S=strat_S, stratified_scores=strat_S["bb_score"].values,
                group1=0, group2=1
            )
            wandb.log({
                "iteration": it,
                "query_budget": budget,
                "delta_estimate": float(delta_final),
                "delta_true": float(self.delta_auc_blackbox),
                "delta_error": abs(delta_final - self.delta_auc_blackbox),
                **{f"ΔAUC/{k}": v for k, v in evals.items()},
                "duration_min": (time.time() - start) / 60,
            })

            print(f"[Iter {it+1}] ΔAUC_est: {delta_final:.4f} | "
                f"ΔAUC_true: {self.delta_auc_blackbox:.4f} | "
                f"Budget: {budget} | Time: {(time.time()-start)/60:.2f}m")

        return delta_final