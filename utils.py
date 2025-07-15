# utils.py

import numpy as np
import pandas as pd
from datasets import Dataset


def df_map(dataset, tokenizer, surrogate):
    df = dataset.copy()

    # Assign labels depending on surrogate mode
    df["labels"] = df["bb_score"] if surrogate else df["true_label"]

    # === Convert group labels (e.g. "male", "female") to integers ===
    if "group" in df.columns:
        unique_groups = sorted(df["group"].unique())
        group_map = {g: i for i, g in enumerate(unique_groups)}
        df["group"] = df["group"].map(group_map)
        print(f"[INFO] Group mapping used: {group_map}")
    else:
        print("[WARNING] 'group' column not found in dataset.")

    df_mapped = Dataset.from_pandas(df)
    df_mapped = df_mapped.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    return df, df_mapped


def tokenize_batch(batch, tokenizer):
    tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    return tokenized


def stratified_sampling(size, dataset):
    dataset = dataset.copy()
    size_per_stratum = size // 4

    strata = {
        "white_1": dataset[(dataset.group == "white") & (dataset.true_label == 1)].index,
        "white_0": dataset[(dataset.group == "white") & (dataset.true_label == 0)].index,
        "black_1": dataset[(dataset.group == "black") & (dataset.true_label == 1)].index,
        "black_0": dataset[(dataset.group == "black") & (dataset.true_label == 0)].index,
    }

    sampled_idx = []
    for idx in strata.values():
        n = min(len(idx), size_per_stratum)
        sampled_idx.extend(np.random.choice(idx, size=n, replace=False))

    sampled_idx = list(set(sampled_idx))
    shortfall = size - len(sampled_idx)
    if shortfall > 0:
        remaining = dataset.index.difference(sampled_idx)
        extra = np.random.choice(remaining, size=shortfall, replace=False)
        sampled_idx.extend(extra)

    return dataset.loc[sampled_idx].reset_index(drop=True)


def select_topk_stratified_disagreement(df_D, disagreements, top_k_per_bucket=5):
    """
    Selects top-k disagreement examples per group-label bucket from df_D.

    Parameters:
        df_D (pd.DataFrame): Full dataset D with at least columns ['group', 'true_label', 'text', 'id']
        disagreements (np.array): Array of disagreement values, same length as df_D
        top_k_per_bucket (int): Number of examples to select per group-label bucket

    Returns:
        new_T (pd.DataFrame): Selected top-k*4 examples with high disagreement, stratified by group and label
    """
    df_eval = df_D.copy()
    df_eval["disagreement"] = disagreements

    # Define stratified buckets
    buckets = {
        "white_1": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 1)],
        "white_0": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 0)],
        "black_1": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 1)],
        "black_0": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 0)],
    }

    selected_dfs = []
    for key, bucket in buckets.items():
        sorted_bucket = bucket.sort_values("disagreement", ascending=False)
        top_k = sorted_bucket.head(top_k_per_bucket)
        selected_dfs.append(top_k)

    new_T = pd.concat(selected_dfs).drop_duplicates(subset="id")
    return new_T


def delta_progress(df_new, df_old, iteration, delta_auc_blackbox, compute_auc_fn):
    """
    Computes the progression of delta AUC as more points are added.

    Parameters:
        df_new (pd.DataFrame): New evaluation samples.
        df_old (pd.DataFrame): Previously evaluated samples.
        iteration (int): Current iteration.
        delta_auc_blackbox (float): Baseline delta AUC to compare against.
        compute_auc_fn (Callable): Function to compute group AUC difference.

    Returns:
        List[float]: List of delta values at each prefix of df_new.
    """
    delta_auc = []

    for i in range(len(df_new)):
        if iteration == 0:
            df_progress = df_new.iloc[:i]
        else:
            df_progress = pd.concat([df_new.iloc[:i], df_old])

        delta = compute_auc_fn(
            labels=df_progress["true_label"].astype(int),
            groups=df_progress["group"],
            scores=df_progress["bb_score"],
        )
        delta_auc.append(abs(delta - delta_auc_blackbox))

    return delta_auc
