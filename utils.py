# utils.py

import numpy as np
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import os
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

import numpy as np
import pandas as pd

def random_ordered_sampling(D: pd.DataFrame, api_fn, seed: int = None):
    """
    Return:
      • rand_D: DataFrame with all rows of D in a random permutation
      • scores:  np.ndarray of black-box scores in that same permuted order
    """
    # shuffle indices without replacement
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D.index.values)
    rand_D = D.loc[perm].reset_index(drop=True)
    # query black‐box once for each text, in this new order
    scores = np.array(api_fn(rand_D["text"].tolist()))
    return rand_D, scores


def stratified_ordered_sampling(
    D: pd.DataFrame,
    api_fn,
    group_col: str = "group",
    group1: str = "white",
    group2: str = "black",
):
    """
    Return:
      • strat_D: DataFrame with all rows of D interleaved by group:
          [g1_0, g2_0, g1_1, g2_1, … , then the leftovers of the larger group]
      • scores:   np.ndarray of black-box scores in that same interleaved order
    """
    # split out each group in original order
    idx1 = D[D[group_col] == group1].index.to_list()
    idx2 = D[D[group_col] == group2].index.to_list()

    interleaved = []
    m = min(len(idx1), len(idx2))
    # interleave one by one
    for i in range(m):
        interleaved.append(idx1[i])
        interleaved.append(idx2[i])
    # append any leftovers
    if len(idx1) > m:
        interleaved.extend(idx1[m:])
    if len(idx2) > m:
        interleaved.extend(idx2[m:])

    strat_D = D.loc[interleaved].reset_index(drop=True)
    scores = np.array(api_fn(strat_D["text"].tolist()))
    return strat_D, scores


def plot_weight_evolution(weight_history, selected_ids_history, save_dir="audit_plots"):
    """
    Plots and saves:
    1. Weight evolution of tracked sample IDs across iterations.
    2. Size of T (number of selected samples) per iteration.

    Parameters:
        weight_history: List of pd.Series (sample weights at each iteration)
        selected_ids_history: List of pd.Series (selected IDs at each iteration)
        save_dir: Directory where to save the plots (default: 'audit_plots')
    """
    os.makedirs(save_dir, exist_ok=True)
    iterations = list(range(len(weight_history)))

    # === Plot 1: Weight evolution for tracked example IDs ===
    all_ids = set().union(*[set(w.index) for w in weight_history])
    tracked_ids = list(all_ids)[:10]

    selected_indices = list(range(0, len(weight_history), 1))[:6]
    num_plots = len(selected_indices)

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), sharey=True)
    if num_plots == 1:
        axes = [axes]

    for ax, i in zip(axes, selected_indices):
        weights = weight_history[i]
        ax.hist(weights, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.grid(True)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "weight_evolution.png")
    plt.savefig(path1)
    plt.show()

    # === Plot 2: Number of selected T samples per iteration ===
    plt.figure(figsize=(8, 4))
    t_sizes = [len(s) for s in selected_ids_history]
    plt.plot(iterations, t_sizes, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of T samples added")
    plt.title("T Size Over Iterations")
    plt.grid(True)
    plt.tight_layout()
    path2 = os.path.join(save_dir, "t_size_over_iterations.png")
    plt.savefig(path2)
    plt.show()

    print(f"Plots saved to:\n - {path1}\n - {path2}")