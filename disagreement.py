# disagreement.py

import numpy as np
import pandas as pd


def select_topk_stratified_disagreement(df_D, disagreements, top_k_per_bucket=5):
    df_eval = df_D.copy()
    df_eval["disagreement"] = disagreements

    buckets = {
        "white_1": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 1)],
        "white_0": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 0)],
        "black_1": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 1)],
        "black_0": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 0)],
    }

    selected_dfs = [
        bucket.sort_values("disagreement", ascending=False).head(top_k_per_bucket)
        for bucket in buckets.values()
    ]
    return pd.concat(selected_dfs).drop_duplicates(subset="id")
