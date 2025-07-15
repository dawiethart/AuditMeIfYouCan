# evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_blackbox_auc_difference(
    labels: pd.Series, groups: pd.Series, scores: np.ndarray, group1="white", group2="black"
) -> float:
    """
    Compute the difference in AUC between two groups using model scores.

    Parameters:
        labels (pd.Series): True binary labels (0/1).
        groups (pd.Series): Group membership for each instance.
        scores (np.ndarray): Predicted probabilities or scores from black-box.
        group1 (str): Name of group 1 (e.g., 'white').
        group2 (str): Name of group 2 (e.g., 'black').

    Returns:
        float: AUC(group1) - AUC(group2), or 0 if one group fails.
    """
    df = pd.DataFrame({"score": scores, "true_label": labels, "group": groups})

    try:
        auc1 = roc_auc_score(
            df[df["group"] == group1]["true_label"], df[df["group"] == group1]["score"]
        )
    except ValueError:
        auc1 = 0.0

    try:
        auc2 = roc_auc_score(
            df[df["group"] == group2]["true_label"], df[df["group"] == group2]["score"]
        )
    except ValueError:
        auc2 = 0.0

    return auc1 - auc2
