import numpy as np
import pandas as pd


def load_jigsaw_subset(path="jigsaw_group.csv", groups=("white", "black"), n_samples=10000):
    df = pd.read_csv(path)
    df["id"] = df["id"].astype("category").cat.codes
    df = df[df["group"].isin(groups)].copy()
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    df["weights"] = np.ones(len(df)) / len(df)
    return df


def load_sbic_and_train_api(api, path="SBIC_group.csv"):
    df = pd.read_csv(path)
    train_df = df.sample(frac=0.9, random_state=42)
    api.train(train_df["text"], train_df["true_label"], train_df["group"])
    return train_df
