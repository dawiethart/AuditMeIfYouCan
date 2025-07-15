# main.py
import os

import wandb
from audit_run import AuditRunner
from blackbox_api import BlackBoxAPI
from config import AuditConfig
from data_loader import load_jigsaw_subset, load_sbic_and_train_api
from surrogate_model import (compute_group_auc_difference,
                             load_lora_bert_surrogate, predict_with_model,
                             train_surrogate)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 1
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To stop tokenizer Warning, is set to false anyway


def main():

    # === Load audit dataset D ===
    dataset_D = load_jigsaw_subset()

    # === Train black-box ===
    api = BlackBoxAPI()
    _ = load_sbic_and_train_api(api, path="SBIC_group.csv")

    # === Define config ===
    config = AuditConfig(
        model="lora",
        size_T=100,
        iterations=5,
        epochs_sur=1,
        epochs_opt=1,
        batch_size=64,
        lambda_penalty=0.5,
        epsilon=1e-2,
        change="test_run",
    )

    wandb.init(project="active-fairness-audit", config=config.__dict__)

    runner = AuditRunner(
        dataset_D=dataset_D,
        black_box_api_fn=api.predict_scores,
        surrogate_model_loader=load_lora_bert_surrogate,
        train_surrogate_fn=train_surrogate,
        predict_fn=predict_with_model,
        compute_group_auc_diff_fn=compute_group_auc_difference,
        config=config,
    )

    runner.run()


if __name__ == "__main__":
    main()
