# config.py
from dataclasses import dataclass

#rank lora
#logs
#groups
#dataset_D

@dataclass
class AuditConfig:
    model: str = "lora"
    size_T: int = 100
    iterations: int = 5
    epochs_sur: int = 3
    epochs_opt: int = 3
    batch_size: int = 32
    lambda_penalty: float = 0.5
    epsilon: float = 1e-2
    change: str = "experiment1"
