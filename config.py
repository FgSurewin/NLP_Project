from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataModuleConfig:
    file_path: str
    batch_size: int
    num_workers: int
    random_state: int
    max_length: int


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int
    dev_run: bool
    num_classes: int
    needed_labels_file_path: str


@dataclass
class WandbConfig:
    wandb_project: str
    wandb_mode: str


@dataclass
class Config:
    data_module: DataModuleConfig
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig
