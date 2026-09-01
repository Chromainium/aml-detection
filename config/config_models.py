from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 0.0005
    weight_decay: float = 5e-4
    patience: int = 20
    batch_size: int = None
    pos_weight_clip: tuple = (1.0, 20.0)


@dataclass
class MLPConfig:
    hidden_channels: int = 128
    dropout: float = 0.3
    num_layers: int = 2


@dataclass
class GNNConfig:
    hidden_channels: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    conv_type: str = 'GCN'


@dataclass
class XGBoostConfig:
    max_depth: int = 8
    n_estimators: int = 500
    learning_rate: float = 0.05
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
