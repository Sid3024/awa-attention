from dataclasses import dataclass

@dataclass
class MyConfig:
    run_training: bool = True
    run_validation: bool = True

my_config = MyConfig()


@dataclass
class DataConfig:
    data_path: str = "src/data/tiny-imagenet-fixed"
    gpu_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True

data_config = DataConfig()

@dataclass
class HyperParamConfig:
    total_batch_size: int = 1
    total_train_steps: int = int(1e5)
    warmup_fraction: float = 0.1
    max_fraction: float = 0.8
    min_lr: float = 1e-3
    max_lr: float = 1e-4
    constant_lr: float = 1e-3
    adamw_weight_decay: float = 1e-2


hyper_param_config = HyperParamConfig()

@dataclass
class ModelConfig:
    pass

model_config = ModelConfig()

@dataclass
class RunConfig():
    MODEL_FILE_PATH: str = None
    LOG_FILE_PATH: str = None
    model_save_interval: int = int(1e4)
    num_classes: int = 200
    val_interval: int = 10000
    

run_config = RunConfig()