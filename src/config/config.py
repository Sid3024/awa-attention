from dataclasses import dataclass, field

@dataclass
class MyConfig:
    run_training: bool = True
    run_validation: bool = True

my_config = MyConfig()


@dataclass
class DataConfig:
    data_path: str = "src/data/tiny-imagenet-fixed"
    gpu_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

data_config = DataConfig()

@dataclass
class HyperParamConfig:
    total_batch_size: int = 128
    total_train_steps: int = 80000
    warmup_fraction: float = 0.1
    max_fraction: float = 0.8
    min_lr: float = 1e-3
    max_lr: float = 1e-4
    constant_lr: float = 1e-3
    adamw_weight_decay: float = 1e-2


hyper_param_config = HyperParamConfig()

@dataclass
class ModelConfig:
    img_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    depths: list = field(default_factory=lambda: [2,2,6,2])
    num_heads: list = field(default_factory=lambda: [3, 6, 12, 24])
    global_num_heads: list = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: float = None
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    fused_window_process: bool = False

model_config = ModelConfig()

@dataclass
class RunConfig():
    round: str = "initial"
    iter_id: str = "002"
    MODEL_FILE_PATH: str = None
    LOG_FILE_PATH: str = None
    model_save_interval: int = int(1e4)
    num_classes: int = 200
    val_interval: int = 10000
    model_version: str = "v1"
    dry_run: bool = False
    

run_config = RunConfig()