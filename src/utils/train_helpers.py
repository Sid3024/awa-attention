from src.config.config import hyper_param_config
from math import cos, pi

warmup_steps = hyper_param_config.total_train_steps * hyper_param_config.warmup_fraction
max_steps = hyper_param_config.total_train_steps * hyper_param_config.max_fraction
min_lr = hyper_param_config.min_lr
max_lr = hyper_param_config.max_lr

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it/warmup_steps)
    elif it > warmup_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1 + cos(pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)