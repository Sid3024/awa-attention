import os
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import Swin_T_Weights


@dataclass
class DataConfig:
    data_path: str
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True


def build_transform(is_train):
    weights = Swin_T_Weights.IMAGENET1K_V1
    return weights.transforms()


def build_dataset(is_train, config):
    split = "train" if is_train else "val"
    root = os.path.join(config.data_path, split)
    transform = build_transform(is_train)

    dataset = datasets.ImageFolder(root=root, transform=transform)
    num_classes = len(dataset.classes)
    return dataset, num_classes


def build_loader(config):
    train_dataset, num_classes = build_dataset(True, config)
    val_dataset, _ = build_dataset(False, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_dataset, val_dataset, train_loader, val_loader, num_classes