from .Download import setup_download
from .DataLoader import DataConfig, build_loader


def get_loaders(batch_size=32, num_workers=0, pin_memory=True):
    fixed_root = setup_download()

    config = DataConfig(
        data_path=fixed_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return build_loader(config)