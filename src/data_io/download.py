import os
import shutil
from pathlib import Path

import kagglehub
from torchvision import datasets
from torchvision.models import Swin_T_Weights


def setup_download(download=True, verify=True):
    if download:
        path = kagglehub.dataset_download("nikhilshingadiya/tinyimagenet200")
    else:
        raise ValueError("download=False is not supported unless you manually provide the path.")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

    dataset_root = os.path.join(path, "tiny-imagenet-200")
    fixed_root = os.path.join(DEFAULT_DATA_DIR, "tiny-imagenet-fixed")

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Could not find raw Tiny ImageNet folder at: {dataset_root}")

    fixed_train = os.path.join(fixed_root, "train")
    fixed_val = os.path.join(fixed_root, "val")

    if os.path.exists(fixed_train) and os.path.exists(fixed_val):
        print(f"Fixed dataset already exists at: {fixed_root}")
    else:
        print("Preparing Tiny ImageNet into ImageFolder format...")
        prepare(dataset_root, fixed_root)
        print(f"Saved fixed dataset to: {fixed_root}")


    if verify:
        weights = Swin_T_Weights.IMAGENET1K_V1
        transform = weights.transforms()

        train_dataset = datasets.ImageFolder(
            root=os.path.join(fixed_root, "train"),
            transform=transform
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(fixed_root, "val"),
            transform=transform
        )

        print("Verification successful.")
        print("Train classes:", len(train_dataset.classes))
        print("Train images:", len(train_dataset))
        print("Val images:", len(val_dataset))
        print("First 5 classes:", train_dataset.classes[:5])

    return fixed_root


def prepare(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    train_src = src_root / "train"
    val_src = src_root / "val"
    val_images_src = val_src / "images"
    val_annotations = val_src / "val_annotations.txt"

    train_dst = dst_root / "train"
    val_dst = dst_root / "val"

    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    # Fix train split
    for class_dir in train_src.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        src_images_dir = class_dir / "images"
        dst_class_dir = train_dst / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in src_images_dir.iterdir():
            if img_file.is_file():
                dst_file = dst_class_dir / img_file.name
                if not dst_file.exists():
                    shutil.copy2(img_file, dst_file)

    # Fix val split
    with open(val_annotations, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0]
            class_name = parts[1]

            src_img = val_images_src / img_name
            dst_class_dir = val_dst / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            if src_img.exists():
                dst_file = dst_class_dir / img_name
                if not dst_file.exists():
                    shutil.copy2(src_img, dst_file)

if __name__ == "__main__":
    setup_download()