import os
import shutil
from data_io import get_loaders
from torchvision.models import swin_t, Swin_T_Weights
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, train_loader, val_loader, num_classes = get_loaders()

weights = Swin_T_Weights.IMAGENET1K_V1
model = swin_t(weights=weights)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

outputs = model(images)

print("Num classes:", num_classes)
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

# cache_dir = r"C:\Users\bluni\.cache\kagglehub\datasets\nikhilshingadiya\tinyimagenet200\versions\1"

# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
#     print("Deleted:", cache_dir)
# else:
#     print("Folder does not exist:", cache_dir)