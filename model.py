import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("CNR-EXT.csv")



# visualization
# remove bad last row
df = df.iloc[:-1].reset_index(drop=True)
cameras = sorted(df["camera"].unique())
current_index = 0

fig = plt.figure(figsize=(10, 10))

def show_camera(index):
    plt.clf()
    cam = cameras[index]
    cam_df = df[df["camera"] == cam].sample(n=9, random_state=42).reset_index(drop=True)

    plt.suptitle(f"Camera {cam}", fontsize=16)

    for i in range(9):
        img_path = cam_df.loc[i, "image_url"]
        avail = int(cam_df.loc[i, "available_spots"])
        total = int(cam_df.loc[i, "total_spots"])

        plt.subplot(3, 3, i + 1)
        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)
        plt.title(f"{avail}/{total}", fontsize=10)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.draw()

def on_key(event):
    global current_index
    if event.key == "right":
        current_index = (current_index + 1) % len(cameras)
        show_camera(current_index)
    elif event.key == "left":
        current_index = (current_index - 1) % len(cameras)
        show_camera(current_index)

fig.canvas.mpl_connect("key_press_event", on_key)

show_camera(current_index)
plt.show()

# transforms
train_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((375, 500)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
])

eval_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((375, 500))
])

# dataset
class ParkingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_url"]
        label = float(self.df.iloc[idx]["available_spots"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label
    
# splits
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# datasets
train_dataset = ParkingDataset(train_df.reset_index(drop=True), transform=train_transforms)
val_dataset = ParkingDataset(val_df.reset_index(drop=True), transform=eval_transforms)
test_dataset = ParkingDataset(test_df.reset_index(drop=True), transform=eval_transforms)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("\nTrain Loader:")
for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break

print("\nValidation Loader:")
for images, labels in val_loader:
    print(images.shape)
    print(labels)
    break

print("\nTest Loader:")
for images, labels in test_loader:
    print(images.shape)
    print(labels)
    break