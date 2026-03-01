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

df = pd.read_csv("CNR-EXT.csv")

# Remove bad last row
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

