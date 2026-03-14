import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("CNR-EXT.csv")

# remove bad last row
df = df.iloc[:-1].reset_index(drop=True)

print("Avg available spots:", df["available_spots"].mean())
print("Avg number of cars:", df["num_cars"].mean())

# visualization
cameras = sorted(df["camera"].unique())
current_index = 0

fig = plt.figure(figsize=(12, 12))

def show_camera(index):
    plt.clf()

    cam = cameras[index]

    cam_df = (
        df[df["camera"] == cam]
        .sample(n=9, random_state=42)
        .reset_index(drop=True)
    )

    plt.suptitle(f"Camera {cam}", fontsize=18)

    for i in range(9):
        img_path = cam_df.loc[i, "image_url"]
        avail = int(cam_df.loc[i, "available_spots"])
        total = int(cam_df.loc[i, "total_spots"])
        num_cars = int(cam_df.loc[i, "num_cars"])
        datetime_val = cam_df.loc[i, "datetime"]

        plt.subplot(3, 3, i + 1)

        img = Image.open(img_path).convert("RGB")
        plt.imshow(img)

        plt.title(
            f"{datetime_val}\nCars: {num_cars} | Avail: {avail} | Total: {total}",
            fontsize=9
        )

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

#show_camera(current_index)
#plt.show()

# transforms
# we use the same augmentations for both experiments
train_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((128, 128)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
])

eval_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((128, 128))
])

# dataset
# target_col lets us switch between: "available_spots" and "num_cars"
# since we are testing both
class ParkingDataset(Dataset):
    def __init__(self, dataframe, target_col, transform=None):
        self.df = dataframe
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_url"]
        label = float(self.df.iloc[idx][self.target_col])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label
    
# splits
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #input, output, kernel, stride, padding
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # kernel size, stride
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # less overfitting, faster training. reduces number of parameters
        # [128, 46, 62] -> [128, 1, 1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 128 -> 64 -> 1
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        # conv forward 
    def forward(self, x):
        # Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # reduce each channel to 1x1
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output

# function to train and test on one target column
def run_experiment(target_col):
    print("\n" + "="*50)
    print(f"Predicting: {target_col}")
    print("="*50)

    # baseline = always predict the training-set mean
    baseline_mean = train_df[target_col].mean()
    baseline_rmse = math.sqrt(np.mean((test_df[target_col] - baseline_mean) ** 2))

    print(f"Baseline mean for {target_col}: {baseline_mean:.4f}")
    print(f"Baseline RMSE for {target_col}: {baseline_rmse:.4f}")

    # datasets
    train_dataset = ParkingDataset(
        train_df.reset_index(drop=True),
        target_col=target_col,
        transform=train_transforms
    )
    val_dataset = ParkingDataset(
        val_df.reset_index(drop=True),
        target_col=target_col,
        transform=eval_transforms
    )
    test_dataset = ParkingDataset(
        test_df.reset_index(drop=True),
        target_col=target_col,
        transform=eval_transforms
    )

    # dataloaders
    # on some local environments, num_workers may cause issues. if so, set num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)
    
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

    # define a new model for this experiment
    model = ConvNet().to(device)

    # test forward pass
    images, labels = next(iter(train_loader))
    images = images.to(device)
    outputs = model(images)

    print("\nForward Pass Test")
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)


    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # training loop
    print("\nStarting Training...\n")
    epochs = 15

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_rmse = math.sqrt(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
         for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                predictions = model(images)
                loss = criterion(predictions, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_rmse = math.sqrt(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {avg_train_loss:.4f} | Train RMSE: {avg_train_rmse:.4f} | "
            f"Val MSE: {avg_val_loss:.4f} | Val RMSE: {avg_val_rmse:.4f}"
        )

    # final testing 
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            predictions = model(images)
            loss = criterion(predictions, labels)

            total_test_loss += loss.item()

    avg_test_mse = total_test_loss / len(test_loader)
    avg_test_rmse = math.sqrt(avg_test_mse)

    print("\n Test Results:")
    print(f"Test MSE: {avg_test_mse:.4f}")
    print(f"Test RMSE: {avg_test_rmse:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    return avg_test_rmse

# run experiment 1: predict empty spots
empty_rmse = run_experiment("available_spots")

# run experiment 2: predict number of cars
cars_rmse = run_experiment("num_cars")


# TODO Coding Portion:
# Predict both the number of open spots and num of cars instead of just the empty slots to see which one does better
# Have better comments for each section so the code is easier to understand
# Try to decrease RMSE (average is around 6 right now after running 10 epochs). Make sure model is not overfitting.