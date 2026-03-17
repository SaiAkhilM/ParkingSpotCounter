import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import v2

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# transform (no randomness)
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224))
])

# model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        return self.fc2(x)

# load models
model_spots = ConvNet().to(device)
model_cars = ConvNet().to(device)

model_spots.load_state_dict(torch.load("available_spots_best_model.pth", map_location=device))
model_cars.load_state_dict(torch.load("num_cars_best_model.pth", map_location=device))

model_spots.eval()
model_cars.eval()

print("best models loaded successfully")

# dataset
df = pd.read_csv("CNR-EXT.csv")
df = df.iloc[:-1].reset_index(drop=True)

sample_df = df.sample(n=5, random_state=42).reset_index(drop=True)

plt.figure(figsize=(12, 6))

for i in range(len(sample_df)):
    row = sample_df.iloc[i]

    img_path = row["image_url"]
    true_spots = row["available_spots"]
    true_cars = row["num_cars"]

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_spots = model_spots(input_tensor).item()
        pred_cars = model_cars(input_tensor).item()

    # clamp negatives
    pred_spots = max(0, pred_spots)
    pred_cars = max(0, pred_cars)

    # compute error
    spot_error = abs(pred_spots - true_spots)
    car_error = abs(pred_cars - true_cars)

    # plot
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)

    plt.title(
        f"Available Spots\n"
        f"True: {true_spots:.0f} | Predicted: {pred_spots:.1f} | Error: {spot_error:.1f}\n"
        f"Parked Cars\n"
        f"True: {true_cars:.0f} | Predicted: {pred_cars:.1f} | Error: {car_error:.1f}",
        fontsize=9
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("demo_results.png")
print("saved demo_results.png")