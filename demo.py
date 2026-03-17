import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import v2

# set device (gpu if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# define the same transform used during evaluation (no randomness)
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224))
])

# define the same model architecture used during training
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

        # global average pooling reduces spatial dimensions to 1x1
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

# create two model instances (one for each prediction task)
model_spots = ConvNet().to(device)
model_cars = ConvNet().to(device)

# load trained weights for each model
model_spots.load_state_dict(torch.load("available_spots_model.pth", map_location=device))
model_cars.load_state_dict(torch.load("num_cars_model.pth", map_location=device))

# set both models to evaluation mode
model_spots.eval()
model_cars.eval()

print("models loaded successfully")

# load dataset
df = pd.read_csv("CNR-EXT.csv")
df = df.iloc[:-1].reset_index(drop=True)

# randomly select a few images for the demo
sample_df = df.sample(n=5, random_state=42).reset_index(drop=True)

# create a figure to display predictions
plt.figure(figsize=(12, 6))

for i in range(len(sample_df)):
    row = sample_df.iloc[i]

    img_path = row["image_url"]
    true_spots = row["available_spots"]
    true_cars = row["num_cars"]

    # load and preprocess image
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # run both models
    with torch.no_grad():
        pred_spots = model_spots(input_tensor).item()
        pred_cars = model_cars(input_tensor).item()

    # display image and predictions
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)

    plt.title(
        f"spots: {true_spots:.0f} → {pred_spots:.1f}\n"
        f"cars: {true_cars:.0f} → {pred_cars:.1f}",
        fontsize=10
    )

    plt.axis("off")

plt.tight_layout()
plt.show()