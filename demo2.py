import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2

# =========================
# CONFIG: ADD YOUR IMAGES HERE
# =========================

images_info = [
    {
        "path": "mostlyEmpty.jpg",
        "true_spots": 85, #excluding corner available spots that are cut for all below
        "true_cars": 3
    },
    {
        "path": "mediumFull.jpg",
        "true_spots": 30,
        "true_cars": 21
    },
    {
        "path": "mostlyFull.jpg",
        "true_spots": 1,
        "true_cars": 68
    }
]

# =========================

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transform (same as training)
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224))
])

# =========================
# MODEL (same as training)
# =========================
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

# =========================
# LOAD MODELS
# =========================
model_spots = ConvNet().to(device)
model_cars = ConvNet().to(device)

model_spots.load_state_dict(torch.load("available_spots_best_model.pth", map_location=device))
model_cars.load_state_dict(torch.load("num_cars_best_model.pth", map_location=device))

model_spots.eval()
model_cars.eval()

print("Models loaded successfully!")

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(15, 5))

for i, item in enumerate(images_info):
    image = Image.open(item["path"]).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    true_spots = item["true_spots"]
    true_cars = item["true_cars"]

    with torch.no_grad():
        pred_spots = model_spots(input_tensor).item()
        pred_cars = model_cars(input_tensor).item()

    # clamp negatives
    pred_spots = max(0, pred_spots)
    pred_cars = max(0, pred_cars)

    # errors
    spot_error = abs(pred_spots - true_spots)
    car_error = abs(pred_cars - true_cars)

    # plot
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)

    plt.title(
        f"Image {i+1}\n\n"
        f"Spots → True: {true_spots} | Pred: {pred_spots:.1f} | Err: {spot_error:.1f}\n"
        f"Cars  → True: {true_cars} | Pred: {pred_cars:.1f} | Err: {car_error:.1f}",
        fontsize=10
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("custom_multi_demo.png")
plt.show()

print("Saved as custom_multi_demo.png")