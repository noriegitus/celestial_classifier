
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# === 1. CONFIGURACIÓN =========================
MODEL_PATH = "models/model_v1.pth"
IMAGE_PATH = "data/processed/test_set/16.jpg"
class_names = ['elliptical', 'spiral']

# === 2. TRANSFORMACIÓN DE IMAGEN ==============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === 3. DEFINICIÓN DE LA ARQUITECTURA ==========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# === 4. CARGAR MODELO ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === 5. CARGAR IMAGEN ===========================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# === 6. HACER PREDICCIÓN ========================
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    class_label = class_names[predicted_class]

print(f"Clase predicha: {class_label}")
