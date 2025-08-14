import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import time

# ===== Agregar la ruta base al sys.path si es necesario =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# ===== IMPORTAR EL MODELO CORRECTAMENTE =====
from models.architectures.cnn_v2 import CNN_v2  

# ===== CONFIGURACIÓN =====
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_cnn_v2.pth")
IMG_TYPE = "simples" # o "simples"
IMAGE_DIR = os.path.join(BASE_DIR, "models", "inferences", "test_images",IMG_TYPE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['elliptical', 'spiral']

# ===== TRANSFORMACIONES =====
start = time.time()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== CARGA DEL MODELO =====
model = CNN_v2().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== PREDICCIÓN MULTI-IMAGEN =====
print("Predicciones sobre imágenes en test_images:\n")

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_DIR, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        predicted_label = CLASS_NAMES[predicted_class.item()]
        confidence_percent = confidence.item() * 100

        print(f"{filename} -> {predicted_label} ({confidence_percent:.2f}%)")

final = time.time()

print(f"Tiempo Total: {final-start} segundos")
print("\Predicción completada.")
