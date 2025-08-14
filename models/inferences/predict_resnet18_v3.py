import os
import torch
import torch.nn as nn  # ¡Esta línea faltaba!
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from torchvision.models import resnet18
import time


# ===== Configuración de rutas =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# ===== Configuración =====
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v3.pth")
IMG_TYPE = "simples" # o "complicado"
IMAGE_DIR = os.path.join(BASE_DIR, "models", "inferences", "test_images",IMG_TYPE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['elliptical', 'spiral']

# ===== Transformaciones =====
start = time.time()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== Carga del modelo =====
def load_model():
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Aquí se usaba nn sin estar definido
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ===== Resto del código permanece igual =====
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    
    return {
        'class': CLASS_NAMES[predicted_class.item()],
        'confidence': confidence.item() * 100,
        'probs': probs.cpu().numpy()[0]
    }

print("\nIniciando predicciones sobre imágenes en test_images:\n")

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            prediction = predict_image(image_path, model, transform)
            print(f"{filename} -> {prediction['class']} ({prediction['confidence']:.2f}%)")
            
            plt.figure(figsize=(8, 4))
            plt.bar(CLASS_NAMES, prediction['probs']*100, color=['blue', 'green'])
            plt.title(f"Predicción: {filename}\nClase: {prediction['class']} ({prediction['confidence']:.1f}%)")
            plt.ylabel("Probabilidad (%)")
            plt.ylim(0, 100)
            plt.show()
            
        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")

final = time.time()

print(f"Tiempo Total: {final-start} segundos")
print("\nPredicción completada.")