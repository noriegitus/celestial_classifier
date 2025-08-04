import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 1. CONFIGURACIÓN DE RUTAS ========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v1.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "test_set_balanced")
BATCH_SIZE = 64
class_names = ['elliptical', 'spiral']

# === 2. TRANSFORMACIONES ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImagenNet mean
                         [0.229, 0.224, 0.225])  # ImagenNet std
])

# === 3. CARGAR DATOS =====================================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 4. CARGAR MODELO ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # Ajustar a tus 2 clases
model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
model.to(device)
model.eval()

# === 5. EVALUACIÓN =======================================
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === 6. REPORTES ========================================
print("📊 Reporte de clasificación:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# === 7. MATRIZ DE CONFUSIÓN =============================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - ResNet18')
plt.tight_layout()
plt.show()
