import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.save_metrics_to_csv import save_metrics_to_csv
from scripts.save_predictions_to_csv import save_predictions_to_csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============ CONFIGURACIÓN ============
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v3.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "test_set_balanced")
BATCH_SIZE = 64
CLASS_NAMES = ['elliptical', 'spiral']

# ============ DISPOSITIVO ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ============ TRANSFORMACIÓN ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============ CARGA DE DATOS ============
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============ CARGA DE MODELO ============
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 clases: elliptical y spiral
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

predictions = []


# ============ EVALUACIÓN ============
y_true = []
y_pred = []
predictions = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        for i in range(len(labels)):
            predictions.append({
                "image": dataset.samples[i][0].split(os.sep)[-1],
                "predicted": CLASS_NAMES[preds[i].item()],
                "confidence": confs[i].item()
            })

# ============ REPORTE ============
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ============ MATRIZ DE CONFUSIÓN ============
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión - ResNet18 v3")
plt.tight_layout()
plt.show()

# === Guardar métricas ===
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "f1_score": f1_score(y_true, y_pred)
}
save_metrics_to_csv("resnet18_v3", metrics)
save_predictions_to_csv("resnet18_v3", predictions)
