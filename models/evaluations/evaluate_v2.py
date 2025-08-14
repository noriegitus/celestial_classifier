import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from outputs.save_metrics_to_csv import save_metrics_to_csv
from outputs.save_predictions_to_csv import save_predictions_to_csv

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1. CONFIGURACIÓN =====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_cnn_v2.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "test_set_balanced")
BATCH_SIZE = 64
class_names = ['elliptical', 'spiral']

# === 2. TRANSFORMACIONES ==================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# === 3. DATASET Y DATALOADER ==============
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === 4. MODELO ============================
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

# === 5. CARGAR MODELO =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# === 6. EVALUACIÓN ========================
y_true = []
y_pred = []
predictions = []


with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probs, 1)

        for i in range(len(labels)):
            predictions.append({
                "image": dataset.samples[i][0].split(os.sep)[-1],
                "predicted": class_names[predicted_classes[i].item()],
                "confidence": confidences[i].item()
        })

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === 7. RESULTADOS ========================
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# === Guardar métricas como CSV ===
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "f1_score": f1_score(y_true, y_pred)
}

save_metrics_to_csv("cnn_v2", metrics) 

# === 8. MATRIZ DE CONFUSIÓN ===============
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

save_predictions_to_csv("cnn_v2", predictions)

# === 9. PAUSA MANUAL ========================
input("\nPresiona Enter para salir...")