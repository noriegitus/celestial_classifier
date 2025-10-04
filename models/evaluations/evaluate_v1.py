import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Bloque para importar funciones de otras carpetas
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.save_metrics_to_csv import save_metrics_to_csv
from scripts.save_predictions_to_csv import save_predictions_to_csv

# ============ CONFIGURACIÓN ============
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_cnn_v1.pth") # <-- CAMBIO AQUÍ
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "test_set_balanced")
BATCH_SIZE = 64
CLASS_NAMES = ['elliptical', 'spiral']

# ============ DISPOSITIVO ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ============ TRANSFORMACIONES ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Generalmente las transformaciones de aumento de datos no se aplican en el set de test
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ============ DATASET Y DATALOADER ============
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============ MODELO ============
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
            nn.Linear(128, len(CLASS_NAMES))
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ============ CARGAR MODELO ============
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ============ EVALUACIÓN ============
y_true = []
y_pred = []
predictions_list = []
file_index = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        # Guardar detalles de cada predicción (CORREGIDO)
        for i in range(len(labels)):
            predictions_list.append({
                "image": os.path.basename(dataset.samples[file_index][0]),
                "predicted": CLASS_NAMES[preds[i].item()],
                "confidence": confs[i].item()
            })
            file_index += 1

# ============ RESULTADOS ============
print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Guardar métricas como CSV
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average='binary'),
    "recall": recall_score(y_true, y_pred, average='binary'),
    "f1_score": f1_score(y_true, y_pred, average='binary')
}
save_metrics_to_csv("cnn_v1", metrics) # <-- CAMBIO AQUÍ

# ============ MATRIZ DE CONFUSIÓN ============
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión - CNN v1") # <-- CAMBIO AQUÍ
plt.tight_layout()
plt.show()

save_predictions_to_csv("cnn_v1", predictions_list) # <-- CAMBIO AQUÍ

print("\nMétricas y predicciones para CNN v1 guardadas exitosamente.")