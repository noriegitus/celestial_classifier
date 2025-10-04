import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import os
import pandas as pd

# --- MODELO CNN (definición local) ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 56 * 56, 128), nn.ReLU(), nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# --- CONFIGURACIÓN ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_FOLDER = PROJECT_ROOT / 'models' / 'pth_files'
DATA_FOLDER = PROJECT_ROOT / 'data' / 'processed' / 'external_img'
OUTPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'csvs'
OUTPUT_FOLDER.mkdir(exist_ok=True)

CLASS_NAMES = ['elliptical', 'spiral'] # Asegúrate que el orden es correcto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- CARGA DE DATOS EXTERNOS ---
outside_dataset = datasets.ImageFolder(DATA_FOLDER, transform=transform)
outside_dataloader = DataLoader(outside_dataset, batch_size=8, shuffle=False)

# --- Lista de modelos a evaluar ---
models_to_evaluate = {
    "cnn_v1": MODELS_FOLDER / "model_cnn_v1.pth",
    "cnn_v2": MODELS_FOLDER / "model_cnn_v2.pth",
    "resnet18_v1": MODELS_FOLDER / "model_resnet18_v1.pth",
    "resnet18_v2": MODELS_FOLDER / "model_resnet18_v2.pth",
    "resnet18_v3": MODELS_FOLDER / "model_resnet18_v3.pth"
}

# --- BUCLE DE EVALUACIÓN ---
print("--- Iniciando evaluación de imágenes externas ---")
for model_name, model_path in models_to_evaluate.items():
    print(f"\nEvaluando con el modelo: {model_name}...")
    
    # Cargar la arquitectura correcta
    if "cnn" in model_name:
        model = CNN()
    else: # ResNet
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs, _ in outside_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            for i in range(len(inputs)):
                predictions.append({
                    "image": os.path.basename(outside_dataset.samples[i][0]),
                    "predicted": CLASS_NAMES[preds[i].item()],
                    "confidence": confs[i].item()
                })
    
    # Guardar predicciones en un CSV
    df_preds = pd.DataFrame(predictions)
    output_path = OUTPUT_FOLDER / f"{model_name}_predictions_outside.csv"
    df_preds.to_csv(output_path, index=False)
    print(f"✔ Predicciones guardadas en: {output_path.relative_to(PROJECT_ROOT)}")

print("\n--- Evaluación de imágenes externas completada ---")