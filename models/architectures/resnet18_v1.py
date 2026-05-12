# train_resnet18_v1_optimized.py

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# === SOLUCIÓN AL ERROR DE IMPORTACIÓN ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from scripts.utils import GalacticDataset

# === 1. CONFIGURACIÓN DE RUTAS ========================
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "train_map.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v1.pth")

# === 2. HYPERPARÁMETROS ================================
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
PATIENCE = 3 


def main():
    # === 3. TRANSFORMACIONES ================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # === 4. CARGA DE DATOS (Standard Pro) ====================
    # Usamos GalacticDataset en lugar de ImageFolder
    dataset = GalacticDataset(CSV_PATH, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    # === 5. MODELO RESNET18 PREENTRENADO ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Usando dispositivo: {device}")

    # Actualizado a la sintaxis moderna de Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Reemplazamos la capa final para 2 clases (Spiral vs Elliptical)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # === 6. FUNCIONES DE ENTRENAMIENTO =======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === 7. ENTRENAMIENTO CON EARLY STOPPING ================
    print(f"Entrenando ResNet18 con {len(dataset)} imágenes...\n")
    best_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {time.time() - epoch_start:.2f}s")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Asegurar que la carpeta exista antes de guardar
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(" ✅ Modelo mejorado y guardado.")
        else:
            epochs_no_improve += 1
            print(" ⚠️ Sin mejora.")

        if epochs_no_improve >= PATIENCE:
            print(" ⛔ Early stopping activado.")
            break

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos")

if __name__ == "__main__":
    # Necesario para el manejo de múltiples procesos en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()