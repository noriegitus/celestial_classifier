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

# ==================== CONFIG ====================
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "train_map.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v2.pth")

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3
NUM_WORKERS = 4


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ============= TRANSFORMACIONES =================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ==================== DATOS =====================
    # Cambio a GalacticDataset profesional
    dataset = GalacticDataset(CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ==================== MODELO ====================
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Congelar todas las capas (Feature Extraction)
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar capa final (esta sí tendrá requires_grad=True por defecto)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model.to(device)

    # =================== ENTRENAMIENTO ===============
    criterion = nn.CrossEntropyLoss()
    
    # IMPORTANTE: Solo pasamos al optimizador los parámetros de model.fc
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    print(f"Iniciando entrenamiento ResNet18 v2 (Fine-tuning capa final)...")
    print(f"Imágenes totales: {len(dataset)}\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        duration = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Tiempo: {duration:.2f}s")

        # Lógica de mejora y Early Stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" ✅ Modelo mejorado. Guardado.")
        else:
            epochs_no_improve += 1
            print(" ⚠️ No hubo mejora.")

        if epochs_no_improve >= PATIENCE:
            print(" ⛔ Early stopping activado.")
            break

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()