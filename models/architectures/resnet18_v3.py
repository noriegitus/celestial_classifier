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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v3.pth")

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 4
NUM_WORKERS = 4
# Pesos: [elliptical, spiral] -> Penaliza más el error en elípticas (1.2)
CLASS_WEIGHTS = torch.tensor([1.2, 0.8]) 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ============= TRANSFORMACIONES AVANZADAS ==============
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Útil en galaxias ya que no hay "arriba" o "abajo" en el espacio
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # =================== DATOS (Vía CSV) ====================
    dataset = GalacticDataset(CSV_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # =================== MODELO ===================
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Congelar capas base
    for param in model.parameters():
        param.requires_grad = False

    # Nueva capa de salida
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.to(device)

    # ================= ENTRENAMIENTO ==============
    class_weights = CLASS_WEIGHTS.to(device)
    # Aplicamos los pesos a la función de pérdida
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizador solo para la capa final
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce el LR un 30% cada 5 épocas para afinar el entrenamiento
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    print(f"Iniciando ResNet18 v3 con Pesos de Clase y Scheduler...")
    print(f"Dataset: {len(dataset)} imágenes | Pesos: {CLASS_WEIGHTS.tolist()}\n")

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

        # Actualizar el Learning Rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_loss = running_loss / len(dataset)
        duration = time.time() - epoch_start

        print(f"📅 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f} - Tiempo: {duration:.2f}s")

        # Lógica de guardado y Early Stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" ✅ Modelo mejorado y guardado.")
        else:
            epochs_no_improve += 1
            print(" ⚠️ Sin mejora.")

        if epochs_no_improve >= PATIENCE:
            print(f" ⛔ Early stopping tras {PATIENCE} épocas sin mejora.")
            break

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos")
    
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()