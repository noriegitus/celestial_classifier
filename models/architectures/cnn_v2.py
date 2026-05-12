import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# === CONFIGURACIÓN DE RUTAS (Arreglado para evitar errores de concatenación) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from scripts.utils import GalacticDataset

# Rutas de archivos
CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "train_map.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_cnn_v2.pth")

# Hiperparámetros
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.0005
PATIENCE = 5 

# =======================
# 2. TRANSFORMACIONES
# =======================
# Nota: GalacticDataset aplicará estas transformaciones al vuelo
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =======================
# 3. CARGA DE DATOS (Enfoque Profesional)
# =======================
# Cargamos el dataset completo desde el CSV
full_dataset = GalacticDataset(CSV_PATH, transform=transform_train)

# División lógica
total_size = len(full_dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Aplicamos la transformación de validación (sin aumentos) al subset de validación
val_dataset.dataset.transform = transform_val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =======================
# 4. DEFINICIÓN DEL MODELO
# =======================
class CNN_v2(nn.Module):
    def __init__(self):
        super(CNN_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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

# =======================
# 5. ENTRENAMIENTO
# =======================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = CNN_v2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # --- Bucle de Entrenamiento ---
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Bucle de Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        # Imprimir progreso por ÉPOCA (movido fuera del bucle de batches para claridad)
        print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Lógica de Early Stopping ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Guardar checkpoint intermedio si prefieres
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f" Early stopping activado en época {epoch+1}")
                break
        
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos")

    # Guardar el mejor modelo encontrado
    model.load_state_dict(best_model_wts)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Mejor modelo guardado en: {MODEL_PATH}")

if __name__ == '__main__':
    train()