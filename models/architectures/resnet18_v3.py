import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# ==================== CONFIG ====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "train_set_balanced")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "pth_files", "model_resnet18_v3.pth")

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 4
NUM_WORKERS = 4
CLASS_WEIGHTS = torch.tensor([1.2, 0.8])  # Penaliza más errores en elliptical

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ============= TRANSFORMACIONES ==============
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # =================== DATOS ====================
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # =================== MODELO ===================
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.to(device)

    # ================= ENTRENAMIENTO ==============
    class_weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    print("\nIniciando entrenamiento...\n")

    for epoch in range(NUM_EPOCHS):
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

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        duration = time.time() - epoch_start

        print(f"\U0001F4C6 Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Tiempo: {duration:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Modelo mejorado. Guardado.")
        else:
            epochs_no_improve += 1
            print("⚠️  No hubo mejora.")

        if epochs_no_improve >= PATIENCE:
            print("⛔ Early stopping activado.")
            break

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.2f} segundos, {total_time/60:.2f} minutos, {total_time/3600:.2f} horas")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
