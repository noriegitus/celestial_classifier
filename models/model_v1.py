# celestial_classifier_train_optimized.py

import os
import time
import torch
import torch.nn as nn   
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    # 1. PATH CONFIG -------------------------------------------
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data", "processed", "train_set_balanced")
    MODEL_PATH = os.path.join(BASE, "models", "model_v1.pth")

    # 2. HYPERPARAMETERS ---------------------------------------
    BATCH_SIZE = 64 
    EPOCHS = 5
    LR = 0.001

    # 3. IMAGE TRANSFORMATIONS ----------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  
    ])

    # 4. LOAD DATASET WITH ImageFolder ---------------------------
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Configura múltiples workers para carga paralela de datos
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)  # Optimización clave

    # 5. DEFINE A SIMPLE CNN -------------------------------------
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
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
                nn.Linear(128, 2)  # 2 clases
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x

    # 6. DEVICE, MODEL, LOSS AND OPTIMIZER -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 7. TRAINING ------------------------------------------------
    print("\nEntrenando el modelo...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} procesado")

        avg_loss = running_loss / len(dataloader)
        print(f"🧠 Época {epoch + 1}/{EPOCHS} | Pérdida Promedio: {avg_loss:.4f} | Tiempo: {time.time() - epoch_start:.2f}s")

    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.2f} segundos.")

    # 8. SAVE TRAINED MODEL -------------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()
