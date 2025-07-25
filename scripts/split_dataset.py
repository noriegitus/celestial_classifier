import os
import random
import shutil
from pathlib import Path

# CONFIGURA TUS RUTAS
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data" / "processed" / "train_set"
DEST_TRAIN = BASE_DIR / "data" / "processed" / "train_set_balanced"
DEST_TEST = BASE_DIR / "data" / "processed" / "test_set_balanced"

SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def split_class(class_name):
    src_folder = SOURCE_DIR / class_name
    images = list(src_folder.glob("*.jpg"))

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    train_dest = DEST_TRAIN / class_name
    test_dest = DEST_TEST / class_name
    ensure_dirs(train_dest, test_dest)

    for img in train_imgs:
        shutil.copy(img, train_dest / img.name)

    for img in test_imgs:
        shutil.copy(img, test_dest / img.name)

    print(f"{class_name.upper()} | Total: {len(images)} | Train: {len(train_imgs)} | Test: {len(test_imgs)}")

def main():
    print("🔄 Separando imágenes...")
    for class_name in os.listdir(SOURCE_DIR):
        if os.path.isdir(SOURCE_DIR / class_name):
            split_class(class_name)

    print("\nDivisión completada. Usa 'train_set_balanced' y 'test_set_balanced' para entrenar y evaluar.")

if __name__ == "__main__":
    main()
