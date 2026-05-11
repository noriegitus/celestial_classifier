import os
import shutil
import pandas as pd
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==========================================
BASE = Path(__file__).resolve().parent.parent

# Orígenes
RAW_IMG_DIR = BASE / "data" / "raw" / "images_gz2" / "images"
MAP_CSV = BASE / "data" / "raw" / "gz2_filename_mapping.csv"
LABELS_CSV = BASE / "data" / "raw" / "gz2_hart16.csv"

# Destinos intermedios y finales
PROCESSED_ALL = BASE / "data" / "processed" / "all_images"
FINAL_TRAIN = BASE / "data" / "final" / "train"
FINAL_TEST = BASE / "data" / "final" / "test"

# Parámetros técnicos
CLASSES = {
    "spiral": "t04_spiral_a08_spiral_debiased",
    "elliptical": "t01_smooth_or_features_a01_smooth_debiased",
}
THRESHOLD = 0.80
SPLIT_RATIO = 0.8
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ==========================================
# 2. ESTACIÓN A: PREPROCESAMIENTO (Limpieza)
# ==========================================

def run_preprocessing():
    print("\n--- Estación A: Filtrando y clasificando imágenes ---")
    if PROCESSED_ALL.exists(): shutil.rmtree(PROCESSED_ALL)
    PROCESSED_ALL.mkdir(parents=True, exist_ok=True)

    df_map = pd.read_csv(MAP_CSV)
    df_map.rename(columns={"asset_id": "filename"}, inplace=True)
    df_map["filename"] = df_map["filename"].astype(str) + ".jpg"
    
    df_labels = pd.read_csv(LABELS_CSV)
    df = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")

    def copy_file(row, class_name):
        src = RAW_IMG_DIR / row["filename"]
        dst = PROCESSED_ALL / class_name / row["filename"]
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            return 1
        return 0

    for class_name, column in CLASSES.items():
        subset = df[df[column] >= THRESHOLD]
        print(f"  -> Clase '{class_name}': Encontradas {len(subset)} candidatas.")
        
        rows = subset.to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda r: copy_file(r, class_name), rows))
        print(f"  ✔ Copiadas {sum(results)} imágenes de {class_name}.")

# ==========================================
# 3. ESTACIÓN B: PREPARACIÓN (Balanceo y Split)
# ==========================================

def run_preparation():
    print("\n--- Estación B: Balanceando y dividiendo en Train/Test ---")
    
    for d in [FINAL_TRAIN, FINAL_TEST]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for class_dir in PROCESSED_ALL.iterdir():
        if not class_dir.is_dir(): continue
        
        class_name = class_dir.name
        files = list(class_dir.glob("*.jpg"))
        random.shuffle(files)

        # Dividir
        split_idx = int(len(files) * SPLIT_RATIO)
        train_pool = files[:split_idx]
        test_pool = files[split_idx:]

        # Guardar temporalmente para balancear después
        train_data[class_name] = train_pool
        test_data[class_name] = test_pool

    # Balanceo (Submuestreo)
    min_count = min(len(f) for f in train_data.values())
    print(f"  -> Punto de balanceo (min imágenes): {min_count}")

    for class_name in train_data:
        # Train (Balanceado)
        for f in train_data[class_name][:min_count]:
            dest = FINAL_TRAIN / class_name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
        
        # Test (Sin balancear, para ver realidad)
        for f in test_data[class_name]:
            dest = FINAL_TEST / class_name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
        
        print(f"  ✔ Clase '{class_name}' lista.")

# ==========================================
# 4. EJECUCIÓN TOTAL
# ==========================================

train_data = {}
test_data = {}

if __name__ == "__main__":
    start = time.time()
    run_preprocessing()
    run_preparation()
    end = time.time()
    print(f"\nPipeline completado con éxito en {round(end - start, 2)} segundos.")