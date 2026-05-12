import pandas as pd
import random
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE = Path(__file__).resolve().parent.parent
RAW_IMG_DIR = BASE / "data" / "raw" / "images_gz2" / "images"
MAP_CSV = BASE / "data" / "raw" / "gz2_filename_mapping.csv"
LABELS_CSV = BASE / "data" / "raw" / "gz2_hart16.csv"
OUTPUT_DIR = BASE / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.80
SPLIT_RATIO = 0.8

def generate_professional_pipeline():
    print("--- Generando Mapa de Datos (Standard Pro) ---")
    
    # 1. Cargar etiquetas
    df_map = pd.read_csv(MAP_CSV)
    df_labels = pd.read_csv(LABELS_CSV)
    df = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")

    # 2. Filtrar por calidad y clasificar
    # t04_spiral... y t01_smooth... son tus columnas de confianza
    spiral_mask = df["t04_spiral_a08_spiral_debiased"] >= THRESHOLD
    elliptical_mask = df["t01_smooth_or_features_a01_smooth_debiased"] >= THRESHOLD
    
    df.loc[spiral_mask, 'label'] = 1 # Spiral
    df.loc[elliptical_mask, 'label'] = 0 # Elliptical
    
    df_final = df.dropna(subset=['label']).copy()
    df_final['path'] = df_final['asset_id'].apply(lambda x: str(RAW_IMG_DIR / f"{x}.jpg"))

    # 3. Balanceo en Memoria (Submuestreo)
    counts = df_final['label'].value_counts()
    min_samples = counts.min()
    
    balanced_df = df_final.groupby('label').sample(n=min_samples, random_state=42)
    
    # 4. Split Train/Test
    train_df = balanced_df.sample(frac=SPLIT_RATIO, random_state=42)
    test_df = balanced_df.drop(train_df.index)

    # 5. Guardar los "Mapas" (No movemos ni una imagen)
    train_df[['path', 'label']].to_csv(OUTPUT_DIR / "train_map.csv", index=False)
    test_df[['path', 'label']].to_csv(OUTPUT_DIR / "test_map.csv", index=False)

    print(f"✔ Pipeline terminado. Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"✔ Mapas guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_professional_pipeline()