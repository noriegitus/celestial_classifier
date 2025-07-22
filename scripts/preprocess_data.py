import os
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

# === CONFIGURACIÓN DE RUTAS ===
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMG_RAW = os.path.join(BASE, "data", "raw", "images_gz2", "images")
FILENAME_MAP_CSV = os.path.join(BASE, 'data', 'raw', 'gz2_filename_mapping.csv')
LABELS_CSV = os.path.join(BASE, 'data', 'raw', 'gz2_hart16.csv')  # o .csv.gz si lo usas comprimido
OUTPUT_PATH = os.path.join(BASE, 'data', 'processed', 'train_set')

# === CATEGORÍAS A CLASIFICAR ===
CLASSES = {
    "spiral": "t04_spiral_a08_spiral_debiased",
    "elliptical": "t01_smooth_or_features_a01_smooth_debiased",
}

# === UMBRAL DE CONFIANZA ===
THERESHOLD = 0.80


def main():
    start_time = time.time()
    print("Cargando CSVs...")

    # === CARGA Y PROCESAMIENTO DE MAPEO DE NOMBRES ===
    df_map = pd.read_csv(FILENAME_MAP_CSV)
    df_map.rename(columns={"asset_id": "filename"}, inplace=True)
    df_map["filename"] = df_map["filename"].astype(str) + ".jpg"

    # Filtra solo los archivos de imagen que realmente existen
    available_files = set(os.listdir(RAW_IMG_RAW))
    df_map = df_map[df_map["filename"].isin(available_files)]

    # === CARGA DE ETIQUETAS Y UNIÓN CON LOS ARCHIVOS ===
    print("Uniendo etiquetas con nombres de archivo...")
    df_labels = pd.read_csv(LABELS_CSV)
    df = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")

    total = 0

    # === PROCESAMIENTO POR CLASE ===
    for class_name, column in CLASSES.items():
        print(f"\n Procesando clase: {class_name}")
        subset = df[df[column] >= THERESHOLD]

        class_dir = os.path.join(OUTPUT_PATH, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Función para copiar archivos individualmente
        def copy_file(row):
            filename = row["filename"]
            src_path = os.path.join(RAW_IMG_RAW, filename)
            dst_path = os.path.join(class_dir, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                return 1
            else:
                print(f"[NO ENCONTRADO] {src_path}")
                return 0

        rows = subset.to_dict(orient="records")

        # Copia paralela usando múltiples hilos
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(copy_file, rows))

        copied = sum(results)
        total += copied
        print(f"✅ {copied} imágenes copiadas a {class_dir}")

    # === FINAL ===
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nProceso completado. Total de imágenes copiadas: {total}")
    print(f"Tiempo total: {elapsed:.2f} segundos")


if __name__ == "__main__":
    main()
