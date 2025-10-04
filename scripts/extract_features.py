import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from pathlib import Path

# --- (Las funciones de ayuda se mantienen igual) ---
def get_dominant_color(image_path, k=1):
    try:
        img = cv2.imread(str(image_path))
        if img is None: return '#000000'
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_.astype(int)[0]
        return '#{:02x}{:02x}{:02x}'.format(dominant_color[2], dominant_color[1], dominant_color[0])
    except Exception:
        return '#000000'

def extract_image_features(file_path, source):
    try:
        filename = file_path.name
        filesize_kb = round(file_path.stat().st_size / 1024)
        folder = file_path.parent.name
        with Image.open(file_path) as img_pil:
            width_px, height_px = img_pil.size
        img_cv2 = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img_cv2 is None: return None
        average_brightness = np.mean(img_cv2)
        average_contrast = np.std(img_cv2)
        dominant_color_hex = get_dominant_color(file_path)
        return {
            'filename': filename, 'height_px': height_px, 'width_px': width_px,
            'filesize_kb': filesize_kb, 'dominant_color_hex': dominant_color_hex,
            'average_contrast': round(average_contrast, 4),
            'average_brightness': round(average_brightness, 4),
            'source': source, 'folder': folder
        }
    except Exception as e:
        print(f"Error procesando {filename}: {e}")
        return None

# --- SCRIPT PRINCIPAL (MODIFICADO) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# En lugar de una sola ruta, creamos una lista de carpetas para procesar.
SOURCES_TO_PROCESS = [
    {
        "path": PROJECT_ROOT / 'data' / 'processed' / 'test_set_balanced',
        "source_name": "Galaxy Zoo 2"
    },
    {
        "path": PROJECT_ROOT / 'data' / 'processed' / 'external_img',
        "source_name": "External Test"
    }
]

all_features = []
print("--- Iniciando extracción de metadatos de imágenes de MÚLTIPLES fuentes ---")

# Bucle principal que recorre la lista de fuentes
for source_info in SOURCES_TO_PROCESS:
    image_root_directory = source_info["path"]
    source_name = source_info["source_name"]
    
    print(f"\nProcesando fuente: '{source_name}' en la carpeta: {image_root_directory.relative_to(PROJECT_ROOT)}")
    
    image_files = list(image_root_directory.glob('**/*.jpg')) + list(image_root_directory.glob('**/*.png'))
    total_images = len(image_files)
    
    if total_images == 0:
        print("No se encontraron imágenes en esta carpeta.")
        continue
        
    print(f"Se encontraron {total_images} imágenes para procesar.")

    for i, file_path in enumerate(image_files):
        # Descomenta la siguiente línea si quieres ver el progreso imagen por imagen
        print(f"  -> Procesando [{i+1}/{total_images}]: {file_path.name}")
        features = extract_image_features(file_path, source_name)
        if features:
            all_features.append(features)

# Guardar el resultado UNIFICADO en un solo archivo
features_df = pd.DataFrame(all_features)
output_csv_path = PROJECT_ROOT / 'outputs' / 'images_metadata_master.csv'
features_df.to_csv(output_csv_path, index=False)

print(f"\nProceso completado!")
print(f"Se extrajeron metadatos de un total de {len(features_df)} imágenes.")
print(f"Archivo maestro UNIFICADO guardado en: {output_csv_path}")