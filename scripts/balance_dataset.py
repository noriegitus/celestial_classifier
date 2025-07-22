import os
import shutil
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time


# PATHS
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE, "data", "processed", "train_set")
BALANCED_DIR = os.path.join(BASE, "data", "processed", "train_set_balanced")
CLASSES = ["spiral", "elliptical"]

# Obtener conteo de imágenes por clase
def contar_imagenes_por_clase():
    conteo = {}
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_DIR, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        conteo[class_name] = files
    return conteo

# Función para copiar imágenes en paralelo
def copiar_imagenes(ruta):
    src, dst = ruta
    if not os.path.exists(dst):  # evita copiar de más
        shutil.copy(src, dst)

def main():
    print("Contando imágenes en cada clase...")
    start_time = time.time()
    imagenes_por_clase = contar_imagenes_por_clase()

    # Determinar tamaño mínimo para balancear
    min_count = min(len(imagenes_por_clase[clase]) for clase in CLASSES)
    print(f"Tamaño objetivo para cada clase: {min_count} imágenes")

    # Crear carpeta destino si no existe
    os.makedirs(BALANCED_DIR, exist_ok=True)

    # Preparar rutas para copiar
    tareas_copia = []

    for clase in CLASSES:
        origen_clase = os.path.join(DATASET_DIR, clase)
        destino_clase = os.path.join(BALANCED_DIR, clase)
        os.makedirs(destino_clase, exist_ok=True)

        archivos = imagenes_por_clase[clase][:min_count]

        for archivo in archivos:
            src = os.path.join(origen_clase, archivo)
            dst = os.path.join(destino_clase, archivo)
            tareas_copia.append((src, dst))

    print(f"Copiando {len(tareas_copia)} imágenes en paralelo...")

    # Copiar imágenes usando 16 hilos
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(copiar_imagenes, tareas_copia)

    end_time = time.time()
    tiempo = round(end_time - start_time, 2)
    print("Dataset balanceado creado en:", BALANCED_DIR)
    print("Tiempo total:", tiempo, "segundos")

if __name__ == "__main__":
    main()
