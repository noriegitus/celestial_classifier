import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- CONFIGURACIÓN ---
# Resuelve la ruta base del proyecto (asume que este script está en una carpeta como 'scripts')
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Directorio de origen con TODAS tus imágenes (antes de dividir)
SOURCE_DIR = BASE_DIR / "data" / "processed" / "all_images" # <--- AJUSTA ESTA RUTA

# 2. Directorios de destino para los nuevos conjuntos
DEST_TRAIN = BASE_DIR / "data" / "final" / "train"
DEST_TEST = BASE_DIR / "data" / "final" / "test"

# 3. Parámetros
SPLIT_RATIO = 0.8  # 80% para entrenamiento, 20% para prueba
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def copy_files(file_list, dest_folder):
    """Copia una lista de archivos a una carpeta de destino."""
    for src_path in file_list:
        dest_path = dest_folder / src_path.name
        shutil.copy(src_path, dest_path)

def main():
    """
    Ejecuta el proceso completo:
    1. Divide el dataset original en entrenamiento y prueba.
    2. Balancea (por submuestreo) ÚNICAMENTE el conjunto de entrenamiento.
    3. Copia los archivos a las carpetas finales.
    """
    print("Iniciando la preparación del dataset...")

    # --- PASO 1: RECOLECTAR Y DIVIDIR LAS IMÁGENES POR CLASE ---
    print(f"Leyendo imágenes desde: {SOURCE_DIR}")
    
    # Validar que el directorio de origen exista
    if not SOURCE_DIR.is_dir():
        print(f"ERROR: El directorio de origen '{SOURCE_DIR}' no existe. Por favor, ajústalo.")
        return

    # Usamos defaultdict para facilitar la agrupación
    train_files_by_class = defaultdict(list)
    test_files_by_class = defaultdict(list)
    
    class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"ERROR: No se encontraron carpetas de clases en '{SOURCE_DIR}'.")
        return

    print("\n--- División Inicial (Train/Test) ---")
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        train_files_by_class[class_name] = train_imgs
        test_files_by_class[class_name] = test_imgs
        
        print(f"Clase '{class_name}': {len(train_imgs)} train | {len(test_imgs)} test")


    # --- PASO 2: BALANCEAR EL CONJUNTO DE ENTRENAMIENTO ---
    print("\n--- Balanceando el conjunto de entrenamiento (undersampling) ---")
    
    # Encontrar la clase con el menor número de imágenes en el conjunto de entrenamiento
    if not train_files_by_class:
        print("No hay archivos de entrenamiento para balancear.")
        return
        
    min_count = min(len(files) for files in train_files_by_class.values())
    print(f"Tamaño objetivo para cada clase de entrenamiento: {min_count} imágenes")

    balanced_train_files = defaultdict(list)
    for class_name, files in train_files_by_class.items():
        # Tomamos solo 'min_count' imágenes de cada clase
        balanced_train_files[class_name] = files[:min_count]


    # --- PASO 3: COPIAR ARCHIVOS A LOS DIRECTORIOS FINALES ---
    print("\n--- Creando directorios finales y copiando archivos ---")
    
    # Limpiar directorios de destino si existen para evitar duplicados
    if DEST_TRAIN.exists():
        shutil.rmtree(DEST_TRAIN)
    if DEST_TEST.exists():
        shutil.rmtree(DEST_TEST)

    # Copiar conjunto de entrenamiento balanceado
    for class_name, files in balanced_train_files.items():
        dest_class_dir = DEST_TRAIN / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        copy_files(files, dest_class_dir)
        print(f"[TRAIN] Copiados {len(files)} archivos para la clase '{class_name}'")

    # Copiar conjunto de prueba (sin balancear)
    for class_name, files in test_files_by_class.items():
        dest_class_dir = DEST_TEST / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        copy_files(files, dest_class_dir)
        print(f"[TEST] Copiados {len(files)} archivos para la clase '{class_name}'")


    print("\n¡Proceso completado!")
    print(f"Conjunto de entrenamiento balanceado listo en: {DEST_TRAIN}")
    print(f"Conjunto de prueba listo en: {DEST_TEST}")


if __name__ == "__main__":
    main()