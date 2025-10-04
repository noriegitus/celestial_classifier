import pandas as pd
from pathlib import Path
import os

def enrich_external_predictions(predictions_path, truth_df, output_path):
    """
    Toma un CSV de predicciones externas, lo une con el archivo de verdad manual
    y guarda un nuevo CSV enriquecido.
    """
    try:
        print(f"Procesando: {os.path.basename(predictions_path)}...")
        df_preds = pd.read_csv(predictions_path)
        
        # Renombrar la columna 'image' a 'filename' para unirla
        df_preds.rename(columns={'image': 'filename'}, inplace=True)

        # Unir las predicciones con el archivo de verdad usando 'filename'
        df_merged = pd.merge(df_preds, truth_df, on='filename', how='left')

        # Crear la nueva columna 'is_correct'
        df_merged['is_correct'] = df_merged['predicted'].str.strip() == df_merged['true_label'].str.strip()

        # Limpiar y reordenar las columnas finales
        final_columns = ['filename', 'true_label', 'predicted_label', 'confidence', 'is_correct']
        df_merged.rename(columns={'predicted': 'predicted_label'}, inplace=True)
        
        df_final = df_merged[final_columns]
        
        # Guardar el resultado
        df_final.to_csv(output_path, index=False)
        print(f"✔ ¡Éxito! Archivo enriquecido guardado en: {output_path.relative_to(Path.cwd())}")

    except Exception as e:
        print(f"✖ ERROR al procesar {os.path.basename(predictions_path)}: {e}")

# --- SCRIPT PRINCIPAL ---

# 1. Definir rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRUTH_FILE = PROJECT_ROOT / 'data' / 'raw' / 'external_images_labels.csv'
INPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'csvs'
OUTPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'processed_data'
OUTPUT_FOLDER.mkdir(exist_ok=True)

# 2. Cargar tu archivo de verdad para las imágenes externas
try:
    print(f"Cargando archivo de verdad desde: {TRUTH_FILE.relative_to(Path.cwd())}")
    truth_dataframe = pd.read_csv(TRUTH_FILE, usecols=['filename', 'true_label'])
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo '{TRUTH_FILE}'. Revisa la ruta.")
    exit()

# 3. Lista de los archivos de predicciones externas que necesitas procesar
prediction_files = [
    'cnn_v1_predictions_outside.csv',
    'cnn_v2_predictions_outside.csv',
    'resnet18_v1_predictions_outside.csv',
    'resnet18_v2_predictions_outside.csv',
    'resnet18_v3_predictions_outside.csv'
]

# 4. Bucle para procesar cada archivo
print("\n--- Iniciando proceso de enriquecimiento de CSVs EXTERNOS ---")
for filename in prediction_files:
    input_path = INPUT_FOLDER / filename
    output_path = OUTPUT_FOLDER / filename.replace('.csv', '_ENRICHED.csv')
    enrich_external_predictions(input_path, truth_dataframe, output_path)

print("\n--- Proceso completado ---")