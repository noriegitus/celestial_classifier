import pandas as pd
from pathlib import Path
import os

def get_simple_class(gz2_class):
    """
    Clasifica como 'elliptical' o 'spiral'. Devuelve None si no es ninguna de las dos,
    para poder filtrar estas filas más adelante.
    """
    gz2_class_upper = str(gz2_class).upper()
    if gz2_class_upper.startswith('E'):
        return 'elliptical'
    if gz2_class_upper.startswith('S'):
        return 'spiral'
    # Esta es la corrección lógica para no sobre-contar 'irregular'.
    return None

def enrich_prediction_file(predictions_path, truth_df, output_path):
    try:
        print(f"Procesando: {os.path.basename(predictions_path)}...")
        df_preds = pd.read_csv(predictions_path)
        df_preds.rename(columns={'image': 'filename'}, inplace=True)

        # Unimos las predicciones con la tabla de verdad ya procesada
        df_merged = pd.merge(df_preds, truth_df, on='filename', how='left')
        
        # Eliminamos filas que no encontraron correspondencia en el archivo de verdad
        df_merged.dropna(subset=['true_label'], inplace=True)
        
        df_merged.rename(columns={'predicted': 'predicted_label'}, inplace=True)
        
        df_merged['is_correct'] = df_merged['predicted_label'].str.strip().str.lower() == df_merged['true_label'].str.strip().str.lower()
        
        final_columns = ['filename', 'true_label', 'predicted_label', 'confidence', 'is_correct']
        df_final = df_merged[final_columns]
        
        df_final.to_csv(output_path, index=False)
        print(f"✔ ¡Éxito! Archivo enriquecido guardado en: {output_path.relative_to(Path.cwd())}")

    except Exception as e:
        print(f"✖ ERROR al procesar {os.path.basename(predictions_path)}: {e}")


# --- SCRIPT PRINCIPAL ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRUTH_FILE = PROJECT_ROOT / 'data' / 'raw' / 'gz2_hart16.csv'
FILENAME_MAP_CSV = PROJECT_ROOT / 'data' / 'raw' / 'gz2_filename_mapping.csv' # <--- ARCHIVO CLAVE
INPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'csvs'
OUTPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'processed_data'
OUTPUT_FOLDER.mkdir(exist_ok=True)

try:
    # --- LÓGICA DE UNIÓN CORRECTA ---
    # 1. Leemos el archivo de etiquetas (el que tiene la clase de la galaxia)
    print("Cargando archivo de etiquetas (gz2_hart16.csv)...")
    df_labels = pd.read_csv(TRUTH_FILE)

    # 2. Leemos el archivo de mapeo de nombres
    print("Cargando archivo de mapeo de nombres (gz2_filename_mapping.csv)...")
    df_map = pd.read_csv(FILENAME_MAP_CSV)

    # 3. Unimos los dos DataFrames para conectar 'dr7objid' con 'asset_id'
    print("Uniendo etiquetas con nombres de archivo...")
    truth_dataframe_merged = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")

    # 4. Creamos la columna 'true_label' usando la función corregida
    truth_dataframe_merged['true_label'] = truth_dataframe_merged['gz2_class'].apply(get_simple_class)

    # 5. Creamos la columna 'filename' a partir de 'asset_id'
    truth_dataframe_merged['filename'] = truth_dataframe_merged['asset_id'].astype(str) + '.jpg'

    # 6. Seleccionamos solo las columnas que necesitamos para el siguiente paso
    final_truth_df = truth_dataframe_merged[['filename', 'true_label']]

except FileNotFoundError as e:
    print(f"✖ ERROR CRÍTICO: No se encontró un archivo esencial: {e}")
    exit()

# Bucle para procesar cada archivo de predicción (esto se mantiene igual)
prediction_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv') and 'predictions' in f and 'outside' not in f]

for pred_file in prediction_files:
    input_path = INPUT_FOLDER / pred_file
    output_filename = f"{pred_file.split('.csv')[0]}_ENRICHED.csv"
    output_path = OUTPUT_FOLDER / output_filename
    enrich_prediction_file(input_path, final_truth_df, output_path)

print("\n--- Proceso de enriquecimiento completado. ---")
