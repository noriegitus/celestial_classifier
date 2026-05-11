import pandas as pd
from pathlib import Path
import os

# ==========================================
# 1. Funciones de apoyo
# ==========================================

def get_simple_class(gz2_class):
    """Limpia las etiquetas complejas de Galaxy Zoo a algo simple."""
    gz2_class_upper = str(gz2_class).upper()
    if gz2_class_upper.startswith('E'): return 'elliptical'
    if gz2_class_upper.startswith('S'): return 'spiral'
    return None

def process_enrichment(input_path, truth_df, output_path, is_external=False):
    """
    Esta es la función maestra que cocina el archivo. 
    Funciona para AMBOS tipos de archivos.
    """
    try:
        df_preds = pd.read_csv(input_path)
        
        # Estandarizamos el nombre de la columna de imagen
        if 'image' in df_preds.columns:
            df_preds.rename(columns={'image': 'filename'}, inplace=True)
        
        # Unimos predicción con la verdad (El "Merge")
        df_merged = pd.merge(df_preds, truth_df, on='filename', how='left')
        
        # Limpiamos filas vacías (si alguna imagen no tenía etiqueta)
        df_merged.dropna(subset=['true_label'], inplace=True)
        
        # Renombramos para que el CSV final sea profesional
        if 'predicted' in df_merged.columns:
            df_merged.rename(columns={'predicted': 'predicted_label'}, inplace=True)
        
        # La prueba de fuego: ¿Acertó el modelo?
        df_merged['is_correct'] = (
            df_merged['predicted_label'].str.strip().str.lower() == 
            df_merged['true_label'].str.strip().str.lower()
        )
        
        # Ordenamos las columnas para que se vea bonito en Power BI
        final_columns = ['filename', 'true_label', 'predicted_label', 'confidence', 'is_correct']
        df_final = df_merged[final_columns]
        
        df_final.to_csv(output_path, index=False)
        print(f"  ✔ Guardado: {output_path.name}")

    except Exception as e:
        print(f"  ✖ ERROR en {input_path.name}: {e}")

# ==========================================
# 2. EL DIRECTOR DE ORQUESTA (Función Principal)
# ==========================================

def main():
    # --- Configuración de Rutas ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'csvs'
    OUTPUT_FOLDER = PROJECT_ROOT / 'outputs' / 'processed_data'
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Rutas de "Ingredientes" (Archivos de Verdad)
    GZ2_LABELS = PROJECT_ROOT / 'data' / 'raw' / 'gz2_hart16.csv'
    GZ2_MAP = PROJECT_ROOT / 'data' / 'raw' / 'gz2_filename_mapping.csv'
    EXTERNAL_LABELS = PROJECT_ROOT / 'data' / 'raw' / 'external_images_labels.csv'

    # 1. Escaneamos la cocina (ver qué archivos hay que procesar)
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv') and 'predictions' in f]
    
    if not all_files:
        print("No hay archivos para procesar en outputs/csvs.")
        return

    # Separamos los pedidos en dos listas
    external_queue = [f for f in all_files if 'outside' in f]
    internal_queue = [f for f in all_files if 'outside' not in f]

    # --- CASO A: PROCESAR GALAXY ZOO (Los pesados) ---
    if internal_queue:
        print(f"\n--- Preparando ingredientes para Galaxy Zoo ({len(internal_queue)} archivos) ---")
        try:
            df_labels = pd.read_csv(GZ2_LABELS)
            df_map = pd.read_csv(GZ2_MAP)
            
            # Unimos los ingredientes de GZ2 una sola vez para ahorrar memoria
            truth_gz2 = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")
            truth_gz2['true_label'] = truth_gz2['gz2_class'].apply(get_simple_class)
            truth_gz2['filename'] = truth_gz2['asset_id'].astype(str) + '.jpg'
            truth_gz2 = truth_gz2[['filename', 'true_label']]
            
            for file_name in internal_queue:
                process_enrichment(INPUT_FOLDER / file_name, truth_gz2, OUTPUT_FOLDER / f"{file_name.replace('.csv', '_ENRICHED.csv')}")
        except FileNotFoundError:
            print("⚠ Error: No se encontraron los archivos de Galaxy Zoo en data/raw/")

    # --- CASO B: PROCESAR EXTERNOS (Los ligeros) ---
    if external_queue:
        print(f"\n--- Preparando ingredientes para Externos ({len(external_queue)} archivos) ---")
        try:
            truth_ext = pd.read_csv(EXTERNAL_LABELS, usecols=['filename', 'true_label'])
            
            for file_name in external_queue:
                process_enrichment(INPUT_FOLDER / file_name, truth_ext, OUTPUT_FOLDER / f"{file_name.replace('.csv', '_ENRICHED.csv')}", is_external=True)
        except FileNotFoundError:
            print("⚠ Error: No se encontró el archivo external_images_labels.csv")

    print("\n--- ¡Cocina limpia! Todos los archivos han sido enriquecidos. ---")

if __name__ == "__main__":
    main()