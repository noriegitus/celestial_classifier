import pandas as pd
import os 

def get_simple_class(gz2_class):
    """
    Traduce la clase detallada de Galaxy Zoo a una de las tres clases simples.
    """
    if isinstance(gz2_class, str):
        if gz2_class.startswith('E'):
            return 'elliptical'
        elif gz2_class.startswith('S'):
            return 'spiral'
    return 'irregular'

def enrich_predictions_with_truth(predictions_path, gz2_data_path, output_path):
    """
    Combina el archivo de predicciones con los datos de Galaxy Zoo para
    añadir la etiqueta real (true_label) y verificar si el modelo acertó.
    """

        # --- CÓDIGO DE DIAGNÓSTICO ---
    # Vamos a imprimir los nombres de las columnas para ver cómo se llaman realmente
    print("--- Diagnosticando columnas de gz2_hart16.csv ---")
    temp_df = pd.read_csv(gz2_data_path, nrows=1) # Leemos solo la primera fila para ver los encabezados
    print("Los nombres de las columnas son:")
    print(temp_df.columns.tolist())
    print("-------------------------------------------------")
    # --- FIN DE DIAGNÓSTICO ---

    try:
        print(f"Cargando predicciones desde: {predictions_path}")
        df_preds = pd.read_csv(predictions_path)

        print(f"Cargando datos de Galaxy Zoo desde: {gz2_data_path}")
        df_gz2 = pd.read_csv(gz2_data_path, usecols=['dr7objid', 'gz2_class'])
        
        # --- El resto del código es igual, el cambio fue en las rutas ---
        
        df_preds['dr7objid'] = df_preds['image'].str.replace('.jpg', '', regex=False).astype(int)

        print("Combinando los archivos...")
        df_merged = pd.merge(df_preds, df_gz2, on='dr7objid', how='left')

        df_merged['true_label'] = df_merged['gz2_class'].apply(get_simple_class)
        df_merged['es_acierto'] = df_merged['true_label'] == df_merged['predicted']

        print("Limpiando y reordenando las columnas finales...")
        final_columns = ['image', 'true_label', 'predicted', 'confidence', 'es_acierto']
        df_final = df_merged[final_columns]
        
        # --- Nos aseguramos que la carpeta de salida exista ---
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio de salida creado en: {output_dir}")

        df_final.to_csv(output_path, index=False)
        print(f"¡Éxito! Archivo enriquecido guardado en: {output_path}\n")

        return df_final

    except FileNotFoundError as e:
        print(f"--- ERROR ---")
        print(f"No se encontró un archivo. Revisa que la ruta sea correcta.")
        print(f"Python está buscando aquí: {e.filename}")
        print(f"Asegúrate que la estructura de carpetas coincida con el script.")
        print(f"---------------")
        return None

# --- CÓMO USARLO (CON LA ESTRUCTURA DE CARPETAS) ---

# 

# 1. Definimos las rutas relativas desde la raíz del proyecto
gz2_file = os.path.join('data', 'raw', 'gz2_hart16.csv')
resnet_preds_file = os.path.join('outputs', 'csvs', 'resnet18_v3_predictions.csv')
cnn_preds_file = os.path.join('outputs', 'csvs', 'cnn_v2_predictions.csv')

# 2. Definimos dónde se guardarán los nuevos archivos
output_folder = 'processed_data'
resnet_output_file = os.path.join('outputs', output_folder, 'resnet18_v3_predictions_ENRICHED.csv')
cnn_output_file = os.path.join('outputs', output_folder, 'cnn_v2_predictions_ENRICHED.csv')

# 3. Ejecutamos el proceso para ambos modelos
enrich_predictions_with_truth(
    predictions_path=resnet_preds_file,
    gz2_data_path=gz2_file,
    output_path=resnet_output_file
)

enrich_predictions_with_truth(
    predictions_path=cnn_preds_file,
    gz2_data_path=gz2_file,
    output_path=cnn_output_file
)