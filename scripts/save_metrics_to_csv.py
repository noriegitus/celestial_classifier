import os
import csv

def save_metrics_to_csv(model_name, metrics_dict, output_dir="outputs/csvs"):
    """
    Guarda las métricas de un modelo en un archivo CSV.

    Args:
        model_name (str): Nombre del modelo.
        metrics_dict (dict): Diccionario con las métricas.
        output_dir (str): Directorio donde se guardará el archivo CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_metrics.csv")

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics_dict.items():
            writer.writerow([key, f"{value:.4f}"])

    print(f"Métricas guardadas en: {output_file}")