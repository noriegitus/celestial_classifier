import os
import csv

def save_predictions_to_csv(model_name, predictions, output_dir="outputs/csvs"):
    """
    Guarda las predicciones en un archivo CSV.

    Args:
        model_name (str): Nombre del modelo.
        predictions (list): Lista de diccionarios con claves 'image', 'predicted', 'confidence'.
        output_dir (str): Directorio donde guardar.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_predictions.csv")

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "predicted", "confidence"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)

    print(f"Predicciones guardadas en: {output_file}")