import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

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


class GalacticDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0] # Columna 'path'
        label = int(self.data.iloc[idx, 1]) # Columna 'label'
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, label