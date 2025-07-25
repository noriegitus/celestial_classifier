# 🪐 Celestial Classifier

**Clasificador de Morfología Galáctica** usando imágenes del proyecto [Galaxy Zoo 2](https://data.galaxyzoo.org/). <br>
Este proyecto busca entrenar un modelo de clasificación de galaxias usando deep learning y visualización científica, con el objetivo a futuro de contribuir a tareas reales de astronomía computacional.


## 📁 Estructura del Proyecto
```bash
celestial_classifier/
│
├── data/
│ ├── raw/ # Datos crudos
│ │ ├── images_gz2/
│ │ │ ├── images/ # Imágenes (Windows)
│ │ │ └── __MACOSX/ # Metadatos (MacOS)
│ │ ├── gz2_hart16.csv # Etiquetas (morfología)
│ │ └── gz2_filename_mapping.csv # Relación imagen ↔ objeto
│ └── processed/ # Dataset listo para entrenamiento
│
├── scripts/
│ ├── preprocess_data.py # Limpieza y clasificación de imágenes
│ └── train_model.py # Entrenamiento del modelo
│
├── models/ # Pesos o modelos guardados
├── notebooks/ # Notebooks exploratorios
├── dashboard/ # Power BI
├── sql/ # Consultas SQL
├── README.md
└── requirements.txt
```

## 📥 Descarga de Datos
Este proyecto utiliza datos del proyecto [Galaxy Zoo 2](https://data.galaxyzoo.org/).

Para poder replicar este clasificador, necesitas descargar los siguientes archivos:

1. Imágenes del dataset
   - URL directa: [`images_gz2.zip`](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images?resource=download&select=images_gz2)
2. Archivo de etiquetas CSV: [`gz2_hart16.csv`](https://static.zooniverse.org/data.galaxyzoo.org/#section-8)
3. Archivo de mapeo CSV: [`gz2_filename_mapping.csv`](https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images?resource=download&select=gz2_filename_mapping.csv)

📁 Una vez descargados, descomprime `images_gz2.zip` en la ruta:
```"data/raw/images_gz2/"```

## 🚀 Instrucciones de Uso
1. **Instalar** Dependencias
```bash
pip install -r requirements.txt
```
2. **Preprocesar** Imágenes
```bash
python scripts/preprocess_data.py
```
3. **Entrenar** el Modelo
```bash
python scripts/train_model.py
```
