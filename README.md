# 🌌 Celestial Classifier 🪐

**Clasificador de Morfología Galáctica** usando imágenes del proyecto [Galaxy Zoo 2](https://data.galaxyzoo.org/). <br>
Este proyecto implementa un **pipeline completo de Machine Learning y Data Engineering.** Se entrenan y evalúan múltiples modelos de Deep Learning (CNN, ResNet18) para clasificar galaxias en dos categorías **(elípticas, espirales)**

El núcleo del proyecto es un pipeline de datos automatizado que extrae, transforma y carga los resultados de las predicciones y los metadatos de las imágenes en una **base de datos PostgreSQL**, dejándola lista para el análisis y la visualización en herramientas de Business Intelligence como **Power BI**.




## 📁 Estructura del Proyecto
```bash
celestial_classifier/
│
├── dashboard/ # Power BI
│
├── data/
│  ├── raw/ # Datos crudos
│  │  ├── images_gz2/
│  │  │  ├── images/ # Imágenes (Windows)
│  │  │  └── __MACOSX/ # Metadatos (MacOS)
│  │  ├── gz2_hart16.csv # Etiquetas (morfología)
│  │  └── gz2_filename_mapping.csv # Relación imagen ↔ objeto
│  └── processed/ # Datasets separados y balanceados para entrenamiento y test
│     ├── all_images/
│     ├── test_set_balanced/
│     └── train_set_balanced/
│
├── scripts/
│  ├── prepare_dataset.py # Balanceo de cantidad de imagenes
│  ├── preprocess_data.py # Limpieza y clasificación de imágenes
│  ├── extract_features.py # Calcular valores de interes adicionales
│  ├── enrich_files.py # Añade 'true_label' y 'is_correct' a las predicciones.
│  └── load_db.py # Carga datos finales de prediccion a la base de datos
│
├── models/ # Pesos o modelos guardados
│  ├── architectures
│  ├── evaluations
│  ├── inferences
│  └── path_files
│
├── sql/ # PostgreSQL
│  ├── architecture/
│  └── query_scripts/
│
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

---

## 🚀 Guía de Instalación y Uso

Sigue estos pasos para configurar y ejecutar el proyecto completo.

### 1. 💾 Configuración Inicial

**Clonar el Repositorio**
```
git clone https://github.com/noriegitus/celestial_classifier.git
cd celestial_classifier
```

**Crear un Entorno Virtual (Recomendado)**
```
python -m venv venv
source venv/bin/activate # En Windows: venv\Scripts\activate
```

**Instalar Dependencias**
```
pip install -r requirements.txt
```


### 🛢 Configuración de la Base de Datos

- Asegúrate de que PostgreSQL esté instalado y corriendo.
- Crea una nueva base de datos (ej. `celestial_classifier`).
- Ejecuta las sentencias de SQL en orden:
```
1. create database.sql
2. insert_model_evaluation_tables.sql
3. insert_prediction_table.sql
4. final_verification.sql

---

### ⚙️ Ejecución del Pipeline de Datos

Ejecuta los scripts en este orden desde la carpeta raíz del proyecto.

**Fase 1: Generar Datos Crudos**
Ejecuta los modelos para generar los archivos de predicciones y métricas. Ejecuta cada script en orden.

```
python scripts/evaluate_(model).py
```

Extrae los metadatos de todas las imágenes
```
python scripts/extract_features.py
```

**Fase 2: Enriquecer los Datos**
Añade las etiquetas correctas ('true_label') y la columna 'is_correct'
```
python scripts/enrich_files.py
```

**Fase 3: Cargar a la Base de Datos**
Limpia las tablas de la base de datos antes de una nueva carga

Ejecutar en psql o pgAdmin:
```
TRUNCATE TABLE "Image" CASCADE;
```
Carga la tabla "Image" con los datos de todas las fuentes
```
python scripts/load_db.py
```

Finalmente, carga la tabla "Prediction" usando la terminal psql
Abre psql, conéctate a tu BD y ejecuta el script SQL con los comandos \copy

---

### 📊 Conexión con Power BI

- Abre Power BI.
- Selecciona "Obtener datos" -> "Base de datos PostgreSQL".
- Introduce las credenciales de tu servidor y base de datos.
- ¡Empieza a crear tus dashboards y a explorar los resultados!
