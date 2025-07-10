import os # construye rutas de carpetas y archivos de forma segura y automática.
import shutil # modulo estandar para operar entre archivos y carpetas
import pandas as pd 

# paths

BASE = "C:/Users/valer/Programming_Projects/celestial_classifier"
RAW_IMG_RAW = os.path.join(BASE, "data/raw/images_gz2/images")
FILENAME_MAP_CSV = os.path.join(BASE, "data/raw/gz2_filename_mapping.csv")
LABELS_CSV = os.path.join(BASE, "data/raw/gz2_hart16.csv")
OUTPUT_PATH = os.path.join(BASE, "data/processed/train_set")

# objective categories
CLASSES = {
    "spiral": "t04_spiral_a08_spiral_debiased",
    "elliptical": "t01_smooth_or_features_a01_smooth_debiased",
}

# certainty filter
THERESHOLD = 0.80

def main():
    print("Cargando CSVs...")
    
    df_map = pd.read_csv(FILENAME_MAP_CSV)
    df_map.rename(columns={"asset_id": "filename"}, inplace=True)
    df_map["filename"] = df_map["filename"].astype(str) + ".jpg"

    # Verifica qué archivos realmente existen en la carpeta
    available_files = set(os.listdir(RAW_IMG_RAW))
    df_map = df_map[df_map["filename"].isin(available_files)]

    # Merge con etiquetas
    print("Uniendo etiquetas con nombres de archivo...")
    df_labels = pd.read_csv(LABELS_CSV)
    df = pd.merge(df_labels, df_map, left_on="dr7objid", right_on="objid")
    

    total = 0
    # column's the name for the class namme's debiased column 
    for class_name, column in CLASSES.items():
        # create a folder for each objective category
        class_dir = os.path.join(OUTPUT_PATH, class_name)
        os.makedirs(class_dir, exist_ok = True)
        
        print(f"Procesando Clase {class_name}")
        subset = df[df[column]>= THERESHOLD]  
        
        # iterates through each row from the filtered subset
        # row each individual galaxy with its name, _ is an unused index
        copied=0
        for _, row in subset.iterrows():  
            
            filename = row["filename"]
            
            src_path = os.path.join(RAW_IMG_RAW, filename)
            dst_path = os.path.join(class_dir, filename)
        
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                copied +=1
                total += 1
            else:
                print(f"[NO ENCONTRADO] {src_path}")
        
        print(f"{len(subset)} imagenes copiadas a {class_dir}")    
        
    print(f"Proceso completado. Total de imágenes copiadas {total}")
            
if __name__ == "__main__":
    main()
            