import os
import shutil
import random

BASE = "C:/Users/valer/Programming_Projects/celestial_classifier"
PROCESSED_PATH = os.path.join(BASE, "data/processed/train_set")
BALANCED_PATH = os.path.join(BASE, "data/processed/balanced_train_set")
CLASSES = ["spiral", "elliptical"]

def balancear_dataset():
    print("Comenzando con el Balanceo del Dataset")
    class_counts={} # dictionary where .jpg files will be saved by class
    
    for class_name in CLASSES:
        class_dir = os.path.join(PROCESSED_PATH, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
        class_counts[class_name]=image_files # updating dictionary of images per class
    
    min_count = min(len(files) for files in class_counts.values()) # minimun value of images for both classes
    print(f"Copiando {min_count} imágenes por clase...")

    for class_name in CLASSES:
        src_dir = os.path.join(PROCESSED_PATH, class_name)
        dst_dir = os.path.join(BALANCED_PATH, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        selected_files = random.sample(class_counts[class_name],min_count) # random sample of min_count images per class 
        
        for filename in selected_files:
            shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, filename)) # copy each of the new images to new directory
            
    print("Dataset balanceado guardado en:", BALANCED_PATH)

if __name__ == "__main__":
    balancear_dataset()