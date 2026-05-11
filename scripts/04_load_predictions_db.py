import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# --- DATABASE CONFIGURATION ---
DB_USER = "postgres"
DB_PASSWORD = "7894"  # <--- MAKE SURE THIS IS CORRECT
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "celestial_classifier"

# --- FILE PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_FOLDER = PROJECT_ROOT / 'outputs' / 'processed_data'
METADATA_FILE = PROJECT_ROOT / 'outputs' / 'images_metadata_master.csv'
EXTERNAL_LABELS_FILE = PROJECT_ROOT / 'data' / 'raw' / 'external_images_labels.csv'

try:
    print(f"\nConnecting to database '{DB_NAME}'...")
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    print(f"Loading metadata from: {METADATA_FILE.relative_to(PROJECT_ROOT)}")
    df_meta = pd.read_csv(METADATA_FILE)

    # --- Step 1: Unify ALL true labels from BOTH sources ---
    print("Unifying true labels from Galaxy Zoo and external sources...")
    enriched_filenames = [
        'cnn_v1_predictions_ENRICHED.csv', 'cnn_v2_predictions_ENRICHED.csv',
        'resnet18_v1_predictions_ENRICHED.csv', 'resnet18_v2_predictions_ENRICHED.csv',
        'resnet18_v3_predictions_ENRICHED.csv'
    ]
    enriched_files_paths = [PROCESSED_DATA_FOLDER / f for f in enriched_filenames]
    df_gz2_labels = pd.concat([pd.read_csv(f, usecols=['filename', 'true_label']) for f in enriched_files_paths])
    df_external_labels = pd.read_csv(EXTERNAL_LABELS_FILE, usecols=['filename', 'true_label'])
    df_all_labels = pd.concat([df_gz2_labels, df_external_labels]).drop_duplicates(subset=['filename'])
    
    # --- Step 2: Combine and Clean ---
    print("Combining metadata with labels...")
    df_images_final = pd.merge(df_meta, df_all_labels, on='filename', how='left')
    df_images_final.dropna(subset=['true_label'], inplace=True)
    df_images_final.drop_duplicates(subset=['filename'], keep='first', inplace=True)

    # --- Step 3: Prepare DataFrame for Database ---
    print("Preparing final columns for database insertion...")
    # Rename columns to exactly match the database table schema
    df_images_final.rename(columns={
        'average_brightness': 'avg_brightness',
        'average_contrast': 'avg_contrast',
        'dominant_color_hex': 'dom_color_hex'
    }, inplace=True)
    
    # Add the 'description' column, which doesn't come from the CSVs
    df_images_final['description'] = None
    
    # Select and order the columns to perfectly match the `Image` table
    final_db_columns = [
        'filename', 'true_label', 'description', 'folder', 'source', 
        'avg_brightness', 'avg_contrast', 'dom_color_hex', 
        'height_px', 'width_px', 'filesize_kb'
    ]
    df_for_db = df_images_final[final_db_columns]

    # --- Step 4: Insert into Database ---
    print(f"Inserting {len(df_for_db)} unique records into the 'Image' table...")
    # Before running, make sure the Image table is empty (`TRUNCATE TABLE Image CASCADE;`)
    df_for_db.to_sql('image', engine, if_exists='append', index=False)
    
    print("\n✔ Success! The 'Image' table has been populated correctly.")

except Exception as e:
    print(f"\n--- An Error Occurred --- \n{e}")