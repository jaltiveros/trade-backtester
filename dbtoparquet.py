import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob
import shutil

# --- 1. CONFIGURATION ---
DB_PATH = "Trading-Database.db"
DB_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL)

# Folder settings
FOLDER_PATH = r"/Users/alexaltiveros/Desktop/trading_backtest_project/OOFiles"
ARCHIVE_PATH = os.path.join(FOLDER_PATH, "processed_archive")

def batch_upload():
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: Folder not found at {FOLDER_PATH}")
        return
    
    os.makedirs(ARCHIVE_PATH, exist_ok=True)
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"ℹ️ No new CSV files found in {FOLDER_PATH}")
        return

    # --- 2. PRE-PROCESS: INTERNAL BACKUP ---
    with engine.begin() as conn:
        table_exists = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='OO';"
        )).fetchone()
        
        if table_exists:
            conn.execute(text('DROP TABLE IF EXISTS "OO_Backup";'))
            conn.execute(text('CREATE TABLE "OO_Backup" AS SELECT * FROM "OO";'))
            print("🔄 Pre-batch backup 'OO_Backup' created.")

    # --- 3. LOOP THROUGH FILES ---
    total_added = 0
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"📖 Processing: {file_name}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # Clean Columns
            df.columns = [c.replace(' ', '_').replace('.', '').replace('/', '') for c in df.columns]
            
            # Clean Strategy Name
            if 'Strategy' in df.columns:
                df['Strategy'] = df['Strategy'].str.split('-').str[2]

            # --- IMPROVED DATE/TIME PARSING (Fixes UserWarnings) ---
            for col in df.columns:
                if 'Date' in col:
                    # 'format="mixed"' is the modern way to handle varied date strings safely
                    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
                elif 'Time' in col:
                    # Specifying format='mixed' here usually silences the "inference" warning
                    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce').dt.strftime('%H:%M:%S')

            # Deduplicate individual file
            df = df.drop_duplicates(subset=["Date_Opened", "Time_Opened", "Legs", "Strategy"])

            # Safe Merge via Staging
            with engine.begin() as conn:
                # Ensure Master exists with correct schema
                df.iloc[:0].to_sql('OO', conn, if_exists='append', index=False)
                
                # Ensure Unique Index exists
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_trade 
                    ON "OO" ("Date_Opened", "Time_Opened", "Legs", "Strategy");
                """))
                
                df.to_sql('oo_staging', conn, if_exists='replace', index=False)
                conn.execute(text('INSERT OR IGNORE INTO "OO" SELECT * FROM oo_staging;'))
                conn.execute(text('DROP TABLE IF EXISTS oo_staging;'))

            # Move file to archive after successful DB merge
            shutil.move(file_path, os.path.join(ARCHIVE_PATH, file_name))
            total_added += len(df)
            
        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")

    print(f"\n✅ Batch Complete!")
    print(f"📊 Processed {total_added} trades. Files moved to {ARCHIVE_PATH}")

if __name__ == "__main__":
    batch_upload()