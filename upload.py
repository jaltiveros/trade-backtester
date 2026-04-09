import pandas as pd
from sqlalchemy import create_engine, text, inspect
import os
import glob
import shutil

# --- 1. DYNAMIC CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Trading-Database.db")
PARQUET_PATH = os.path.join(BASE_DIR, "trading_data.parquet")
FOLDER_PATH = os.path.join(BASE_DIR, "OOFiles")
ARCHIVE_PATH = os.path.join(FOLDER_PATH, "processed_archive")

DB_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL)

def sync_parquet():
    print(f"\n🔄 Syncing {PARQUET_PATH} for high-performance Streamlit access...")
    try:
        query = 'SELECT Strategy, Date_Opened, Time_Opened, Legs, PL FROM "OO"'
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            print("⚠️ SQL table 'OO' is empty. Skipping Parquet sync.")
            return

        df['Strategy'] = df['Strategy'].astype('category')
        df['Time_Opened'] = df['Time_Opened'].astype('category')
        df['Date_Opened'] = pd.to_datetime(df['Date_Opened'])
        df['PL'] = pd.to_numeric(df['PL'], errors='coerce').fillna(0).astype('float32')

        temp_path = os.path.join(BASE_DIR, "temp_sync.parquet")
        df.to_parquet(temp_path, index=False, engine='pyarrow', compression='snappy')
        
        if os.path.exists(PARQUET_PATH):
            os.remove(PARQUET_PATH)
        os.rename(temp_path, PARQUET_PATH)
        print(f"✅ Parquet Sync Complete! {len(df):,} rows optimized.")
    except Exception as e:
        print(f"❌ Parquet Sync Failed: {e}")

def batch_upload():
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: Folder not found at {FOLDER_PATH}")
        return
    
    os.makedirs(ARCHIVE_PATH, exist_ok=True)
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"ℹ️ No new CSV files found in {FOLDER_PATH}")
        sync_parquet()
        return

    # --- 2. PRE-PROCESS: GET DB SCHEMA & BACKUP ---
    with engine.begin() as conn:
        table_exists = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='OO';"
        )).fetchone()
        
        if not table_exists:
            print("❌ Table 'OO' does not exist. Please run initial setup.")
            return

        # Get existing column names from the DB
        inspector = inspect(engine)
        db_columns = [col['name'] for col in inspector.get_columns("OO")]

        conn.execute(text('DROP TABLE IF EXISTS "OO_Backup";'))
        conn.execute(text('CREATE TABLE "OO_Backup" AS SELECT * FROM "OO";'))
        print(f"🔄 Backup created. Monitoring for {len(db_columns)} specific columns.")

    # --- 3. LOOP THROUGH FILES ---
    total_added = 0
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"📖 Processing: {file_name}...")
        
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.replace(' ', '_').replace('.', '').replace('/', '') for c in df.columns]

            # --- COLUMN RECONCILIATION LOGIC ---
            # 1. Add missing columns with default value 0
            for col in db_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # 2. Drop columns that don't exist in the DB (like 'Underlying')
            df = df[db_columns]

            # Clean Strategy Name
            if 'Strategy' in df.columns:
                df['Strategy'] = df['Strategy'].astype(str).apply(lambda x: x.split('-')[2] if len(x.split('-')) > 2 else x)

            # Date/Time Parsing
            for col in df.columns:
                if 'Date' in col:
                    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
                elif 'Time' in col:
                    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce').dt.strftime('%H:%M:%S')

            df = df.drop_duplicates(subset=["Date_Opened", "Time_Opened", "Legs", "Strategy"])

            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_trade 
                    ON "OO" ("Date_Opened", "Time_Opened", "Legs", "Strategy");
                """))
                
                df.to_sql('oo_staging', conn, if_exists='replace', index=False)
                conn.execute(text('INSERT OR IGNORE INTO "OO" SELECT * FROM oo_staging;'))
                conn.execute(text('DROP TABLE IF EXISTS oo_staging;'))

            shutil.move(file_path, os.path.join(ARCHIVE_PATH, file_name))
            total_added += len(df)
            
        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")

    print(f"\n✅ Batch Complete! {total_added} trades processed.")
    sync_parquet()

if __name__ == "__main__":
    batch_upload()