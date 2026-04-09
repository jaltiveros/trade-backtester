import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob
import shutil

# --- 1. CONFIGURATION ---
DB_PATH = "Trading-Database.db"
DB_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DB_URL)

# RECOMMENDATION: Use a separate folder for Calendar CSVs
FOLDER_PATH = r"/Users/alexaltiveros/trading_backtest_project/CalendarFiles"
ARCHIVE_PATH = os.path.join(FOLDER_PATH, "processed_archive")

def batch_upload_calendar():
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: Folder not found at {FOLDER_PATH}")
        return
    
    os.makedirs(ARCHIVE_PATH, exist_ok=True)
    csv_files = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    
    if not csv_files:
        print(f"ℹ️ No new Calendar CSV files found in {FOLDER_PATH}")
        return

    # --- 2. PRE-PROCESS: INTERNAL BACKUP ---
    with engine.begin() as conn:
        table_exists = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='CalendarEvents';"
        )).fetchone()
        
        if table_exists:
            conn.execute(text('DROP TABLE IF EXISTS "CalendarEvents_Backup";'))
            conn.execute(text('CREATE TABLE "CalendarEvents_Backup" AS SELECT * FROM "CalendarEvents";'))
            print("🔄 Pre-batch backup 'CalendarEvents_Backup' created.")

    # --- 3. LOOP THROUGH FILES ---
    total_added = 0
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"📖 Processing: {file_name}...")
        
        try:
            df = pd.read_csv(file_path)
            
            # Clean Column Names
            df.columns = [c.replace(' ', '_').replace('.', '').replace('/', '') for c in df.columns]
            
            # --- DATE/TIME PARSING ---
            # Extract Date and Time from the 'Start' column
            if 'Start' in df.columns:
                dt_series = pd.to_datetime(df['Start'], format='mixed', errors='coerce')
                df['Event_Date'] = dt_series.dt.strftime('%Y-%m-%d')
                df['Event_Time'] = dt_series.dt.strftime('%H:%M:%S')
                # Optional: Drop the original 'Start' to keep DB clean
                df = df.drop(columns=['Start'])

            # Deduplicate individual file based on the unique ID
            df = df.drop_duplicates(subset=["Id"])

            # Safe Merge via Staging
            with engine.begin() as conn:
                # Ensure Master table exists with the correct schema
                df.iloc[:0].to_sql('CalendarEvents', conn, if_exists='append', index=False)
                
                # Ensure Unique Index exists on the ID column
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_event 
                    ON "CalendarEvents" ("Id");
                """))
                
                # Upload to staging and merge
                df.to_sql('calendar_staging', conn, if_exists='replace', index=False)
                conn.execute(text('INSERT OR IGNORE INTO "CalendarEvents" SELECT * FROM calendar_staging;'))
                conn.execute(text('DROP TABLE IF EXISTS calendar_staging;'))

            # Move file to archive
            shutil.move(file_path, os.path.join(ARCHIVE_PATH, file_name))
            total_added += len(df)
            
        except Exception as e:
            print(f"⚠️ Error processing {file_name}: {e}")

    print(f"\n✅ Calendar Update Complete!")
    print(f"📊 Processed {total_added} events. Files moved to {ARCHIVE_PATH}")

if __name__ == "__main__":
    batch_upload_calendar()