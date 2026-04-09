import pandas as pd
import sqlite3

# Load your CSV
df = pd.read_csv('OODatabase.csv')

# Create connection to local file (it will create the file if it doesn't exist)
conn = sqlite3.connect('Trading-Database.db')

# Save data to a table named 'OO-Database'
df.to_sql('OO-Database', conn, if_exists='replace', index=False)
conn.close()
print("Local SQLite database created!")