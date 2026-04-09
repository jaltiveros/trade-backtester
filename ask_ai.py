import vanna
from vanna.remote import VannaDefault

# 1. Setup with your Vanna credentials
api_key = 'vn-87f5a02f6e2e42feb25ddbf2058fe6a7'
model_name = 'echo'
vn = VannaDefault(model=model_name, api_key=api_key)

# 2. Connect to your Neon Database
vn.connect_to_postgres(
    host='ep-long-bread-aa3wc5ag.westus3.azure.neon.tech',
    dbname='neondb',
    user='neondb_owner',
    password='npg_2t8HlKuGTbmQ',
    port=5432
)

# 3. Training: Run this ONCE to teach the AI your schema
# After the first run, you can comment this block out with #
training_plan = """
The database table is named "OO". It contains the following columns:
- "Date Opened" (date)
- "Time Opened" (time)
- "Opening Price" (numeric)
- "Legs" (text)
- "Premium" (numeric)
- "Closing Price" (numeric)
- "Date Closed" (date)
- "Time Closed" (time)
- "Avg. Closing Cost" (numeric)
- "Reason For Close" (text)
- "P/L" (numeric)
- "P/L %" (numeric)
- "No. of Contracts" (numeric)
- "Funds at Close" (numeric)
- "Margin Req." (numeric)
- "Strategy" (text)
- "Opening Commissions + Fe" (numeric)
- "Closing Commissions + Fe" (numeric)
- "Opening Short/Long Ratio" (numeric)
- "Closing Short/Long Ratio" (numeric)
- "Gap" (numeric)
- "Movement" (numeric)
- "Max Profit" (numeric)
- "Max Loss" (numeric)
"""
vn.train(documentation=training_plan)

# 4. Ask a Question
print("--- Echo is online ---")
question = "What is the average P/L per Strategy?"
answer = vn.ask(question)