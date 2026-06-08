import os
import json
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()
client = PageIndexClient(os.getenv('PAGEINDEX_API_KEY'))
retrieval_id = 'sr-cmq50m0av04yk01qxmg6ia4t6'

print("Polling...")
res = client.get_retrieval(retrieval_id)
status = res.get("status")
print(f"Status: {status}")
if status == "completed":
    with open("scratch/retrieval_res.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved to scratch/retrieval_res.json")
