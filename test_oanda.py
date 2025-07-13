import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OANDA_API_KEY")
account_id = os.getenv("OANDA_ACCOUNT_ID")

url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/summary"
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)
print("Status code:", response.status_code)
print(response.json())
