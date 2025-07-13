import os
import requests
from dotenv import load_dotenv


class OandaAPI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.base_url = "https://api-fxpractice.oanda.com/v3"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print("[OandaAPI] Loaded from:", __file__)

    def place_market_order(self, instrument, units, sl=None, tp=None):
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        order_data = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }

        if sl is not None:
            order_data["order"]["stopLossOnFill"] = {"price": f"{sl:.5f}"}
        if tp is not None:
            order_data["order"]["takeProfitOnFill"] = {"price": f"{tp:.5f}"}

        response = requests.post(url, headers=self.headers, json=order_data)
        if response.status_code != 201:
            print(f"[OandaAPI] Order failed: {response.text}")
            return None
        return response.json()

    def get_account_summary(self):
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get account summary: {response.text}")
        return response.json()

    def get_open_trades(self):
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get open trades: {response.text}")
        return response.json()

    def get_account_details(self):
        url = f"{self.base_url}/accounts/{self.account_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get account details: {response.text}")
        return response.json()
