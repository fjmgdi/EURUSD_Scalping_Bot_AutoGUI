import os
import requests
import pandas as pd
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

    def get_candles(self, instrument, count=200, granularity="M5"):
        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M"
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            candles = []
            for candle in data["candles"]:
                if candle["complete"]:
                    candles.append({
                        "time": candle["time"],
                        "open": float(candle["mid"]["o"]),
                        "high": float(candle["mid"]["h"]),
                        "low": float(candle["mid"]["l"]),
                        "close": float(candle["mid"]["c"]),
                        "volume": candle["volume"]
                    })
            return candles
        except Exception as e:
            print(f"[OandaAPI] Candle fetch error: {e}")
            return []

    def place_market_order(self, instrument, units, sl=None, tp=None):
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        order = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }

        if tp:
            order["order"]["takeProfitOnFill"] = {
                "price": str(round(tp, 5)),
                "timeInForce": "GTC"
            }
        if sl:
            order["order"]["stopLossOnFill"] = {
                "price": str(round(sl, 5)),
                "timeInForce": "GTC",
                "triggerMode": "TOP_OF_BOOK"
            }

        try:
            response = requests.post(url, headers=self.headers, json=order)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[OandaAPI] Order error: {e}")
            return None

    def get_account_summary(self):
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[OandaAPI] Account summary error: {e}")
            return None

    def get_open_trades(self):
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[OandaAPI] Open trades error: {e}")
            return None
