import os
import json
import pandas as pd
import requests
from dotenv import load_dotenv
from oandapyV20 import API
from oandapyV20.endpoints import accounts, instruments, orders, trades

load_dotenv()  # Load environment variables from .env

class OandaAPI:
    def __init__(self):
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.api_key = os.getenv("OANDA_API_KEY")
        self.base_url = "https://api-fxpractice.oanda.com/v3"
        self.client = API(access_token=self.api_key)
        print("[OandaAPI] Initialized")

    def get_candles(self, instrument, count=200, granularity="M5"):
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"
            }
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            candles = r.response["candles"]

            data = []
            for c in candles:
                if c["complete"]:
                    data.append({
                        "time": c["time"],
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low": float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"]),
                        "volume": c["volume"]
                    })

            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            return df

        except Exception as e:
            print(f"[OandaAPI] get_candles error: {e}")
            return pd.DataFrame()

    def get_price(self, instrument):
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/pricing?instruments={instrument}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            prices = response.json()["prices"][0]
            bid = float(prices["bids"][0]["price"])
            ask = float(prices["asks"][0]["price"])
            return (bid + ask) / 2
        except Exception as e:
            print(f"[OandaAPI] get_price error: {e}")
            return None

    def place_market_order(self, instrument, units, sl_price=None, tp_price=None):
        try:
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }

            if sl_price:
                order_data["order"]["stopLossOnFill"] = {"price": str(round(sl_price, 5))}
            if tp_price:
                order_data["order"]["takeProfitOnFill"] = {"price": str(round(tp_price, 5))}

            r = orders.OrderCreate(self.account_id, data=order_data)
            response = self.client.request(r)
            return response

        except Exception as e:
            print(f"[OandaAPI] place_market_order error: {e}")
            return None

    def close_position(self, instrument):
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/positions/{instrument}/close"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "longUnits": "ALL",
                "shortUnits": "ALL"
            }
            response = requests.put(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"[OandaAPI] close_position error: {e}")
            return None

    def get_balance(self):
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            response = self.client.request(r)
            return float(response["account"]["balance"])
        except Exception as e:
            print(f"[OandaAPI] get_balance error: {e}")
            return None

    def get_open_trades(self):
        try:
            url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()["trades"]
        except Exception as e:
            print(f"[OandaAPI] get_open_trades error: {e}")
            return []
