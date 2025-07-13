import pandas as pd
import requests

class SignalGenerator:
    def __init__(self, token, account_id, instrument="EUR_USD"):
        self.token = token
        self.account_id = account_id
        self.instrument = instrument
        self.url = f"https://api-fxpractice.oanda.com/v3/instruments/{self.instrument}/candles"

    def set_instrument(self, instrument):
        self.instrument = instrument
        self.url = f"https://api-fxpractice.oanda.com/v3/instruments/{self.instrument}/candles"

    def fetch_candles(self):
        params = {
            "count": 100,
            "granularity": "M1",
            "price": "M"
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(self.url, headers=headers, params=params)
            data = response.json()
            candles = data["candles"]
            df = pd.DataFrame([{
                "time": c["time"],
                "close": float(c["mid"]["c"])
            } for c in candles if c["complete"]])
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["bb_upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
            df["bb_lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
            return df
        except Exception as e:
            print(f"Error fetching candles for {self.instrument}: {e}")
            return None

    def generate_signal(self, df):
        if df is None or len(df) < 21:
            return "HOLD"

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if prev["close"] < prev["bb_lower"] and last["close"] > last["bb_lower"] and last["close"] > last["ema20"]:
            return "BUY"
        elif prev["close"] > prev["bb_upper"] and last["close"] < last["bb_upper"] and last["close"] < last["ema20"]:
            return "SELL"

        return "HOLD"
