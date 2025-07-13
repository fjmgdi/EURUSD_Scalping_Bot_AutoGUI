import pandas as pd
import numpy as np
import datetime

class SignalGenerator:
    def __init__(self, api=None, instrument="EUR_USD"):
        self.api = api
        self.instrument = instrument

    def fetch_candles(self, count=100, granularity="M5"):
        if not self.api:
            print("[SignalGenerator] Error: API not initialized.")
            return pd.DataFrame()

        try:
            candles = self.api.get_candles(self.instrument, count, granularity)
            if not candles:
                print(f"[SignalGenerator] No candle data for {self.instrument}")
                return pd.DataFrame()

            data = {
                "time": [c["time"] for c in candles],
                "close": [float(c["mid"]["c"]) for c in candles]
            }
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

            return self.compute_indicators(df)

        except Exception as e:
            print(f"[SignalGenerator] Exception in fetch_candles: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df):
        if df.empty:
            return df

        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["ema20"] + (2 * std)
        df["bb_lower"] = df["ema20"] - (2 * std)

        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        df = df.bfill().dropna().reset_index()
        df["signal"] = df.apply(self.generate_signal, axis=1)
        return df

    def generate_signal(self, row):
        if (
            row["ema9"] > row["ema21"] and
            row["rsi"] < 70 and
            row["macd"] > row["signal"] and
            row["close"] < row["bb_upper"]
        ):
            return "buy"
        elif (
            row["ema9"] < row["ema21"] and
            row["rsi"] > 30 and
            row["macd"] < row["signal"] and
            row["close"] > row["bb_lower"]
        ):
            return "sell"
        return "none"

    def get_latest_signal(self, df):
        if df.empty or "signal" not in df.columns:
            return None
        return df["signal"].iloc[-1]
