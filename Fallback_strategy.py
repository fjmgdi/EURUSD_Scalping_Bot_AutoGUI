import pandas as pd
import numpy as np
import datetime

class SignalGenerator:
    def __init__(self, api=None, instrument="EUR_USD"):
        self.api = api
        self.instrument = instrument

    def fetch_candles(self, count=200, granularity="M5", rsi_sens=14, macd_sens=12, atr_filter=3):
        if not self.api:
            print("[SignalGenerator] Error: API instance not provided.")
            return pd.DataFrame()

        try:
            candles = self.api.get_candles(self.instrument, count, granularity)
            if not candles:
                print("[SignalGenerator] No candles received.")
                return pd.DataFrame()

            df = pd.DataFrame({
                "time": [c["time"] for c in candles],
                "close": [float(c["mid"]["c"]) for c in candles],
                "high": [float(c["mid"]["h"]) for c in candles],
                "low": [float(c["mid"]["l"]) for c in candles]
            })

            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return self.compute_indicators(df, rsi_sens, macd_sens, atr_filter)

        except Exception as e:
            print(f"[SignalGenerator] Error fetching candles: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df, rsi_sens, macd_sens, atr_filter):
        if df.empty:
            return df

        # Synthetic volume for visualization
        df["volume"] = np.random.randint(100, 1000, size=len(df))

        # EMAs
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        stddev = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["ema20"] + (2 * stddev)
        df["bb_lower"] = df["ema20"] - (2 * stddev)

        # RSI
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rsi_sens).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_sens).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"].fillna(50, inplace=True)

        # MACD
        ema_fast = df["close"].ewm(span=macd_sens, adjust=False).mean()
        ema_slow = df["close"].ewm(span=macd_sens * 2, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_prev"] = df["macd"].shift(1)

        # ATR
        df["tr"] = df[["high", "low", "close"]].apply(
            lambda row: max(
                row["high"] - row["low"],
                abs(row["high"] - row["close"]),
                abs(row["low"] - row["close"])
            ), axis=1
        )
        df["atr"] = df["tr"].rolling(window=atr_filter).mean()

        df = df.bfill().dropna()
        df = df.reset_index()
        df["signal"] = df.apply(self.generate_signal, axis=1)

        return df

    def generate_signal(self, row):
        macd_rising = row["macd"] > row["signal"] and row["macd"] > row["macd_prev"]
        macd_falling = row["macd"] < row["signal"] and row["macd"] < row["macd_prev"]

        buy = (
            row["ema9"] >= row["ema21"] and
            row["rsi"] >= 50 and
            macd_rising and
            row["close"] >= row["ema20"]
        )

        sell = (
            row["ema9"] <= row["ema21"] and
            row["rsi"] <= 50 and
            macd_falling and
            row["close"] <= row["ema20"]
        )

        signal = "none"
        if buy:
            signal = "buy"
        elif sell:
            signal = "sell"

        # Debug output for each row
        print(f"[DEBUG] Time={row['time']} Close={row['close']:.5f} EMA9={row['ema9']:.5f} EMA21={row['ema21']:.5f} RSI={row['rsi']:.2f} MACD={row['macd']:.5f} Signal={row['signal']:.5f} --> {signal.upper()}")

        return signal

    def get_latest_signal(self, df):
        if df.empty or "signal" not in df.columns:
            return None
        return df["signal"].iloc[-1]
