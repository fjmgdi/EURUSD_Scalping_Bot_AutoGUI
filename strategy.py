import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, api=None, instrument="EUR_USD"):
        self.api = api
        self.instrument = instrument

    def fetch_candles(self, count=200, granularity="M5", rsi_sens=14, macd_sens=12, atr_filter=3):
        if not self.api:
            print("[SignalGenerator] Error: API instance not provided.")
            return pd.DataFrame()

        try:
            # Fetch M5 candles for primary signal generation
            m5_candles = self.api.get_candles(self.instrument, count, granularity)
            if not m5_candles:
                print("[SignalGenerator] No M5 candles received.")
                return pd.DataFrame()

            m5_df = pd.DataFrame({
                "time": [c["time"] for c in m5_candles],
                "close": [float(c["mid"]["c"]) for c in m5_candles],
                "high": [float(c["mid"]["h"]) for c in m5_candles],
                "low": [float(c["mid"]["l"]) for c in m5_candles]
            })
            m5_df["time"] = pd.to_datetime(m5_df["time"])
            m5_df.set_index("time", inplace=True)

            # Compute H1 trend confirmation
            self.compute_h1_trend()

            # Compute indicators on M5
            m5_df = self.compute_indicators(m5_df, rsi_sens, macd_sens, atr_filter)
            return m5_df.reset_index()

        except Exception as e:
            print(f"[SignalGenerator] Error fetching candles: {e}")
            return pd.DataFrame()

    def compute_h1_trend(self):
        try:
            h1_candles = self.api.get_candles(self.instrument, 100, "H1")
            if not h1_candles:
                print("[SignalGenerator] No H1 candles received.")
                self.h1_trend = None
                return

            h1_df = pd.DataFrame({
                "time": [c["time"] for c in h1_candles],
                "close": [float(c["mid"]["c"]) for c in h1_candles],
            })
            h1_df["time"] = pd.to_datetime(h1_df["time"])
            h1_df.set_index("time", inplace=True)

            # Compute H1 EMA50 as trend indicator
            h1_df["ema50"] = h1_df["close"].ewm(span=50, adjust=False).mean()

            # Latest H1 price and EMA50 determine trend
            last_price = h1_df["close"].iloc[-1]
            last_ema = h1_df["ema50"].iloc[-1]
            self.h1_trend = "up" if last_price > last_ema else "down"

            print(f"[SignalGenerator] H1 Trend detected: {self.h1_trend.upper()} (price={last_price:.5f} vs EMA50={last_ema:.5f})")
        except Exception as e:
            print(f"[SignalGenerator] Error computing H1 trend: {e}")
            self.h1_trend = None

    def compute_indicators(self, df, rsi_sens, macd_sens, atr_filter):
        if df.empty:
            return df

        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        stddev = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["ema20"] + (2 * stddev)
        df["bb_lower"] = df["ema20"] - (2 * stddev)

        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rsi_sens).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_sens).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"].fillna(50, inplace=True)

        ema_fast = df["close"].ewm(span=macd_sens, adjust=False).mean()
        ema_slow = df["close"].ewm(span=macd_sens * 2, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_prev"] = df["macd"].shift(1)

        # ATR for dynamic SL/TP
        df["tr"] = df[["high", "low", "close"]].apply(
            lambda row: max(
                row["high"] - row["low"],
                abs(row["high"] - row["close"]),
                abs(row["low"] - row["close"])
            ), axis=1
        )
        df["atr"] = df["tr"].rolling(window=atr_filter).mean()

        df = df.bfill().dropna()
        df["signal"] = df.apply(self.generate_signal, axis=1)
        return df

    def generate_signal(self, row):
        macd_rising = row["macd"] > row["signal"] and row["macd"] > row["macd_prev"]
        macd_falling = row["macd"] < row["signal"] and row["macd"] < row["macd_prev"]

        buy = (
            row["ema9"] >= row["ema21"] and
            row["rsi"] >= 50 and
            macd_rising and
            row["close"] >= row["ema20"] and
            self.h1_trend == "up"  # new H1 trend confirmation
        )

        sell = (
            row["ema9"] <= row["ema21"] and
            row["rsi"] <= 50 and
            macd_falling and
            row["close"] <= row["ema20"] and
            self.h1_trend == "down"  # new H1 trend confirmation
        )

        signal = "buy" if buy else "sell" if sell else "none"
        print(f"[DEBUG] Time={row.name} Close={row['close']:.5f} H1Trend={self.h1_trend} --> {signal.upper()}")
        return signal

    def get_latest_signal(self, df):
        if df.empty or "signal" not in df.columns:
            return None
        return df["signal"].iloc[-1]
