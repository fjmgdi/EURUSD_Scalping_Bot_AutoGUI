import pandas as pd
import numpy as np
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import xgboost as xgb


class SignalGenerator:
    def __init__(self, access_token, account_id, instrument="EUR_USD"):
        self.client = API(access_token=access_token)
        self.account_id = account_id
        self.instrument = instrument

        # Load XGBoost model for signal prediction (optional)
        try:
            self.model = xgb.Booster()
            self.model.load_model("xgb_next_candle.model")
            print("[SignalGenerator] XGBoost model loaded")
        except Exception as e:
            self.model = None
            print(f"[SignalGenerator] Could not load XGBoost model: {e}")

    def fetch_live_candles(self, count=100, granularity="M5"):
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M",  # Mid prices
        }
        r = InstrumentsCandles(instrument=self.instrument, params=params)
        self.client.request(r)
        candles = r.response.get("candles", [])

        # Parse candles to DataFrame
        data = []
        for c in candles:
            if c['complete']:
                mid = c["mid"]
                data.append({
                    "time": pd.to_datetime(c["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c["volume"]),
                })
        df = pd.DataFrame(data)
        return df

    def generate_signals_live(self):
        df = self.fetch_live_candles()
        if df.empty:
            return pd.DataFrame()

        # Add EMA20
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

        # Add RSI (14 period)
        df["rsi"] = self._calculate_rsi(df["close"], 14)

        # Add MACD (12,26,9)
        macd, signal_line = self._calculate_macd(df["close"])
        df["macd"] = macd
        df["macd_signal"] = signal_line

        # Generate signals based on EMA20 crossover
        df["signal"] = "hold"
        for i in range(1, len(df)):
            if df["close"].iloc[i - 1] < df["ema20"].iloc[i - 1] and df["close"].iloc[i] > df["ema20"].iloc[i]:
                df.at[df.index[i], "signal"] = "buy"
            elif df["close"].iloc[i - 1] > df["ema20"].iloc[i - 1] and df["close"].iloc[i] < df["ema20"].iloc[i]:
                df.at[df.index[i], "signal"] = "sell"

        # Confidence: use model prediction if available, else default 0.5
        df["confidence"] = 0.5
        if self.model is not None:
            feature_cols = ["open", "high", "low", "close"]
            X = df[feature_cols].tail(10).values
            if len(X) > 0:
                dmatrix = xgb.DMatrix(X)
                preds = self.model.predict(dmatrix)
                df.loc[df.index[-10:], "confidence"] = preds[-1]

        return df

    @staticmethod
    def _calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    @staticmethod
    def _calculate_macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
