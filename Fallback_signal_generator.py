import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange


class SignalGenerator:
    def __init__(self, api, instrument):
        self.api = api
        self.instrument = instrument
        self.xgb_model = xgb.XGBClassifier()
        self.proba_history = []

        try:
            self.xgb_model.load_model("xgb_next_candle.model")
            print("[SignalGenerator] XGBoost model loaded")
        except Exception as e:
            print(f"[SignalGenerator] Error loading XGBoost model: {e}")

    def generate_signals(self):
        candles = self.api.get_candles(self.instrument, count=200, granularity='M5')
        if candles is None or len(candles) == 0:
            print("[SignalGenerator] No candle data received")
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        if "time" not in df.columns and "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"])
        elif "time" not in df.columns:
            df["time"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="5min")

        if "volume" not in df.columns:
            df["volume"] = np.random.randint(100, 1000, size=len(df))

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            print(f"[SignalGenerator] Missing required columns: {df.columns}")
            return pd.DataFrame()

        df["rsi"] = RSIIndicator(close=df["close"]).rsi()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()

        df["macd_diff"] = df["macd"] - df["macd_signal"]
        df["candle_range"] = df["high"] - df["low"]

        support = []
        resistance = []
        window = 5
        for i in range(len(df)):
            if i < window or i > len(df) - window - 1:
                support.append(np.nan)
                resistance.append(np.nan)
            else:
                sl = df["low"].iloc[i - window:i + window + 1]
                sh = df["high"].iloc[i - window:i + window + 1]
                support.append(sl.min() if df["low"].iloc[i] == sl.min() else np.nan)
                resistance.append(sh.max() if df["high"].iloc[i] == sh.max() else np.nan)

        df["support"] = pd.Series(support).ffill().bfill()
        df["resistance"] = pd.Series(resistance).ffill().bfill()

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        features = [
            "open", "high", "low", "close", "rsi",
            "macd", "macd_signal", "atr", "macd_diff", "candle_range"
        ]
        X = df[features]

        try:
            proba = self.xgb_model.predict_proba(X.tail(1))[0][1]
            print(f"[SignalGenerator] Proba: {proba:.4f}")
        except Exception as e:
            print(f"[SignalGenerator] Prediction error: {e}")
            proba = 0.5

        self.proba_history.append(proba)
        if len(self.proba_history) > 100:
            self.proba_history.pop(0)

        mean_proba = np.mean(self.proba_history)

        if proba >= mean_proba + 0.02:
            signal = "buy"
        elif proba <= mean_proba - 0.02:
            signal = "sell"
        else:
            signal = "hold"

        df["signal"] = "hold"
        df["confidence"] = 0.5
        df["ml_confidence"] = proba
        df.at[df.index[-1], "signal"] = signal
        df.at[df.index[-1], "confidence"] = proba

        print(f"[SignalGenerator] Signal: {signal}, Confidence: {proba:.4f}")
        return df
