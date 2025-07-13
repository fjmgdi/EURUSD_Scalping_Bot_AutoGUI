import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from oanda_api import OandaAPI  # your API wrapper

def main():
    api = OandaAPI()
    instrument = "EUR_USD"

    print("[TrainingData] Fetching candles...")
    candles = api.get_candles(instrument, count=1000, granularity="M5")

    if not candles or len(candles) == 0:
        print("No candle data fetched.")
        return

    # Flatten candle data
    data = []
    for c in candles:
        try:
            data.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c.get("volume", 0)
            })
        except KeyError as e:
            print(f"Missing key in candle data: {e}")
            continue

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])

    # Calculate indicators
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range()

    # Additional features
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    df["candle_range"] = df["high"] - df["low"]

    # You can add labeling here for next candle movement (buy/sell/hold) if needed

    # Save processed data to CSV
    df.to_csv("training_data.csv", index=False)
    print("[TrainingData] Saved training_data.csv")

if __name__ == "__main__":
    main()
