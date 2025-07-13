import pandas as pd
from datetime import datetime
from api import OandaAPI

def backtest():
    api = OandaAPI()

    INSTRUMENT = "EUR_USD"
    GRANULARITY = "M5"
    START_DATE = datetime(2024, 1, 1, 0, 0, 0)
    END_DATE = datetime(2024, 3, 31, 23, 59, 59)

    print(f"[Backtest] Fetching candles for {INSTRUMENT} from {START_DATE.isoformat()} to {END_DATE.isoformat()}...")

    candles = api.get_historical_candles(INSTRUMENT, START_DATE, END_DATE, GRANULARITY)

    if not candles:
        print("[Backtest] Failed to fetch candles.")
        return

    # Parse candles into DataFrame
    data = {
        "time": [],
        "close": [],
        "high": [],
        "low": [],
        "volume": []
    }

    for candle in candles:
        if not candle["complete"]:
            continue  # skip incomplete candles
        data["time"].append(pd.to_datetime(candle["time"]))
        data["close"].append(float(candle["mid"]["c"]))
        data["high"].append(float(candle["mid"]["h"]))
        data["low"].append(float(candle["mid"]["l"]))
        data["volume"].append(int(candle.get("volume", 0)))

    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)

    print(df.head())
    print(f"[Backtest] Retrieved {len(df)} candles.")

    # Here you can add your backtest logic on df, e.g., apply indicators, generate signals, simulate trades

if __name__ == "__main__":
    backtest()
