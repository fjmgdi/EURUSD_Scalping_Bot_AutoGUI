import pandas as pd
from signal_generator import SignalGenerator

def backtest(csv_file="EUR_USD_M5.csv"):
    df = pd.read_csv(csv_file, parse_dates=["time"])
    df.set_index("time", inplace=True)

    # Since your SignalGenerator fetches data from API, let's make a lightweight signal function here:
    # We'll recreate the indicators & signals as in your signal_generator.py

    # Example indicator: 20 EMA
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Example signal logic:
    df["signal"] = "hold"
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        # Buy if price crosses above EMA20
        if prev["close"] < prev["ema20"] and curr["close"] > curr["ema20"]:
            df.at[df.index[i], "signal"] = "buy"
        # Sell if price crosses below EMA20
        elif prev["close"] > prev["ema20"] and curr["close"] < curr["ema20"]:
            df.at[df.index[i], "signal"] = "sell"

    # Simulate trades
    position = 0
    entry_price = 0
    balance = 10000  # start balance
    for i, row in df.iterrows():
        signal = row["signal"]
        price = row["close"]

        if signal == "buy" and position == 0:
            position = 1
            entry_price = price
            print(f"{i} BUY @ {price}")

        elif signal == "sell" and position == 1:
            profit = price - entry_price
            balance += profit * 1000  # assuming 1000 units per trade
            print(f"{i} SELL @ {price} Profit: {profit * 1000:.2f}")
            position = 0

    print(f"Final balance: {balance:.2f}")

if __name__ == "__main__":
    backtest()
