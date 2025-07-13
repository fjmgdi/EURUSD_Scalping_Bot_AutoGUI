#!/usr/bin/env python3
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# ----------------- CONFIGURATION -----------------
OANDA_API_KEY = "4c33eb8b086b50f08183047393d6363a-743a0d749e8d8fa1539724b7878a23c9"
ACCOUNT_ID = "101-002-28367236-001"
ENVIRONMENT = "practice"  # use "live" if you have a live account
PAIR = "EUR_USD"          # or CAD_JPY, GBP_JPY etc.
CANDLE_COUNT = 100        # number of candles to fetch
GRANULARITY = "M5"        # e.g., M1, M5, H1

# ----------------- LOGGING -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- MAIN -----------------
def main():
    logger.info("Connecting to OANDA API...")
    client = API(access_token=OANDA_API_KEY, environment=ENVIRONMENT)

    # Request candles
    params = {"count": CANDLE_COUNT, "granularity": GRANULARITY}
    candles_request = InstrumentsCandles(instrument=PAIR, params=params)

    try:
        response = client.request(candles_request)
        logger.info("Successfully fetched candles")
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return

    # Parse candles
    times = []
    closes = []

    for candle in response.get('candles', []):
        if candle['complete']:
            time_str = candle['time'].replace("Z", "")
            times.append(datetime.fromisoformat(time_str))
            closes.append(float(candle['mid']['c']))

    if not closes:
        logger.error("No data returned for plotting - empty price series!")
        return

    # Build DataFrame
    df = pd.DataFrame({"time": times, "close": closes}).set_index("time")
    logger.info(f"First few rows of data:\n{df.head()}")

    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label=f"{PAIR} Close Price", color="blue")
    plt.title(f"{PAIR} Live Candles ({GRANULARITY})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
