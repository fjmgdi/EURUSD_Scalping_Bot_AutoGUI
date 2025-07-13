import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def export_candles_to_csv():
    api_key = os.getenv("OANDA_API_KEY")
    if not api_key:
        print("Please set OANDA_API_KEY in your environment.")
        return

    url = "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "count": 200,
        "granularity": "M5",
        "price": "M"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        candles = data.get("candles", [])
        if not candles:
            print("No candles data received.")
            return

        df = pd.DataFrame(candles)

        # Extract OHLC from the 'mid' dictionary
        df['open'] = df['mid'].apply(lambda x: float(x['o']))
        df['high'] = df['mid'].apply(lambda x: float(x['h']))
        df['low'] = df['mid'].apply(lambda x: float(x['l']))
        df['close'] = df['mid'].apply(lambda x: float(x['c']))

        # Select desired columns
        df = df[['time', 'open', 'high', 'low', 'close']]

        # Save to CSV
        df.to_csv("EUR_USD_M5.csv", index=False)
        print("Export completed successfully. Saved as EUR_USD_M5.csv")

    except Exception as e:
        print(f"Error fetching or saving data: {e}")

if __name__ == "__main__":
    export_candles_to_csv()
