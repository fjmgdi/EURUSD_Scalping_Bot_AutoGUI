import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from api import OandaAPI
from strategy import SignalGenerator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="bot_activity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Init
TOKEN = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "CAD_JPY"]
api = OandaAPI()

# Main loop
def run():
    logging.info("Starting trading bot...")
    while True:
        for instrument in INSTRUMENTS:
            try:
                signal_generator = SignalGenerator(TOKEN, ACCOUNT_ID, instrument)
                df = signal_generator.fetch_candles()
                if df is None or len(df) < 20:
                    logging.warning(f"Insufficient data for {instrument}")
                    continue

                signal = signal_generator.generate_signal(df)
                logging.info(f"{instrument} - Signal: {signal}")

                pos = api.get_open_position(instrument)
                pnl = api.get_unrealized_pnl(instrument)
                logging.info(f"{instrument} - Position: {pos if pos else 'None'} - PnL: {pnl:.2f}")

                if signal in ["BUY", "SELL"] and not pos:
                    logging.info(f"{instrument} - Placing {signal} order...")
                    success = api.execute_trade(instrument, signal)
                    if success:
                        logging.info(f"{instrument} - {signal} order placed successfully.")
                    else:
                        logging.error(f"{instrument} - {signal} order failed.")

            except Exception as e:
                logging.exception(f"Error with {instrument}: {str(e)}")

        time.sleep(60)  # wait 60 seconds before next check

if __name__ == "__main__":
    run()
