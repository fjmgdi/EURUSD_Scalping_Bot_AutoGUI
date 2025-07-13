import pandas as pd
from oanda_api import OandaAPI
from strategy import SignalGenerator

class Backtester:
    """
    Backtester simulates trades on historical candles using your strategy.
    """

    def __init__(self, df, strategy, initial_balance=10000, sl_pips=10, tp_pips=20):
        self.df = df
        self.strategy = strategy
        self.balance = initial_balance
        self.position = None  # active trade: {'side': 'buy'/'sell', 'entry': price, 'time': timestamp}
        self.trades = []
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips

    def run(self):
        # Compute indicators & generate signals
        df_with_indicators = self.strategy.compute_indicators(self.df.copy(), rsi_sens=14, macd_sens=12, atr_filter=14)
        if df_with_indicators.empty:
            print("[Backtester] No data or indicators calculation failed.")
            return

        for _, row in df_with_indicators.iterrows():
            price = row['close']
            signal = row['signal']
            time = row['time']

            # Check SL/TP on active position
            if self.position:
                if self._check_exit(price, time):
                    continue  # closed position in this candle, skip opening new one here

            # Open new trade if signal generated & no active position
            if not self.position:
                if signal == "buy":
                    self._open_trade("buy", price, time)
                elif signal == "sell":
                    self._open_trade("sell", price, time)

        print(f"\n[Backtester] Backtest complete. Final balance: {self.balance:.2f}")
        self._print_summary()

    def _check_exit(self, price, time):
        entry = self.position['entry']
        side = self.position['side']
        pip_size = 0.0001 if "JPY" not in self.df.columns[0] else 0.01  # crude heuristic

        if side == 'buy':
            sl_price = entry - self.sl_pips * pip_size
            tp_price = entry + self.tp_pips * pip_size
            if price <= sl_price:
                self._close_trade(price, time, 'SL')
                return True
            elif price >= tp_price:
                self._close_trade(price, time, 'TP')
                return True
        elif side == 'sell':
            sl_price = entry + self.sl_pips * pip_size
            tp_price = entry - self.tp_pips * pip_size
            if price >= sl_price:
                self._close_trade(price, time, 'SL')
                return True
            elif price <= tp_price:
                self._close_trade(price, time, 'TP')
                return True
        return False

    def _open_trade(self, side, price, time):
        self.position = {"side": side, "entry": price, "time": time}
        print(f"[TRADE] {time}: Opened {side.upper()} at {price:.5f}")

    def _close_trade(self, exit_price, time, reason):
        entry = self.position['entry']
        side = self.position['side']
        pip_size = 0.0001 if "JPY" not in self.df.columns[0] else 0.01
        units = 100000  # assume 1 standard lot

        # Calculate PnL in dollars
        if side == "buy":
            profit_pips = (exit_price - entry) / pip_size
        else:
            profit_pips = (entry - exit_price) / pip_size
        profit = profit_pips * units * pip_size
        self.balance += profit

        self.trades.append({
            "side": side,
            "entry": entry,
            "exit": exit_price,
            "profit": profit,
            "time": time,
            "reason": reason
        })
        print(f"[TRADE] {time}: Closed {side.upper()} at {exit_price:.5f} | PnL: {profit:.2f} | Reason: {reason}")
        self.position = None

    def _print_summary(self):
        if not self.trades:
            print("\n[SUMMARY] No trades executed.")
            return
        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['profit'] > 0]
        win_rate = len(wins) / len(trades_df) * 100
        print("\n[SUMMARY]")
        print(trades_df)
        print(f"Total Trades: {len(trades_df)} | Win Rate: {win_rate:.2f}% | Final Balance: {self.balance:.2f}")


if __name__ == "__main__":
    # === CONFIG ===
    instrument = "EUR_USD"
    use_live_candles = False  # toggle between CSV or live candles

    # === Load data ===
    if use_live_candles:
        # Fetch from OANDA API
        api = OandaAPI()  # ✅ loads ACCESS_TOKEN + ACCOUNT_ID from .env automatically
        signal_generator = SignalGenerator(api=api, instrument=instrument)
        df = signal_generator.fetch_candles(count=500, granularity="M5")
    else:
        # Load from local CSV file
        try:
            df = pd.read_csv("historical_data.csv", parse_dates=["time"])
        except Exception as e:
            print(f"[Backtester] Failed to load historical_data.csv: {e}")
            exit(1)

    if df.empty:
        print("[Backtester] No data to backtest.")
    else:
        api = OandaAPI()  # still create API to pass into SignalGenerator
        strategy = SignalGenerator(api=api, instrument=instrument)
        backtester = Backtester(df, strategy, initial_balance=10000, sl_pips=10, tp_pips=20)
        backtester.run()
