import sys
import time
import threading
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QComboBox, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from oanda_api import OandaAPI
from signal_generator import SignalGenerator


class TradingBotGUI(QWidget):
    def __init__(self):
        super().__init__()
        print("[GUI] Initializing TradingBotGUI")

        self.api = OandaAPI()
        self.instrument = "EUR_USD"
        self.signal_gen = SignalGenerator(
            access_token=self.api.api_key,
            account_id=self.api.account_id,
            instrument=self.instrument
        )

        self.auto_trade_enabled = False
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_all)
        self.update_all()
        self.timer.start(10000)

    def init_ui(self):
        self.setWindowTitle("AI Forex Trading Bot")
        self.resize(1200, 800)

        layout = QVBoxLayout()

        top_controls = QHBoxLayout()

        self.inst_combo = QComboBox()
        self.inst_combo.addItems(["EUR_USD", "USD_JPY", "GBP_USD"])
        self.inst_combo.currentTextChanged.connect(self.change_instrument)
        top_controls.addWidget(QLabel("Instrument:"))
        top_controls.addWidget(self.inst_combo)

        self.auto_trade_btn = QPushButton("Start Auto Trade")
        self.auto_trade_btn.setCheckable(True)
        self.auto_trade_btn.clicked.connect(self.toggle_auto_trade)
        top_controls.addWidget(self.auto_trade_btn)

        self.buy_btn = QPushButton("BUY")
        self.buy_btn.clicked.connect(self.manual_buy)
        top_controls.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("SELL")
        self.sell_btn.clicked.connect(self.manual_sell)
        top_controls.addWidget(self.sell_btn)

        self.balance_label = QLabel("Balance: Fetching...")
        self.balance_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_controls.addWidget(self.balance_label, stretch=1)

        layout.addLayout(top_controls)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        layout.addWidget(self.plot_widget, stretch=5)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.show()

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_box.append(timestamp + message)
        print("[LOG]", message)

    def change_instrument(self, new_instrument):
        self.instrument = new_instrument
        self.signal_gen.instrument = new_instrument
        self.log(f"Switched to instrument: {new_instrument}")
        self.update_all()

    def toggle_auto_trade(self):
        self.auto_trade_enabled = self.auto_trade_btn.isChecked()
        if self.auto_trade_enabled:
            self.log("Auto trading started.")
            threading.Thread(target=self.auto_trade_loop, daemon=True).start()
            self.auto_trade_btn.setText("Stop Auto Trade")
        else:
            self.log("Auto trading stopped.")
            self.auto_trade_btn.setText("Start Auto Trade")

    def auto_trade_loop(self):
        while self.auto_trade_enabled:
            try:
                df = self.signal_gen.generate_signals_live()
                if df.empty:
                    self.log("No data for auto trading.")
                    time.sleep(10)
                    continue

                latest = df.iloc[-1]
                signal = latest["signal"]
                confidence = latest["confidence"]
                price = latest["close"]

                sl_pips = 100
                tp_pips = 100
                units = 1000

                if signal == "buy":
                    sl = price - sl_pips * 0.0001
                    tp = price + tp_pips * 0.0001
                    self.log(f"Auto BUY @ {price:.5f} (conf: {confidence:.2f})")
                    self.api.place_market_order(self.instrument, units, sl, tp)

                elif signal == "sell":
                    sl = price + sl_pips * 0.0001
                    tp = price - tp_pips * 0.0001
                    self.log(f"Auto SELL @ {price:.5f} (conf: {confidence:.2f})")
                    self.api.place_market_order(self.instrument, -units, sl, tp)

                else:
                    self.log("Auto trade: no buy/sell signal.")

            except Exception as e:
                self.log(f"Auto trade error: {e}")

            time.sleep(30)

    def manual_buy(self):
        try:
            df = self.signal_gen.generate_signals_live()
            if df.empty:
                self.log("No data for manual buy.")
                return
            price = df.iloc[-1]["close"]
            sl = price - 0.01
            tp = price + 0.01
            units = 1000
            self.api.place_market_order(self.instrument, units, sl, tp)
            self.log(f"Manual BUY placed @ {price:.5f}")
        except Exception as e:
            self.log(f"Manual buy error: {e}")

    def manual_sell(self):
        try:
            df = self.signal_gen.generate_signals_live()
            if df.empty:
                self.log("No data for manual sell.")
                return
            price = df.iloc[-1]["close"]
            sl = price + 0.01
            tp = price - 0.01
            units = 1000
            self.api.place_market_order(self.instrument, -units, sl, tp)
            self.log(f"Manual SELL placed @ {price:.5f}")
        except Exception as e:
            self.log(f"Manual sell error: {e}")

    def update_all(self):
        self.update_balance()
        self.update_chart()

    def update_balance(self):
        try:
            summary = self.api.get_account_summary()
            balance = summary["account"]["balance"]
            self.balance_label.setText(f"Balance: {balance}")
            
            # Get and log leverage
            details = self.api.get_account_details()
            margin_rate = float(details['account']['marginRate'])
            leverage = int(1 / margin_rate)
            self.log(f"Account leverage: {leverage}:1")

        except Exception as e:
            self.log(f"Failed to fetch balance: {e}")
            self.balance_label.setText("Balance: Error")

    def update_chart(self):
        self.log(f"Fetching data for {self.instrument}...")
        try:
            df = self.signal_gen.generate_signals_live()
            if df.empty:
                self.log("No data for chart.")
                return

            print("[DEBUG] Close prices:", df["close"].tail(10).tolist())

            self.plot_widget.clear()

            times = pd.to_datetime(df["time"])
            x = np.arange(len(times))

            self.plot_widget.plot(x, df["close"], pen=pg.mkPen(color="black", width=2), name="Close")

            if "ema20" in df.columns:
                self.plot_widget.plot(x, df["ema20"], pen=pg.mkPen(color="blue", width=1), name="EMA20")

            if "rsi" in df.columns:
                rsi_scaled = (df["rsi"] - 50) / 50 * max(df["close"]) * 0.5 + min(df["close"])
                self.plot_widget.plot(x, rsi_scaled, pen=pg.mkPen(color="green", width=1), name="RSI")

            if "macd" in df.columns and "macd_signal" in df.columns:
                macd_range = df["macd"].max() - df["macd"].min()
                macd_scaled = (df["macd"] - df["macd"].min()) / macd_range * max(df["close"]) * 0.5 + min(df["close"])
                macd_signal_scaled = (df["macd_signal"] - df["macd_signal"].min()) / macd_range * max(df["close"]) * 0.5 + min(df["close"])
                self.plot_widget.plot(x, macd_scaled, pen=pg.mkPen(color="purple", width=1), name="MACD")
                self.plot_widget.plot(x, macd_signal_scaled, pen=pg.mkPen(color="magenta", width=1), name="MACD Signal")

            buys = df[df["signal"] == "buy"]
            if not buys.empty:
                buy_x = [df.index.get_loc(i) for i in buys.index]
                buy_y = buys["close"].values
                self.plot_widget.plot(buy_x, buy_y, pen=None, symbol='t', symbolBrush='g', symbolSize=14, name="Buy")

            sells = df[df["signal"] == "sell"]
            if not sells.empty:
                sell_x = [df.index.get_loc(i) for i in sells.index]
                sell_y = sells["close"].values
                self.plot_widget.plot(sell_x, sell_y, pen=None, symbol='t1', symbolBrush='r', symbolSize=14, name="Sell")

            self.plot_widget.setLabel('bottom', 'Time')
            self.plot_widget.setLabel('left', 'Price')
            self.plot_widget.showGrid(x=True, y=True)
            self.plot_widget.enableAutoRange(axis='y')
            y_min = df["close"].min()
            y_max = df["close"].max()
            self.plot_widget.setYRange(y_min * 0.9995, y_max * 1.0005)

            self.log("Charts updated.")

        except Exception as e:
            self.log(f"Chart update error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingBotGUI()
    sys.exit(app.exec_())
