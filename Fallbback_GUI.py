import sys
import datetime
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from api import OandaAPI
from signal_generator import SignalGenerator

class TradingBotGUI(QWidget):
    def __init__(self):
        super().__init__()
        print("[GUI] Initializing TradingBotGUI")
        self.api = OandaAPI()
        self.instrument = "EUR_USD"
        self.signal_gen = SignalGenerator(api=self.api, instrument=self.instrument)
        print("[SignalGenerator] Initialized for", self.instrument)

        self.init_ui()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_all)
        self.refresh_timer.start(60000)

        self.refresh_all()
        self.start_auto_trade()  # Auto-start trading if within active hours

    def init_ui(self):
        print("[GUI] Setting up UI...")
        self.setWindowTitle("Forex Trading Bot")
        self.setGeometry(100, 100, 1200, 900)
        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.log_tab = QWidget()
        self.tabs.addTab(self.log_tab, "Log")
        self.init_log_tab()

        self.chart_tab = QWidget()
        self.tabs.addTab(self.chart_tab, "Charts")
        self.init_chart_tab()

        self.account_tab = QWidget()
        self.tabs.addTab(self.account_tab, "Account Info")
        self.init_account_tab()

        self.control_tab = QWidget()
        self.tabs.addTab(self.control_tab, "Controls")
        self.init_control_tab()

        self.setLayout(main_layout)
        print("[GUI] UI setup complete.")

    def init_log_tab(self):
        layout = QVBoxLayout()
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        layout.addWidget(self.log_window)
        self.log_tab.setLayout(layout)

    def init_chart_tab(self):
        layout = QVBoxLayout()
        self.chart_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        layout.addWidget(self.chart_canvas)
        self.ax_price = self.chart_canvas.figure.add_subplot(311)
        self.ax_rsi = self.chart_canvas.figure.add_subplot(312, sharex=self.ax_price)
        self.ax_atr = self.chart_canvas.figure.add_subplot(313, sharex=self.ax_price)
        self.chart_tab.setLayout(layout)

    def init_account_tab(self):
        layout = QVBoxLayout()
        self.balance_label = QLabel("Balance: Loading...")
        self.open_positions_table = QTableWidget()
        self.open_positions_table.setColumnCount(4)
        self.open_positions_table.setHorizontalHeaderLabels(["Instrument", "Units", "Price", "Unrealized P/L"])
        layout.addWidget(self.balance_label)
        layout.addWidget(self.open_positions_table)
        self.account_tab.setLayout(layout)

    def init_control_tab(self):
        layout = QVBoxLayout()

        inst_layout = QHBoxLayout()
        inst_label = QLabel("Instrument:")
        self.inst_combo = QComboBox()
        self.inst_combo.addItems(["EUR_USD", "USD_JPY", "GBP_USD"])
        self.inst_combo.currentIndexChanged.connect(self.on_instrument_change)
        inst_layout.addWidget(inst_label)
        inst_layout.addWidget(self.inst_combo)
        layout.addLayout(inst_layout)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Auto Trade")
        self.start_btn.clicked.connect(self.start_auto_trade)
        self.stop_btn = QPushButton("Stop Auto Trade")
        self.stop_btn.clicked.connect(self.stop_auto_trade)
        self.close_btn = QPushButton("Close Position")
        self.close_btn.clicked.connect(self.close_position)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

        self.control_tab.setLayout(layout)

    def log(self, message):
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("[%Y-%m-%d %H:%M:%S]")
        self.log_window.append(f"{timestamp} {message}")
        print("[LOG]", message)

    def on_instrument_change(self):
        self.instrument = self.inst_combo.currentText()
        self.signal_gen.instrument = self.instrument
        self.log(f"Instrument changed to {self.instrument}")
        self.refresh_all()

    def start_auto_trade(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        pst_offset = datetime.timedelta(hours=-8)
        pst_now = now + pst_offset
        start_hour, end_hour = 5, 9

        if start_hour <= pst_now.hour < end_hour:
            self.log("Starting auto trading...")
            # Add actual trade start logic here
        else:
            self.log("Outside active trading hours (05:00–09:00 PST). Auto trading NOT started.")

    def stop_auto_trade(self):
        self.log("Stopping auto trading...")

    def close_position(self):
        self.log(f"Closing position on {self.instrument}")
        try:
            self.api.close_position(self.instrument)
            self.log("Position closed successfully.")
            self.refresh_account_info()
        except Exception as e:
            self.log(f"Error closing position: {e}")

    def refresh_all(self):
        self.update_charts()
        self.refresh_account_info()

    def update_charts(self):
        self.log(f"Fetching data for {self.instrument}...")
        try:
            df = self.signal_gen.generate_signals()
            if df.empty:
                self.log("No data available for plotting.")
                return

            df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

            if "rsi" not in df.columns:
                df["rsi"] = self.signal_gen.calculate_rsi(df["close"])

            if "atr" not in df.columns:
                df["atr"] = self.signal_gen.calculate_atr(df)

            self.ax_price.clear()
            self.ax_price.plot(df["time"], df["close"], label="Close Price", color="blue")
            self.ax_price.plot(df["time"], df["ema12"], label="EMA12", color="orange")
            self.ax_price.plot(df["time"], df["ema26"], label="EMA26", color="purple")

            buys = df[df["signal_type"] == "buy"]
            sells = df[df["signal_type"] == "sell"]
            self.ax_price.scatter(buys["time"], buys["close"], marker="^", color="green", label="Buy", s=100)
            self.ax_price.scatter(sells["time"], sells["close"], marker="v", color="red", label="Sell", s=100)

            self.ax_price.set_title(f"{self.instrument} Price & Signals")
            self.ax_price.set_ylabel("Price")
            self.ax_price.legend()

            self.ax_rsi.clear()
            self.ax_rsi.plot(df["time"], df["rsi"], label="RSI", color="magenta")
            self.ax_rsi.axhline(70, color="red", linestyle="--")
            self.ax_rsi.axhline(30, color="green", linestyle="--")
            self.ax_rsi.set_ylabel("RSI")
            self.ax_rsi.legend()

            self.ax_atr.clear()
            self.ax_atr.plot(df["time"], df["atr"], label="ATR", color="brown")
            self.ax_atr.set_ylabel("ATR")
            self.ax_atr.set_xlabel("Time")
            self.ax_atr.legend()

            self.chart_canvas.draw()
            self.log(f"Data and signals updated for {self.instrument}")

        except Exception as e:
            self.log(f"Error updating charts: {e}")

    def refresh_account_info(self):
        self.log("Refreshing account info...")
        try:
            account = self.api.get_account_summary()
            balance = account.get("balance", "N/A")
            self.balance_label.setText(f"Balance: {balance}")

            positions = self.api.get_open_positions() if hasattr(self.api, "get_open_positions") else []
            self.open_positions_table.setRowCount(len(positions))

            for row, pos in enumerate(positions):
                self.open_positions_table.setItem(row, 0, QTableWidgetItem(pos.get("instrument", "")))
                self.open_positions_table.setItem(row, 1, QTableWidgetItem(str(pos.get("units", ""))))
                self.open_positions_table.setItem(row, 2, QTableWidgetItem(str(pos.get("price", ""))))
                self.open_positions_table.setItem(row, 3, QTableWidgetItem(str(pos.get("unrealizedPL", ""))))

            self.open_positions_table.resizeColumnsToContents()
            self.log("Account info updated.")
        except Exception as e:
            self.log(f"Error refreshing account info: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingBotGUI()
    window.show()
    sys.exit(app.exec_())
