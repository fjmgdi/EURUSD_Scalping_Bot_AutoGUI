import sys
import logging
from datetime import datetime
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QTextEdit, QHBoxLayout, QLabel, QComboBox
)
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OANDA_API_KEY = "YOUR_OANDA_API_KEY"
ACCOUNT_ID = "YOUR_OANDA_ACCOUNT_ID"
ENVIRONMENT = "practice"  # or "live"

class TradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OANDA Live Price Chart with Markers")
        self.resize(900, 600)

        # OANDA API client
        self.client = API(access_token=OANDA_API_KEY, environment=ENVIRONMENT)

        # Selected currency pair
        self.pairs = ["EUR_USD", "CAD_JPY", "GBP_JPY"]
        self.selected_pair = self.pairs[0]

        # UI Components
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controls
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        control_layout.addWidget(QLabel("Select Pair:"))
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(self.pairs)
        control_layout.addWidget(self.pair_combo)

        self.btn_fetch = QPushButton("Fetch & Plot")
        control_layout.addWidget(self.btn_fetch)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output, stretch=1)

        # Matplotlib Figure and Canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=4)

        # Connect signals
        self.btn_fetch.clicked.connect(self.fetch_and_plot)
        self.pair_combo.currentIndexChanged.connect(self.pair_changed)

        # Sample buy/sell signals (for demonstration)
        self.buy_times = []
        self.buy_prices = []
        self.sell_times = []
        self.sell_prices = []

    def pair_changed(self, index):
        self.selected_pair = self.pairs[index]
        self.log(f"Pair changed to {self.selected_pair}")

    def log(self, msg):
        self.log_output.append(msg)
        logger.info(msg)

    def fetch_candles(self):
        params = {"count": 100, "granularity": "M5"}
        candles_req = InstrumentsCandles(instrument=self.selected_pair, params=params)
        try:
            resp = self.client.request(candles_req)
            self.log(f"Fetched {len(resp['candles'])} candles for {self.selected_pair}")
        except Exception as e:
            self.log(f"Error fetching candles: {e}")
            return pd.DataFrame()  # empty DF on error

        times = []
        closes = []

        for candle in resp.get('candles', []):
            if candle['complete']:
                time_str = candle['time'].replace("Z", "")
                times.append(datetime.fromisoformat(time_str))
                closes.append(float(candle['mid']['c']))

        df = pd.DataFrame({"time": times, "close": closes}).set_index("time")
        return df

    def generate_sample_signals(self, df):
        # Simple demo signals: buy when price up 0.1% from prev close, sell when down 0.1%
        self.buy_times.clear()
        self.buy_prices.clear()
        self.sell_times.clear()
        self.sell_prices.clear()

        for i in range(1, len(df)):
            change = (df['close'].iloc[i] - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
            if change > 0.001:
                self.buy_times.append(df.index[i])
                self.buy_prices.append(df['close'].iloc[i])
            elif change < -0.001:
                self.sell_times.append(df.index[i])
                self.sell_prices.append(df['close'].iloc[i])

    def fetch_and_plot(self):
        df = self.fetch_candles()
        if df.empty:
            self.log("No data to plot")
            return

        self.generate_sample_signals(df)

        self.ax.clear()
        self.ax.plot(df.index, df['close'], label="Close Price", color='blue')
        self.ax.set_title(f"{self.selected_pair} Close Price (M5)")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)

        # Format x-axis dates
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        self.figure.autofmt_xdate()

        # Plot buy signals as green triangles up
        self.ax.scatter(self.buy_times, self.buy_prices, marker='^', color='green', label='Buy', zorder=5)

        # Plot sell signals as red triangles down
        self.ax.scatter(self.sell_times, self.sell_prices, marker='v', color='red', label='Sell', zorder=5)

        self.ax.legend()
        self.canvas.draw()
        self.log(f"Plotted {len(df)} candles with {len(self.buy_times)} buy and {len(self.sell_times)} sell markers")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingGUI()
    window.show()
    sys.exit(app.exec_())
