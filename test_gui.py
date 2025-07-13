import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg
import numpy as np

class SimpleTradingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Trading Bot Visualization")

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Price plot
        self.price_plot = pg.PlotWidget(title="Price Chart")
        layout.addWidget(self.price_plot)

        # RSI plot
        self.rsi_plot = pg.PlotWidget(title="RSI")
        layout.addWidget(self.rsi_plot)
        self.rsi_plot.setYRange(0, 100)

        # MACD plot
        self.macd_plot = pg.PlotWidget(title="MACD")
        layout.addWidget(self.macd_plot)

        # Generate dummy data
        x = np.arange(100)
        price = np.sin(x * 0.1) * 10 + 100
        rsi = 50 + 40 * np.sin(x * 0.15)
        macd = np.sin(x * 0.07)
        signal = np.cos(x * 0.07)

        # Plot data
        self.price_plot.plot(x, price, pen='w')
        self.rsi_plot.plot(x, rsi, pen='m')
        self.macd_plot.plot(x, macd, pen='b')
        self.macd_plot.plot(x, signal, pen='y')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleTradingGUI()
    window.show()
    sys.exit(app.exec_())
