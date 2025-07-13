import sys
from PyQt5.QtWidgets import QApplication, QWidget
import pyqtgraph as pg

app = QApplication(sys.argv)
w = pg.PlotWidget()
w.show()
sys.exit(app.exec_())
