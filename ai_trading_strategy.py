import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import joblib
import shap
from dotenv import load_dotenv
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeClose, OpenTrades
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit, QDoubleSpinBox,
    QFormLayout, QTabWidget, QCheckBox, QComboBox, QMessageBox,
    QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog, QSpinBox
)
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mplfinance.original_flavor import candlestick_ohlc

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("backups/models", exist_ok=True)

class DataFetcher(QThread):
    data_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api, symbol="EUR_USD", granularities=["M5"], count=500):
        super().__init__()
        self.api = api
        self.symbol = symbol
        self.granularities = granularities
        self.count = count
        self.running = False
        self.last_update = {}
        
    def process_candles(self, candles):
        try:
            if not candles or 'candles' not in candles:
                raise ValueError("No candle data received")
                
            data = []
            for candle in candles['candles']:
                if not candle['complete']:
                    continue
                    
                candle_data = {
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                }
                data.append(candle_data)
                
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error processing candles: {str(e)}")
            raise
            
    def run(self):
        self.running = True
        logger.info(f"Data fetcher started for {self.symbol}")
        
        while self.running:
            try:
                data = {}
                for granularity in self.granularities:
                    params = {
                        "granularity": granularity,
                        "count": self.count,
                        "price": "M"
                    }
                    endpoint = InstrumentsCandles(instrument=self.symbol, params=params)
                    response = self.api.request(endpoint)
                    df = self.process_candles(response)
                    data[granularity] = df
                
                if data:
                    self.data_updated.emit(data)
                    self.last_update = data
                    
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in data fetcher: {str(e)}")
                self.error_occurred.emit(str(e))
                time.sleep(30)
    
    def stop(self):
        self.running = False
        logger.info("Data fetcher stopped")

class AITradingStrategy:
    def __init__(self, symbol="EUR_USD", model_path=None):
        self.symbol = symbol
        self.model = None
        self.indicators = {}
        self.current_position = None
        self.position_size = 0
        self.entry_price = 0
        self.daily_pnl = 0
        self.total_pnl = 0
        self.win_rate = 0
        self.trade_count = 0
        self.win_count = 0
        self.last_signal = None
        self.min_confidence = 0.65
        self.base_atr_multiplier = 1.8
        self.auto_trading = True
        self.feature_names = [
            'RSI', 'MACD', 'Stoch_K', 'Stoch_D', 'ADX', 
            'ATR', 'VWAP', 'BB_Upper_Dev', 'BB_Lower_Dev', 
            'EMA_Diff', 'Price_EMA50_Diff', 'Price_EMA200_Diff',
            'BB_Squeeze', 'Volume_Spike'
        ]
        self.features = []
        self.backtesting = False
        
        if model_path:
            self._load_model(model_path)
        else:
            self._train_model()
            
    def _train_model(self):
        try:
            # Enhanced synthetic data generation
            np.random.seed(42)
            n_samples = 10000
            X = np.zeros((n_samples, len(self.feature_names)))
            
            for i in range(n_samples):
                trend = np.sin(i / 100) * 0.5 + 0.5
                noise = np.random.normal(0, 0.1, len(self.feature_names))
                
                # Momentum features
                X[i, 0] = trend * 0.8 + noise[0]  # RSI
                X[i, 1] = np.sin(i / 50) * 0.3 + noise[1]  # MACD
                X[i, 2:4] = trend * 0.6 + noise[2:4]  # Stochastic
                
                # Trend features
                X[i, 4] = np.abs(trend * 0.7 + noise[4])  # ADX
                
                # Volatility features
                X[i, 5] = 0.02 + np.abs(noise[5])  # ATR
                X[i, 7:9] = trend * 0.5 + noise[7:9]  # Bollinger
                
                # Volume features
                X[i, 6] = trend * 0.9 + noise[6]  # VWAP
                
                # Additional features
                X[i, 9] = np.sin(i / 70) * 0.4 + noise[9]  # EMA diff
                X[i, 10] = trend * 0.3 + noise[10]  # Price-EMA50 diff
                X[i, 11] = trend * 0.2 + noise[11]  # Price-EMA200 diff
                X[i, 12] = 1.0 + noise[12]  # BB squeeze ratio
                X[i, 13] = 1.0 + abs(noise[13])  # Volume spike
            
            # Define conditions separately for clarity
            momentum_condition = (X[:, 0] > 0.65) & (X[:, 1] > 0)
            trend_condition = (X[:, 4] > 0.5) & (X[:, 10] > 0.1) & (X[:, 11] > 0.05)
            oversold_condition = (X[:, 2] < 0.2) & (X[:, 0] < 0.35) & (X[:, 12] > 1.5)
            
            # Combine conditions
            y = np.where(
                momentum_condition | trend_condition | oversold_condition,
                1, 
                0
            )
            
            # Add noise to targets
            flip_mask = np.random.random(n_samples) < 0.03
            y[flip_mask] = 1 - y[flip_mask]
            
            # Split into train and validation sets
            val_size = int(0.2 * n_samples)
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
            
            # Enhanced model parameters
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.75,
                min_child_samples=30,
                reg_alpha=0.2,
                reg_lambda=0.2,
                random_state=42,
                verbose=-1,
                class_weight='balanced'
            )
            
            # Fit the model with validation
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
                callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=True)]
            )
            
            # Create pipeline with trained model
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('lgbm', model)
            ])
            
            logger.info("Model trained successfully with enhanced synthetic data")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def _load_model(self, path):
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._train_model()
            
    def get_atr_multiplier(self):
        try:
            if 'atr' not in self.indicators or 'bollinger_upper' not in self.indicators:
                return self.base_atr_multiplier
                
            bb_width = (self.indicators['bollinger_upper'].iloc[-1] - 
                       self.indicators['bollinger_lower'].iloc[-1])
            
            # High volatility - use tighter stops
            if bb_width > 0.0020:  # 20 pips
                return max(1.2, self.base_atr_multiplier * 0.7)
            # Low volatility - wider stops
            elif bb_width < 0.0010:  # 10 pips
                return min(3.0, self.base_atr_multiplier * 1.3)
            else:
                return self.base_atr_multiplier
        except:
            return self.base_atr_multiplier
            
    def calculate_indicators(self, df):
        try:
            if len(df) < 200:
                logger.warning("Not enough data points for all indicators")
                return False
                
            # Data quality checks
            if df[['open', 'high', 'low', 'close']].isnull().values.any():
                logger.warning("NaN values detected in price data")
                return False
                
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                logger.warning("Invalid price values (zero or negative)")
                return False
                
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Momentum Indicators
            self.indicators['rsi'] = RSIIndicator(close=close, window=14).rsi()
            self.indicators['macd'] = MACD(
                close=close, 
                window_slow=26, 
                window_fast=12, 
                window_sign=9
            ).macd()
            
            stoch = StochasticOscillator(
                high=high, low=low, close=close, 
                window=14, smooth_window=3
            )
            self.indicators['stoch_k'] = stoch.stoch()
            self.indicators['stoch_d'] = stoch.stoch_signal()
            
            # Trend Indicators
            self.indicators['adx'] = ADXIndicator(
                high=high, low=low, close=close, window=14
            ).adx()
            
            self.indicators['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator()
            self.indicators['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator()
            
            # Volatility Indicators
            self.indicators['atr'] = AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            
            bb = BollingerBands(close=close, window=20, window_dev=2)
            self.indicators['bollinger_upper'] = bb.bollinger_hband()
            self.indicators['bollinger_lower'] = bb.bollinger_lband()
            self.indicators['bollinger_mavg'] = bb.bollinger_mavg()
            
            # Volume Indicators
            self.indicators['vwap'] = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=volume, window=20
            ).volume_weighted_average_price()
            
            self.indicators['obv'] = OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume()
            
            # Verify all indicators have values
            for name, values in self.indicators.items():
                if values.empty or values.isnull().any():
                    logger.warning(f"Problem with indicator: {name}")
                    return False
                    
            logger.debug("Indicators calculated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False
            
    def prepare_features(self, df):
        try:
            if not self.calculate_indicators(df):
                raise ValueError("Indicator calculation failed")
                
            # Calculate additional features
            bb_squeeze = ((self.indicators['bollinger_upper'].iloc[-1] - 
                          self.indicators['bollinger_lower'].iloc[-1]) / 
                         self.indicators['atr'].iloc[-1])
                         
            volume_spike = (df['volume'].iloc[-1] / 
                          df['volume'].rolling(20).mean().iloc[-1])
            
            features = [
                self.indicators['rsi'].iloc[-1],
                self.indicators['macd'].iloc[-1],
                self.indicators['stoch_k'].iloc[-1],
                self.indicators['stoch_d'].iloc[-1],
                self.indicators['adx'].iloc[-1],
                self.indicators['atr'].iloc[-1],
                self.indicators['vwap'].iloc[-1],
                self.indicators['bollinger_upper'].iloc[-1] - df['close'].iloc[-1],
                df['close'].iloc[-1] - self.indicators['bollinger_lower'].iloc[-1],
                self.indicators['ema_50'].iloc[-1] - self.indicators['ema_200'].iloc[-1],
                df['close'].iloc[-1] - self.indicators['ema_50'].iloc[-1],
                df['close'].iloc[-1] - self.indicators['ema_200'].iloc[-1],
                bb_squeeze,
                volume_spike
            ]
            
            feature_array = np.array(features).reshape(1, -1)
            self.features.append(feature_array)
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return np.zeros((1, len(self.feature_names)))
            
    def generate_signal(self, data_dict):
        signal = {
            "signal": "HOLD",
            "reason": "No clear signal",
            "price": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 0.0,
            "time": datetime.now().strftime('%H:%M:%S'),
            "indicators": {}
        }
        
        try:
            if not data_dict or 'M5' not in data_dict:
                signal['reason'] = "No data available"
                return signal
                
            df = data_dict['M5']
            if df.empty:
                signal['reason'] = "Empty data frame"
                return signal
                
            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]
            
            # Trading hours filter (avoid low liquidity periods)
            current_hour = current_time.hour
            if current_hour in {21, 22, 23, 0, 1}:
                signal.update({
                    "signal": "HOLD",
                    "reason": "Avoiding low liquidity hours",
                    "price": current_price
                })
                return signal
                
            features = self.prepare_features(df)
            prediction = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = max(proba)
            
            atr = self.indicators['atr'].iloc[-1]
            atr_multiplier = self.get_atr_multiplier()
            tp = current_price + (atr * atr_multiplier)
            sl = current_price - (atr * atr_multiplier * 0.8)  # Slightly tighter SL
            
            # Prepare indicators for display
            indicators = {}
            for name, values in self.indicators.items():
                try:
                    indicators[name] = values.iloc[-1]
                except Exception as e:
                    logger.error(f"Error getting indicator {name}: {str(e)}")
                    indicators[name] = None
            
            signal.update({
                "price": current_price,
                "tp": tp,
                "sl": sl,
                "confidence": confidence,
                "time": current_time.strftime('%H:%M:%S'),
                "indicators": indicators
            })
            
            # Enhanced signal conditions
            oversold_condition = (
                self.indicators['rsi'].iloc[-1] < 35 and
                self.indicators['stoch_k'].iloc[-1] < 20 and
                self.indicators['stoch_d'].iloc[-1] < 20
            )
            
            bullish_divergence = (
                df['close'].iloc[-1] < df['close'].iloc[-3] and
                self.indicators['rsi'].iloc[-1] > self.indicators['rsi'].iloc[-3]
            )
            
            trend_strength = self.indicators['adx'].iloc[-1]
            ema_direction = (self.indicators['ema_50'].iloc[-1] > 
                           self.indicators['ema_50'].iloc[-3])
            
            # Buy signal conditions
            if (prediction == 1 and 
                confidence >= self.min_confidence and
                (oversold_condition or bullish_divergence) and
                current_price > self.indicators['ema_200'].iloc[-1] and
                trend_strength > 15):
                
                signal.update({
                    "signal": "BUY",
                    "reason": f"Strong buy signal (RSI: {self.indicators['rsi'].iloc[-1]:.2f}, Stoch: {self.indicators['stoch_k'].iloc[-1]:.2f})"
                })
                
            # Sell/Close conditions
            elif (self.current_position == "BUY" and 
                  (confidence < 0.4 or 
                   self.indicators['rsi'].iloc[-1] > 70 or
                   current_price < self.indicators['ema_50'].iloc[-1])):
                
                signal.update({
                    "signal": "CLOSE",
                    "reason": "Exit conditions met"
                })
                
            self.last_signal = signal
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            signal.update({
                "signal": "ERROR",
                "reason": str(e)
            })
            return signal

class TradingBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Forex Trading Bot")
        self.setWindowIcon(QIcon("icon.png"))
        self.resize(1400, 900)
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.api = API(access_token=self.api_key)
        
        # Initialize components
        self.data_fetcher = None
        self.strategy = AITradingStrategy()
        self.timer = QTimer(self)
        self.settings = QSettings("AI_Trading_Bot", "Forex_Trading_Bot")
        self.bot_running = False
        self.current_data = None
        
        # Create UI
        self.create_ui()
        self.load_settings()
        
        # Connect signals
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(1000)
        
    def create_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Connection group
        connection_group = QGroupBox("Connection Settings")
        connection_layout = QFormLayout(connection_group)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.account_id_input = QLineEdit()
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_GBP"])
        self.granularity_combo = QComboBox()
        self.granularity_combo.addItems(["M1", "M5", "M15", "H1", "H4", "D1"])
        self.granularity_combo.setCurrentText("M5")
        
        connection_layout.addRow("API Key:", self.api_key_input)
        connection_layout.addRow("Account ID:", self.account_id_input)
        connection_layout.addRow("Symbol:", self.symbol_combo)
        connection_layout.addRow("Granularity:", self.granularity_combo)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_api)
        
        # Bot control group
        control_group = QGroupBox("Bot Controls")
        control_layout = QHBoxLayout(control_group)
        
        self.start_btn = QPushButton("Start Bot")
        self.start_btn.setStyleSheet("background-color: green; color: white;")
        self.start_btn.clicked.connect(self.start_bot)
        
        self.stop_btn = QPushButton("Stop Bot")
        self.stop_btn.setStyleSheet("background-color: red; color: white;")
        self.stop_btn.clicked.connect(self.stop_bot)
        self.stop_btn.setEnabled(False)
        
        self.debug_btn = QPushButton("Debug")
        self.debug_btn.clicked.connect(self.debug_indicator_calculation)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.debug_btn)
        
        # Strategy group
        strategy_group = QGroupBox("Trading Strategy")
        strategy_layout = QFormLayout(strategy_group)
        
        self.auto_trading_check = QCheckBox("Auto Trading")
        self.auto_trading_check.setChecked(True)
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.1, 1.0)
        self.min_confidence_spin.setValue(0.65)
        self.min_confidence_spin.setSingleStep(0.01)
        self.atr_multiplier_spin = QDoubleSpinBox()
        self.atr_multiplier_spin.setRange(0.5, 5.0)
        self.atr_multiplier_spin.setValue(1.8)
        self.atr_multiplier_spin.setSingleStep(0.1)
        
        strategy_layout.addRow(self.auto_trading_check)
        strategy_layout.addRow("Min Confidence:", self.min_confidence_spin)
        strategy_layout.addRow("Base ATR Multiplier:", self.atr_multiplier_spin)
        
        self.save_strategy_btn = QPushButton("Save Strategy")
        self.save_strategy_btn.clicked.connect(self.save_strategy_settings)
        
        # Trading controls
        trading_group = QGroupBox("Trading Controls")
        trading_layout = QVBoxLayout(trading_group)
        
        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet("background-color: green; color: white;")
        self.buy_btn.clicked.connect(self.manual_buy)
        
        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet("background-color: red; color: white;")
        self.sell_btn.clicked.connect(self.manual_sell)
        
        self.close_btn = QPushButton("CLOSE POSITION")
        self.close_btn.setStyleSheet("background-color: gray; color: white;")
        self.close_btn.clicked.connect(self.close_position)
        
        trading_layout.addWidget(self.buy_btn)
        trading_layout.addWidget(self.sell_btn)
        trading_layout.addWidget(self.close_btn)
        
        # Model controls
        model_group = QGroupBox("Model Management")
        model_layout = QVBoxLayout(model_group)
        
        self.train_model_btn = QPushButton("Train New Model")
        self.train_model_btn.clicked.connect(self.train_model)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        
        self.explain_btn = QPushButton("Explain Prediction")
        self.explain_btn.clicked.connect(self.explain_prediction)
        
        model_layout.addWidget(self.train_model_btn)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.save_model_btn)
        model_layout.addWidget(self.explain_btn)
        
        # Performance stats
        stats_group = QGroupBox("Performance Stats")
        stats_layout = QFormLayout(stats_group)
        
        self.win_rate_label = QLabel("0.00%")
        self.daily_pnl_label = QLabel("0.00")
        self.total_pnl_label = QLabel("0.00")
        self.trade_count_label = QLabel("0")
        
        stats_layout.addRow("Win Rate:", self.win_rate_label)
        stats_layout.addRow("Today's P/L:", self.daily_pnl_label)
        stats_layout.addRow("Total P/L:", self.total_pnl_label)
        stats_layout.addRow("Total Trades:", self.trade_count_label)
        
        # Add groups to left panel
        left_layout.addWidget(connection_group)
        left_layout.addWidget(control_group)
        left_layout.addWidget(strategy_group)
        left_layout.addWidget(trading_group)
        left_layout.addWidget(model_group)
        left_layout.addWidget(stats_group)
        left_layout.addStretch()
        
        # Right panel (charts and info)
        right_panel = QTabWidget()
        
        # Chart tab
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        
        self.chart_canvas = FigureCanvas(plt.figure(figsize=(12, 7)))
        chart_layout.addWidget(self.chart_canvas)
        
        # Signal tab
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        
        self.signal_text = QTextEdit()
        self.signal_text.setReadOnly(True)
        signal_layout.addWidget(self.signal_text)
        
        # Indicators tab
        indicators_tab = QWidget()
        indicators_layout = QVBoxLayout(indicators_tab)
        
        self.indicators_table = QTableWidget()
        self.indicators_table.setColumnCount(2)
        self.indicators_table.setHorizontalHeaderLabels(["Indicator", "Value"])
        self.indicators_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        indicators_layout.addWidget(self.indicators_table)
        
        # Trades tab
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels(["Time", "Type", "Price", "Size", "P/L", "Duration", "Status"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
        
        # Add tabs
        right_panel.addTab(chart_tab, "Chart")
        right_panel.addTab(signal_tab, "Signals")
        right_panel.addTab(indicators_tab, "Indicators")
        right_panel.addTab(trades_tab, "Trades")
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def load_settings(self):
        self.api_key_input.setText(self.settings.value("api_key", ""))
        self.account_id_input.setText(self.settings.value("account_id", ""))
        self.symbol_combo.setCurrentText(self.settings.value("symbol", "EUR_USD"))
        self.granularity_combo.setCurrentText(self.settings.value("granularity", "M5"))
        self.auto_trading_check.setChecked(self.settings.value("auto_trading", True, type=bool))
        self.min_confidence_spin.setValue(self.settings.value("min_confidence", 0.65, type=float))
        self.atr_multiplier_spin.setValue(self.settings.value("atr_multiplier", 1.8, type=float))
        
    def save_settings(self):
        self.settings.setValue("api_key", self.api_key_input.text())
        self.settings.setValue("account_id", self.account_id_input.text())
        self.settings.setValue("symbol", self.symbol_combo.currentText())
        self.settings.setValue("granularity", self.granularity_combo.currentText())
        self.settings.setValue("auto_trading", self.auto_trading_check.isChecked())
        self.settings.setValue("min_confidence", self.min_confidence_spin.value())
        self.settings.setValue("atr_multiplier", self.atr_multiplier_spin.value())
        
    def connect_to_api(self):
        try:
            self.api_key = self.api_key_input.text()
            self.account_id = self.account_id_input.text()
            
            if not self.api_key or not self.account_id:
                QMessageBox.warning(self, "Error", "API Key and Account ID are required")
                return
                
            self.api = API(access_token=self.api_key)
            
            # Test connection
            endpoint = AccountDetails(accountID=self.account_id)
            response = self.api.request(endpoint)
            
            if 'account' not in response:
                raise ValueError("Invalid account details")
                
            self.save_settings()
            QMessageBox.information(self, "Success", "Connected to API successfully")
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")
            
    def start_bot(self):
        try:
            if not self.api_key or not self.account_id:
                QMessageBox.warning(self, "Error", "Please connect to API first")
                return
                
            if self.data_fetcher and self.data_fetcher.isRunning():
                QMessageBox.information(self, "Info", "Bot is already running")
                return
                
            # Update strategy parameters
            self.strategy.min_confidence = self.min_confidence_spin.value()
            self.strategy.base_atr_multiplier = self.atr_multiplier_spin.value()
            self.strategy.auto_trading = self.auto_trading_check.isChecked()
            self.strategy.symbol = self.symbol_combo.currentText()
            
            # Start data fetcher
            self.data_fetcher = DataFetcher(
                api=self.api,
                symbol=self.symbol_combo.currentText(),
                granularities=[self.granularity_combo.currentText()],
                count=500
            )
            
            self.data_fetcher.data_updated.connect(self.handle_new_data)
            self.data_fetcher.error_occurred.connect(self.handle_error)
            self.data_fetcher.start()
            
            self.bot_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.connect_btn.setEnabled(False)
            
            logger.info("Trading bot started")
            self.signal_text.append(f"{datetime.now().strftime('%H:%M:%S')} - Bot started")
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start bot: {str(e)}")
            
    def stop_bot(self):
        try:
            if self.data_fetcher:
                self.data_fetcher.stop()
                self.data_fetcher.wait()
                
            self.bot_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.connect_btn.setEnabled(True)
            
            logger.info("Trading bot stopped")
            self.signal_text.append(f"{datetime.now().strftime('%H:%M:%S')} - Bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to stop bot: {str(e)}")
            
    def handle_new_data(self, data):
        try:
            self.current_data = data
            self.update_chart()
            
            # Debugging
            logger.debug("New data received - calculating indicators...")
            self.debug_indicator_calculation()
            
            # Force indicator calculation and display
            df = data['M5']
            if self.strategy.calculate_indicators(df):
                indicators = {}
                for name, values in self.strategy.indicators.items():
                    try:
                        if not values.empty:
                            indicators[name] = values.iloc[-1]
                        else:
                            logger.warning(f"Empty values for indicator: {name}")
                    except Exception as e:
                        logger.error(f"Error getting indicator {name}: {str(e)}")
                
                logger.debug(f"Prepared indicators for display: {indicators}")
                self._update_indicators_table(indicators)
            
            if self.auto_trading_check.isChecked() and self.bot_running:
                signal = self.strategy.generate_signal(data)
                self.process_signal(signal)
                
        except Exception as e:
            logger.error(f"Error handling new data: {str(e)}")
            
    def _update_indicators_table(self, indicators):
        """Update indicators table with thread-safe operations"""
        try:
            if not indicators or not isinstance(indicators, dict):
                logger.warning(f"Invalid indicators received: {indicators}")
                return
                
            logger.debug(f"Updating indicators table with indicators: {indicators}")
            
            # Group related indicators together
            indicator_groups = {
                'Momentum': ['rsi', 'macd', 'stoch_k', 'stoch_d'],
                'Trend': ['adx', 'ema_50', 'ema_200'],
                'Volatility': ['atr', 'bollinger_upper', 'bollinger_lower', 'bollinger_mavg'],
                'Volume': ['vwap', 'obv']
            }
            
            self.indicators_table.setRowCount(0)
            self.indicators_table.setRowCount(len(indicators))
            
            row = 0
            for group_name, indicator_names in indicator_groups.items():
                # Add group header
                group_item = QTableWidgetItem(group_name)
                group_item.setFlags(Qt.ItemIsEnabled)
                group_item.setBackground(QColor(240, 240, 240))
                group_item.setForeground(QColor(0, 0, 0))
                font = group_item.font()
                font.setBold(True)
                group_item.setFont(font)
                
                self.indicators_table.setItem(row, 0, group_item)
                self.indicators_table.setSpan(row, 0, 1, 2)
                row += 1
                
                # Add indicators in this group
                for name in indicator_names:
                    if name in indicators:
                        name_item = QTableWidgetItem(name.replace('_', ' ').title())
                        value_item = QTableWidgetItem(f"{indicators[name]:.4f}" if indicators[name] is not None else "N/A")
                        
                        self.indicators_table.setItem(row, 0, name_item)
                        self.indicators_table.setItem(row, 1, value_item)
                        row += 1
            
        except Exception as e:
            logger.error(f"Error updating indicators table: {str(e)}")
            
    def update_chart(self):
        try:
            if not self.current_data or 'M5' not in self.current_data:
                return
                
            df = self.current_data['M5']
            if len(df) < 100:
                return
                
            # Clear previous plot
            plt.clf()
            
            # Prepare data for candlestick chart
            df_plot = df.iloc[-100:].copy()
            df_plot['date_num'] = mdates.date2num(df_plot.index.to_pydatetime())
            
            # Create subplots
            ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4)
            ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, sharex=ax1)
            ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, sharex=ax1)
            
            # Plot candlesticks
            candlestick_ohlc(
                ax1, 
                df_plot[['date_num', 'open', 'high', 'low', 'close']].values,
                width=0.002, 
                colorup='g', 
                colordown='r'
            )
            
            # Plot indicators
            if 'ema_50' in self.strategy.indicators:
                ax1.plot(
                    df_plot.index,
                    self.strategy.indicators['ema_50'].iloc[-100:],
                    'b-',
                    linewidth=1,
                    label='EMA 50'
                )
                
            if 'ema_200' in self.strategy.indicators:
                ax1.plot(
                    df_plot.index,
                    self.strategy.indicators['ema_200'].iloc[-100:],
                    'm-',
                    linewidth=1,
                    label='EMA 200'
                )
                
            if 'bollinger_upper' in self.strategy.indicators:
                ax1.plot(
                    df_plot.index,
                    self.strategy.indicators['bollinger_upper'].iloc[-100:],
                    'c--',
                    linewidth=0.5,
                    label='BB Upper'
                )
                
            if 'bollinger_lower' in self.strategy.indicators:
                ax1.plot(
                    df_plot.index,
                    self.strategy.indicators['bollinger_lower'].iloc[-100:],
                    'c--',
                    linewidth=0.5,
                    label='BB Lower'
                )
                
            # Plot volume
            ax2.bar(
                df_plot.index,
                df_plot['volume'],
                color='k',
                width=0.002
            )
            
            # Plot RSI
            if 'rsi' in self.strategy.indicators:
                ax3.plot(
                    df_plot.index,
                    self.strategy.indicators['rsi'].iloc[-100:],
                    'b-',
                    linewidth=1,
                    label='RSI'
                )
                ax3.axhline(70, color='r', linestyle='--', linewidth=0.5)
                ax3.axhline(30, color='g', linestyle='--', linewidth=0.5)
                ax3.set_ylim(0, 100)
                
            # Formatting
            ax1.set_title(f"{self.strategy.symbol} Price Chart")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax2.set_title('Volume')
            ax3.set_title('RSI')
            
            # Rotate x-axis labels
            for ax in [ax1, ax2, ax3]:
                plt.setp(ax.get_xticklabels(), rotation=45)
                
            plt.tight_layout()
            self.chart_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating chart: {str(e)}")
            
    def process_signal(self, signal):
        try:
            if signal['signal'] == 'HOLD':
                return
                
            self.signal_text.append(
                f"{signal['time']} - {signal['signal']} signal: {signal['reason']} "
                f"(Price: {signal['price']:.5f}, Confidence: {signal['confidence']:.2f})"
            )
            
            if not self.strategy.auto_trading:
                return
                
            # Execute trade based on signal
            if signal['signal'] == 'BUY':
                self.execute_trade(
                    direction='BUY',
                    price=signal['price'],
                    tp=signal['tp'],
                    sl=signal['sl']
                )
            elif signal['signal'] == 'CLOSE' and self.strategy.current_position:
                self.close_position()
                
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
            
    def execute_trade(self, direction, price, tp, sl):
        try:
            if not self.api_key or not self.account_id:
                logger.warning("Cannot execute trade - not connected to API")
                return
                
            # Calculate position size based on account balance and risk
            endpoint = AccountDetails(accountID=self.account_id)
            response = self.api.request(endpoint)
            balance = float(response['account']['balance'])
            
            # Risk 1% of account balance per trade
            risk_amount = balance * 0.01
            risk_per_unit = abs(price - sl)
            units = int(risk_amount / risk_per_unit)
            
            # OANDA requires minimum units based on instrument
            min_units = 1000
            units = max(units, min_units)
            
            # Create order
            order_data = {
                "order": {
                    "instrument": self.strategy.symbol,
                    "units": str(units) if direction == 'BUY' else f"-{units}",
                    "type": "MARKET",
                    "takeProfitOnFill": {
                        "price": str(round(tp, 5))
                    },
                    "stopLossOnFill": {
                        "price": str(round(sl, 5))
                    }
                }
            }
            
            endpoint = OrderCreate(accountID=self.account_id, data=order_data)
            response = self.api.request(endpoint)
            
            if 'orderFillTransaction' in response:
                self.strategy.current_position = direction
                self.strategy.entry_price = price
                self.strategy.position_size = units
                
                self.signal_text.append(
                    f"{datetime.now().strftime('%H:%M:%S')} - "
                    f"Executed {direction} order at {price:.5f} "
                    f"(TP: {tp:.5f}, SL: {sl:.5f})"
                )
                
                logger.info(f"Trade executed: {direction} {units} units at {price}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            self.signal_text.append(f"Trade execution failed: {str(e)}")
            
    def close_position(self):
        try:
            if not self.strategy.current_position:
                return
                
            endpoint = OpenTrades(accountID=self.account_id)
            response = self.api.request(endpoint)
            
            if 'trades' in response and response['trades']:
                for trade in response['trades']:
                    if trade['instrument'] == self.strategy.symbol:
                        endpoint = TradeClose(accountID=self.account_id, tradeID=trade['id'])
                        close_response = self.api.request(endpoint)
                        
                        if 'orderFillTransaction' in close_response:
                            exit_price = float(close_response['orderFillTransaction']['price'])
                            pnl = (exit_price - self.strategy.entry_price) * self.strategy.position_size
                            if self.strategy.current_position == 'SELL':
                                pnl *= -1
                                
                            self.strategy.daily_pnl += pnl
                            self.strategy.total_pnl += pnl
                            self.strategy.trade_count += 1
                            
                            if pnl > 0:
                                self.strategy.win_count += 1
                                
                            self.strategy.win_rate = (
                                self.strategy.win_count / self.strategy.trade_count * 100 
                                if self.strategy.trade_count > 0 else 0
                            )
                            
                            self.strategy.current_position = None
                            self.strategy.entry_price = 0
                            self.strategy.position_size = 0
                            
                            self.signal_text.append(
                                f"{datetime.now().strftime('%H:%M:%S')} - "
                                f"Closed position at {exit_price:.5f} (P/L: {pnl:.2f})"
                            )
                            
                            logger.info(f"Position closed at {exit_price:.5f} (P/L: {pnl:.2f})")
                            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            self.signal_text.append(f"Position close failed: {str(e)}")
            
    def manual_buy(self):
        try:
            if not self.current_data or 'M5' not in self.current_data:
                QMessageBox.warning(self, "Error", "No data available")
                return
                
            df = self.current_data['M5']
            current_price = df['close'].iloc[-1]
            
            if not self.strategy.calculate_indicators(df):
                QMessageBox.warning(self, "Error", "Could not calculate indicators")
                return
                
            atr = self.strategy.indicators['atr'].iloc[-1]
            atr_multiplier = self.strategy.get_atr_multiplier()
            tp = current_price + (atr * atr_multiplier)
            sl = current_price - (atr * atr_multiplier * 0.8)
            
            self.execute_trade(
                direction='BUY',
                price=current_price,
                tp=tp,
                sl=sl
            )
            
        except Exception as e:
            logger.error(f"Error in manual buy: {str(e)}")
            QMessageBox.critical(self, "Error", f"Manual buy failed: {str(e)}")
            
    def manual_sell(self):
        try:
            if not self.current_data or 'M5' not in self.current_data:
                QMessageBox.warning(self, "Error", "No data available")
                return
                
            df = self.current_data['M5']
            current_price = df['close'].iloc[-1]
            
            if not self.strategy.calculate_indicators(df):
                QMessageBox.warning(self, "Error", "Could not calculate indicators")
                return
                
            atr = self.strategy.indicators['atr'].iloc[-1]
            atr_multiplier = self.strategy.get_atr_multiplier()
            tp = current_price - (atr * atr_multiplier)
            sl = current_price + (atr * atr_multiplier * 0.8)
            
            self.execute_trade(
                direction='SELL',
                price=current_price,
                tp=tp,
                sl=sl
            )
            
        except Exception as e:
            logger.error(f"Error in manual sell: {str(e)}")
            QMessageBox.critical(self, "Error", f"Manual sell failed: {str(e)}")
            
    def update_gui(self):
        try:
            # Update performance stats
            self.win_rate_label.setText(f"{self.strategy.win_rate:.2f}%")
            self.daily_pnl_label.setText(f"{self.strategy.daily_pnl:.2f}")
            self.total_pnl_label.setText(f"{self.strategy.total_pnl:.2f}")
            self.trade_count_label.setText(f"{self.strategy.trade_count}")
            
            # Update position display
            if self.strategy.current_position:
                color = "green" if self.strategy.current_position == "BUY" else "red"
                self.signal_text.append(
                    f"<span style='color:{color}'>Current position: "
                    f"{self.strategy.current_position} at {self.strategy.entry_price:.5f}</span>"
                )
                
        except Exception as e:
            logger.error(f"Error updating GUI: {str(e)}")
            
    def train_model(self):
        try:
            reply = QMessageBox.question(
                self,
                "Confirm Training",
                "This will train a new model using synthetic data. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.strategy._train_model()
                QMessageBox.information(self, "Success", "Model trained successfully")
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Model training failed: {str(e)}")
            
    def load_model(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                "models",
                "Model Files (*.pkl *.joblib)"
            )
            
            if path:
                self.strategy._load_model(path)
                QMessageBox.information(self, "Success", f"Model loaded from {path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Model load failed: {str(e)}")
            
    def save_model(self):
        try:
            if not self.strategy.model:
                QMessageBox.warning(self, "Warning", "No model to save")
                return
                
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model",
                f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                "Model Files (*.joblib)"
            )
            
            if path:
                joblib.dump(self.strategy.model, path)
                QMessageBox.information(self, "Success", f"Model saved to {path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Model save failed: {str(e)}")
            
    def explain_prediction(self):
        try:
            if not self.current_data or 'M5' not in self.current_data:
                QMessageBox.warning(self, "Error", "No data available")
                return
                
            df = self.current_data['M5']
            features = self.strategy.prepare_features(df)
            
            if not hasattr(self.strategy.model.named_steps['lgbm'], 'feature_importances_'):
                QMessageBox.warning(self, "Error", "Model doesn't support feature importance")
                return
                
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.strategy.model.named_steps['lgbm'])
            shap_values = explainer.shap_values(features)
            
            # Plot SHAP values
            plt.clf()
            shap.summary_plot(shap_values, features, feature_names=self.strategy.feature_names)
            plt.tight_layout()
            
            # Create new window for SHAP plot
            plot_window = QMainWindow()
            plot_window.setWindowTitle("Prediction Explanation")
            plot_window.resize(800, 600)
            
            canvas = FigureCanvas(plt.gcf())
            plot_window.setCentralWidget(canvas)
            plot_window.show()
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            QMessageBox.critical(self, "Error", f"Prediction explanation failed: {str(e)}")
            
    def debug_indicator_calculation(self):
        try:
            if not self.current_data or 'M5' not in self.current_data:
                QMessageBox.warning(self, "Error", "No data available")
                return
                
            df = self.current_data['M5']
            success = self.strategy.calculate_indicators(df)
            
            if success:
                indicators = {}
                for name, values in self.strategy.indicators.items():
                    try:
                        indicators[name] = values.iloc[-1] if not values.empty else None
                    except Exception as e:
                        logger.error(f"Error getting indicator {name}: {str(e)}")
                        indicators[name] = None
                
                debug_text = "Indicator Calculation Debug:\n"
                debug_text += f"Data points: {len(df)}\n"
                debug_text += "Last values:\n"
                
                for name, value in indicators.items():
                    debug_text += f"{name}: {value}\n"
                    
                QMessageBox.information(
                    self,
                    "Debug Info",
                    debug_text
                )
            else:
                QMessageBox.warning(
                    self,
                    "Debug Info",
                    "Indicator calculation failed"
                )
                
        except Exception as e:
            logger.error(f"Error in debug: {str(e)}")
            QMessageBox.critical(self, "Error", f"Debug failed: {str(e)}")
            
    def save_strategy_settings(self):
        try:
            self.strategy.min_confidence = self.min_confidence_spin.value()
            self.strategy.base_atr_multiplier = self.atr_multiplier_spin.value()
            self.strategy.auto_trading = self.auto_trading_check.isChecked()
            
            self.save_settings()
            QMessageBox.information(self, "Success", "Strategy settings saved")
            
        except Exception as e:
            logger.error(f"Error saving strategy settings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
            
    def handle_error(self, error_msg):
        self.signal_text.append(f"ERROR: {error_msg}")
        logger.error(f"Error received: {error_msg}")
        
    def closeEvent(self, event):
        self.stop_bot()
        self.save_settings()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = TradingBotGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()