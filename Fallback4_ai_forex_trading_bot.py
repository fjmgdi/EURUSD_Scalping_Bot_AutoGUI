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
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeClose, OpenTrades
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit, QDoubleSpinBox,
    QFormLayout, QTabWidget, QCheckBox, QComboBox, QMessageBox,
    QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog, QSpinBox,
    QStyleFactory
)
from PyQt5.QtGui import QColor, QIcon, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mplfinance.original_flavor import candlestick_ohlc
from logging.handlers import RotatingFileHandler
import warnings
import json
import traceback

# Configure warning settings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

# Configure advanced logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler(
        'trading_bot_debug.log', 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("backups/models", exist_ok=True)
os.makedirs("model_analysis", exist_ok=True)

class EnhancedDataFetcher(QThread):
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
        self.fetch_interval = 30
        self.last_fetch_time = 0
        self.retry_count = 0
        self.max_retries = 3
        
    def process_candles(self, candles):
        try:
            if not candles or 'candles' not in candles:
                return pd.DataFrame()
                
            data = []
            for candle in candles['candles']:
                if not candle['complete']:
                    continue
                    
                try:
                    candle_data = {
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    }
                    data.append(candle_data)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error processing candle: {str(e)}")
                    continue
                    
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            return df[-self.count:]
            
        except Exception as e:
            logger.error(f"Error processing candles: {str(e)}")
            return pd.DataFrame()
            
    def run(self):
        self.running = True
        logger.info(f"Data fetcher started for {self.symbol}")
        
        while self.running and self.retry_count < self.max_retries:
            try:
                current_time = time.time()
                if current_time - self.last_fetch_time < self.fetch_interval:
                    time.sleep(1)
                    continue
                    
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
                    if not df.empty:
                        data[granularity] = df
                        self.retry_count = 0
                
                if data:
                    self.data_updated.emit(data)
                    self.last_update = data
                    self.last_fetch_time = current_time
                    
                time.sleep(1)
                
            except Exception as e:
                self.retry_count += 1
                error_msg = f"Error in data fetcher (attempt {self.retry_count}/{self.max_retries}): {str(e)}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                time.sleep(10)
                
        if self.retry_count >= self.max_retries:
            logger.error("Max retries reached, stopping data fetcher")
            self.error_occurred.emit("Max retries reached, please check connection")
    
    def stop(self):
        self.running = False
        logger.info("Data fetcher stopped")

class EnhancedTradingStrategy:
    def __init__(self, symbol="EUR_USD", model_path=None):
        self.symbol = symbol
        self.model = None
        self.indicators = {}
        self.current_position = None
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        self.daily_pnl = 0
        self.total_pnl = 0
        self.win_rate = 0
        self.trade_count = 0
        self.win_count = 0
        self.last_signal = None
        self.last_trade_time = None
        self.min_confidence = 0.6
        self.base_atr_multiplier = 1.8
        self.auto_trading = True
        self.max_daily_loss_pct = 2.0
        self.max_position_size_pct = 2.0
        self.trading_active = True
        self.max_hold_time = timedelta(minutes=30)
        
        self.feature_names = [
            'RSI', 'MACD', 'Stoch_K', 'Stoch_D', 'ADX', 'ATR', 'VWAP',
            'BB_Upper_Dev', 'BB_Lower_Dev', 'EMA_Diff', 'Price_EMA50_Diff',
            'Price_EMA200_Diff', 'BB_Squeeze', 'Volume_Spike', 'RSI_Change',
            'MACD_Change', 'Volume_Change', '5Period_Return', 
            'Resistance_Level', 'Support_Level', 'Volume_ZScore',
            'Volume_Momentum', 'Time_of_Day'
        ]
        
        self.features = []
        self.backtesting = False
        self.settings = QSettings("AI_Trading_Bot", "Forex_Trading_Bot")
        
        if model_path:
            self._load_model(model_path)
        else:
            self._train_model()
            
    def validate_data(self, df):
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
            
        if df.isnull().values.any() or len(df) < 200:
            return False
            
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False
            
        return True
            
    def _train_model(self):
        try:
            np.random.seed(42)
            n_samples = 50000
            X = np.zeros((n_samples, len(self.feature_names)))
            
            regimes = [
                {'volatility': 0.02, 'trend_strength': 0.8, 'duration': 0.2},
                {'volatility': 0.05, 'trend_strength': 0.3, 'duration': 0.3},
                {'volatility': 0.10, 'trend_strength': 0.1, 'duration': 0.5}
            ]
            
            regime_idx = 0
            regime_duration = int(n_samples * regimes[0]['duration'])
            
            for i in range(n_samples):
                if i > regime_duration:
                    regime_idx = (regime_idx + 1) % len(regimes)
                    regime_duration = i + int(n_samples * regimes[regime_idx]['duration'])
                
                current_regime = regimes[regime_idx]
                base_trend = current_regime['trend_strength'] * (np.sin(i/100) + 0.5*np.sin(i/30))
                noise_level = current_regime['volatility']
                
                X[i, 0] = 50 + 30*base_trend + noise_level*np.random.normal(0, 5)
                X[i, 1] = 0.5*base_trend + 0.3*np.sin(i/20) + noise_level*np.random.normal(0, 0.1)
                X[i, 2:4] = 50 + 40*base_trend + noise_level*np.random.normal(0, 5, 2)
                X[i, 4] = 20 + 30*abs(base_trend) + noise_level*abs(np.random.normal(0, 5))
                X[i, 5] = noise_level * (0.05 + 0.02*np.random.random())
                X[i, 6] = 1.0 + 0.2*base_trend + 0.1*np.random.normal(0, 0.05)
                X[i, 7:9] = base_trend*0.01 + noise_level*np.random.normal(0, 0.01, 2)
                X[i, 9] = base_trend*0.002 + noise_level*np.random.normal(0, 0.002)
                X[i, 10:12] = base_trend*0.01 + noise_level*np.random.normal(0, 0.01, 2)
                X[i, 12] = 1.0 + noise_level*np.random.normal(0, 0.5)
                X[i, 13] = 1.0 + abs(noise_level*np.random.normal(0, 1.0))
                X[i, 14] = X[i, 0] - X[max(0,i-5), 0]
                X[i, 15] = X[i, 1] - X[max(0,i-3), 1]
                X[i, 16] = X[i, 13] - X[max(0,i-10), 13]
                X[i, 17] = base_trend * 0.01 + noise_level * 0.01
                X[i, 18] = (1 - base_trend) * 0.02 + noise_level * 0.02
                X[i, 19] = (1 + base_trend) * 0.02 + noise_level * 0.02
                X[i, 20] = noise_level * np.random.normal(0, 2)
                X[i, 21] = X[i, 16] * 2
                X[i, 22] = (i % 1440) / 60
            
            # Define buy conditions
            condition1 = (X[:, 4] > 25) & (X[:, 0] > 50) & (X[:, 1] > 0) & (X[:, 17] > 0.002)
            condition2 = (X[:, 4] > 15) & (X[:, 5] < 0.001) & (X[:, 0] > 55) & (X[:, 20] > 1.0)
            condition3 = (X[:, 12] > 1.2) & (X[:, 0] < 35) & (np.random.random(n_samples) > 0.3)

            buy_signal = condition1 | condition2 | condition3
            y = np.where(buy_signal, 1, 0)
            
            X, y = self.augment_data(X, y)
            
            val_size = int(0.2 * len(X))
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'max_depth': 5,
                'min_data_in_leaf': 50,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'min_gain_to_split': 0.01,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.LGBMClassifier(**params)
            
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=True),
                    lgb.log_evaluation(50)
                ]
            )
            
            self.model = Pipeline([
                ('scaler', scaler),
                ('lgbm', model)
            ])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"models/model_{timestamp}.joblib"
            joblib.dump(self.model, model_path)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val_scaled)
            
            if isinstance(shap_values, list) and len(shap_values) == 2:
                plt.figure()
                shap.summary_plot(shap_values[1], X_val_scaled, feature_names=self.feature_names)
                plt.savefig(f"model_analysis/shap_summary_{timestamp}.png")
                plt.close()
            
            predictions = model.predict(X_val_scaled)
            probas = model.predict_proba(X_val_scaled)[:,1]
            report = classification_report(y_val, predictions, output_dict=True)
            
            logger.info("\nEnhanced Model Performance:")
            logger.info(f"Accuracy: {report['accuracy']:.4f}")
            logger.info(f"Precision: {report['1']['precision']:.4f}")
            logger.info(f"Recall: {report['1']['recall']:.4f}")
            logger.info(f"F1-Score: {report['1']['f1-score']:.4f}")
            logger.info(f"AUC: {roc_auc_score(y_val, probas):.4f}")
            
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nFeature Importance (Gain-based):\n" + importance.to_string())
            
        except Exception as e:
            logger.error(f"Error in enhanced model training: {str(e)}")
            raise
            
    def augment_data(self, X, y):
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X + noise
        
        reverse_pattern = np.array([-1]*14 + [1]*9)
        X_reversed = X * reverse_pattern
        y_reversed = 1 - y
        
        X_random = np.random.normal(0, 1, (int(X.shape[0]*0.2), X.shape[1]))
        y_random = np.random.randint(0, 2, int(X.shape[0]*0.2))
        
        return np.vstack([X, X_noisy, X_reversed, X_random]), np.concatenate([y, y, y_reversed, y_random])
            
    def _load_model(self, path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Loaded model doesn't support probability predictions")
                
            test_input = np.zeros((1, len(self.feature_names)))
            _ = self.model.predict_proba(test_input)
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            QMessageBox.critical(None, "Error", f"Failed to load model: {str(e)}\nTraining new model...")
            self._train_model()
            
    def calculate_indicators(self, df):
        try:
            if not self.validate_data(df):
                return False
                
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            self.indicators = {
                'close': close,
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'volume': volume
            }
            
            self.indicators['rsi'] = RSIIndicator(close=close, window=14).rsi().fillna(50)
            
            macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            self.indicators['macd'] = macd.macd().fillna(0)
            self.indicators['macd_signal'] = macd.macd_signal().fillna(0)
            self.indicators['macd_diff'] = macd.macd_diff().fillna(0)
            
            stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
            self.indicators['stoch_k'] = stoch.stoch().fillna(50)
            self.indicators['stoch_d'] = stoch.stoch_signal().fillna(50)
            
            self.indicators['adx'] = ADXIndicator(high=high, low=low, close=close, window=14).adx().fillna(20)
            
            self.indicators['ema_10'] = EMAIndicator(close=close, window=10).ema_indicator().fillna(close)
            self.indicators['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator().fillna(close)
            self.indicators['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator().fillna(close)
            
            atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
            self.indicators['atr'] = atr.bfill().ffill().fillna(0.001)
            
            bb = BollingerBands(close=close, window=20, window_dev=2)
            self.indicators['bollinger_upper'] = bb.bollinger_hband().fillna(close)
            self.indicators['bollinger_lower'] = bb.bollinger_lband().fillna(close)
            self.indicators['bollinger_mavg'] = bb.bollinger_mavg().fillna(close)
            
            self.indicators['vwap'] = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=volume, window=20
            ).volume_weighted_average_price().fillna(close)
            
            self.indicators['obv'] = OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume().fillna(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False
            
    def prepare_features(self, df):
        try:
            if not self.calculate_indicators(df):
                raise ValueError("Indicator calculation failed")
                
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
                ((self.indicators['bollinger_upper'].iloc[-1] - 
                  self.indicators['bollinger_lower'].iloc[-1]) / 
                 max(0.00001, self.indicators['atr'].iloc[-1])),
                (df['volume'].iloc[-1] / 
                 max(1, df['volume'].rolling(20).mean().iloc[-1])),
                self.indicators['rsi'].iloc[-1] - self.indicators['rsi'].iloc[-5],
                self.indicators['macd'].iloc[-1] - self.indicators['macd'].iloc[-3],
                (df['volume'].iloc[-1] - df['volume'].iloc[-10]) / 
                 max(1, df['volume'].iloc[-10]),
                (df['close'].iloc[-1] / df['close'].rolling(5).mean().iloc[-1] - 1),
                ((df['high'].iloc[-5:].max() - df['close'].iloc[-1]) / 
                 max(0.00001, self.indicators['atr'].iloc[-1])),
                ((df['close'].iloc[-1] - df['low'].iloc[-5:].min()) / 
                 max(0.00001, self.indicators['atr'].iloc[-1])),
                (df['volume'].iloc[-1] / 
                 max(0.00001, df['volume'].rolling(20).std().iloc[-1])),
                ((df['volume'].iloc[-1] - df['volume'].iloc[-5]) / 
                 max(0.00001, self.indicators['atr'].iloc[-1])),
                (df.index[-1].hour + df.index[-1].minute/60)
            ]
            
            return pd.DataFrame([features], columns=self.feature_names)
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame([np.zeros(len(self.feature_names))], columns=self.feature_names)
            
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
                return signal
                
            df = data_dict['M5']
            if len(df) < 200:
                return signal
                
            features = self.prepare_features(df)
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = max(proba)
            else:
                pred = self.model.predict(features)
                confidence = 0.7 if pred[0] == 1 else 0.3
                logger.warning("Using fallback confidence calculation")
            
            current_price = df['close'].iloc[-1]
            atr = self.indicators['atr'].iloc[-1]
            hour_of_day = df.index[-1].hour + df.index[-1].minute/60
            
            trading_hours = (7 <= hour_of_day <= 18)
            
            atr_multiplier = self.get_atr_multiplier()
            sl = current_price - (atr * atr_multiplier)
            tp = current_price + (atr * atr_multiplier * 1.5)
            
            signal.update({
                "price": current_price,
                "tp": tp,
                "sl": sl,
                "confidence": confidence,
                "time": df.index[-1].strftime('%H:%M:%S'),
                "indicators": {name: values.iloc[-1] for name, values in self.indicators.items()}
            })
            
            if self.last_trade_time and (datetime.now() - self.last_trade_time) < timedelta(minutes=5):
                signal['reason'] = "Trade cooldown active"
                return signal
            
            # Enhanced Buy Conditions
            buy_conditions = (
                self.trading_active and
                confidence >= (self.min_confidence * 0.8) and (
                    # Scenario 1: Classic oversold bounce
                    (self.indicators['rsi'].iloc[-1] < 45) or
                    # Scenario 2: Strong momentum with volume
                    (self.indicators['rsi'].iloc[-1] < 70 and 
                     features['Volume_ZScore'].iloc[0] > 3.0 and
                     self.indicators['macd_diff'].iloc[-1] > 0.0002) or
                    # Scenario 3: Breakout with confirmation
                    (current_price > self.indicators['bollinger_upper'].iloc[-1] and
                     features['Volume_ZScore'].iloc[0] > 2.0)
                )
            )
            
            # Enhanced Exit Conditions
            close_conditions = (
                self.current_position and (
                    (confidence < 0.4) or
                    (current_price >= tp) or
                    (current_price <= sl) or
                    # Dynamic RSI exit based on position
                    ((self.current_position == "BUY" and self.indicators['rsi'].iloc[-1] > 70) or
                     (self.current_position == "SELL" and self.indicators['rsi'].iloc[-1] < 30)) or
                    # Time-based exits
                    (hour_of_day > 18) or
                    (datetime.now() - self.entry_time > self.max_hold_time) or
                    # Trend reversal detection
                    (self.indicators['macd_diff'].iloc[-1] * 
                     self.indicators['macd_diff'].iloc[-2] < 0)  # MACD crossing
                )
            )
            
            if buy_conditions:
                signal.update({
                    "signal": "BUY",
                    "reason": f"Enhanced buy signal (Conf: {confidence:.2f}, RSI: {self.indicators['rsi'].iloc[-1]:.1f})"
                })
                self.last_trade_time = datetime.now()
            elif close_conditions:
                signal.update({
                    "signal": "CLOSE",
                    "reason": f"Exit conditions met (Conf: {confidence:.2f}, Price: {current_price:.5f})"
                })
                
            logger.info(f"Signal Conditions: RSI={self.indicators['rsi'].iloc[-1]:.1f}, "
                       f"MACD={self.indicators['macd_diff'].iloc[-1]:.5f}, "
                       f"Conf={confidence:.2f}, VolZ={features['Volume_ZScore'].iloc[0]:.1f}, "
                       f"Signal={signal['signal']}")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            signal.update({"signal": "ERROR", "reason": str(e)})
            return signal

    def get_atr_multiplier(self):
        try:
            if 'atr' not in self.indicators:
                return self.base_atr_multiplier
                
            atr = self.indicators['atr'].iloc[-1]
            rsi = self.indicators['rsi'].iloc[-1]
            hour_of_day = datetime.now().hour + datetime.now().minute/60
            
            multiplier = self.base_atr_multiplier
            
            if atr > 0.0015:
                multiplier *= 0.7
            elif atr < 0.0005:
                multiplier *= 1.3
                
            if rsi > 70 or rsi < 30:
                multiplier *= 0.8
                
            if 13 <= hour_of_day <= 15:
                multiplier *= 1.2
                
            return min(max(multiplier, 1.0), 3.0)
            
        except Exception as e:
            logger.error(f"Error calculating ATR multiplier: {str(e)}")
            return self.base_atr_multiplier

    def calculate_position_size(self, entry_price, stop_loss_price, account_balance):
        try:
            # Current risk calculation
            risk_amount = account_balance * (self.max_position_size_pct / 100)
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff <= 0:
                return 1000
                
            # Enhanced calculation with volatility adjustment
            atr = self.indicators['atr'].iloc[-1]
            volatility_factor = min(max(atr / 0.001, 0.5), 2.0)  # Adjust between 0.5-2.0 based on volatility
            adjusted_risk = risk_amount * volatility_factor
            
            units = adjusted_risk / price_diff
            
            # Dynamic cap based on account size
            max_units = min(1000, account_balance * 10)  # 10x leverage max
            return min(int(units), max_units)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1000

    def show_performance(self):
        perf_str = f"\nPerformance Summary:\n" \
                   f"Total Trades: {self.trade_count}\n" \
                   f"Win Rate: {self.win_rate:.1%}\n" \
                   f"Total PnL: ${self.total_pnl:.2f}\n" \
                   f"Daily PnL: ${self.daily_pnl:.2f}"
        logger.info(perf_str)
        return perf_str

class EnhancedTradingBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.setWindowTitle("Enhanced AI Forex Trading Bot")
            self.setWindowIcon(QIcon("icon.png"))
            self.resize(1600, 1000)
            
            load_dotenv()
            self.api_key = os.getenv("OANDA_API_KEY")
            self.account_id = os.getenv("OANDA_ACCOUNT_ID")
            
            if not self.api_key or not self.account_id:
                raise ValueError("API credentials not found in .env file")
                
            self.api = None
            self.strategy = EnhancedTradingStrategy()
            self.data_fetcher = None
            
            self.init_ui()
            self.init_signals()
            self.load_settings()
            
            # Test API connection immediately
            self.connect_to_api()
            
        except Exception as e:
            logger.error(f"GUI initialization failed: {str(e)}")
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize application:\n{str(e)}")
            raise
    
    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        self.left_panel = QWidget()
        self.left_panel.setMaximumWidth(450)
        self.left_layout = QVBoxLayout(self.left_panel)
        
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        self.init_connection_panel()
        self.init_trading_panel()
        self.init_strategy_panel()
        self.init_performance_panel()
        self.init_charts()
        self.init_log_panel()
        self.init_status_bar()
        
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)
    
    def init_performance_panel(self):
        group = QGroupBox("Performance Summary")
        layout = QVBoxLayout()
        
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setStyleSheet("font-family: monospace;")
        
        self.update_perf_btn = QPushButton("Update Performance")
        self.update_perf_btn.setStyleSheet("background-color: #2196F3; color: white;")
        
        layout.addWidget(self.performance_text)
        layout.addWidget(self.update_perf_btn)
        group.setLayout(layout)
        
        self.left_layout.addWidget(group)
    
    def init_connection_panel(self):
        group = QGroupBox("Connection Settings")
        layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("OANDA API Key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        if self.api_key:
            self.api_key_input.setText(self.api_key)
        
        self.account_id_input = QLineEdit()
        self.account_id_input.setPlaceholderText("Account ID")
        if self.account_id:
            self.account_id_input.setText(self.account_id)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        layout.addRow("API Key:", self.api_key_input)
        layout.addRow("Account ID:", self.account_id_input)
        layout.addRow(self.connect_btn)
        group.setLayout(layout)
        
        self.left_layout.addWidget(group)
    
    def init_trading_panel(self):
        group = QGroupBox("Trading Settings")
        layout = QFormLayout()
        
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"])
        
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 5.0)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setSuffix("%")
        
        self.max_loss_spin = QDoubleSpinBox()
        self.max_loss_spin.setRange(0.1, 10.0)
        self.max_loss_spin.setValue(2.0)
        self.max_loss_spin.setSuffix("%")
        
        self.pos_size_spin = QDoubleSpinBox()
        self.pos_size_spin.setRange(0.1, 10.0)
        self.pos_size_spin.setValue(2.0)
        self.pos_size_spin.setSuffix("%")
        
        self.auto_trade_check = QCheckBox("Auto Trading")
        self.auto_trade_check.setChecked(True)
        
        self.start_stop_btn = QPushButton("Start Trading")
        self.start_stop_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        layout.addRow("Symbol:", self.symbol_combo)
        layout.addRow("Risk per Trade:", self.risk_spin)
        layout.addRow("Max Daily Loss:", self.max_loss_spin)
        layout.addRow("Max Position Size:", self.pos_size_spin)
        layout.addRow(self.auto_trade_check)
        layout.addRow(self.start_stop_btn)
        group.setLayout(layout)
        
        manual_group = QGroupBox("Manual Trading")
        manual_layout = QVBoxLayout()
        
        self.buy_btn = QPushButton("Buy")
        self.buy_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        
        self.sell_btn = QPushButton("Sell")
        self.sell_btn.setStyleSheet("background-color: #F44336; color: white;")
        
        self.close_btn = QPushButton("Close Position")
        self.close_btn.setStyleSheet("background-color: #FF9800; color: white;")
        
        manual_layout.addWidget(self.buy_btn)
        manual_layout.addWidget(self.sell_btn)
        manual_layout.addWidget(self.close_btn)
        manual_group.setLayout(manual_layout)
        
        self.left_layout.addWidget(group)
        self.left_layout.addWidget(manual_group)
    
    def init_strategy_panel(self):
        group = QGroupBox("Strategy Settings")
        layout = QFormLayout()
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.6)
        
        self.atr_mult_spin = QDoubleSpinBox()
        self.atr_mult_spin.setRange(0.5, 5.0)
        self.atr_mult_spin.setValue(1.8)
        
        self.max_hold_spin = QSpinBox()
        self.max_hold_spin.setRange(1, 240)
        self.max_hold_spin.setValue(30)
        self.max_hold_spin.setSuffix(" min")
        
        self.load_model_btn = QPushButton("Load Model")
        self.train_model_btn = QPushButton("Train New Model")
        
        layout.addRow("Min Confidence:", self.confidence_spin)
        layout.addRow("ATR Multiplier:", self.atr_mult_spin)
        layout.addRow("Max Hold Time:", self.max_hold_spin)
        layout.addRow(self.load_model_btn)
        layout.addRow(self.train_model_btn)
        group.setLayout(layout)
        
        self.left_layout.addWidget(group)
        self.left_layout.addStretch()
    
    def init_charts(self):
        self.chart_tabs = QTabWidget()
        
        self.price_chart = QWidget()
        self.price_layout = QVBoxLayout(self.price_chart)
        
        self.price_fig = plt.figure(figsize=(12, 6))
        self.price_canvas = FigureCanvas(self.price_fig)
        self.price_layout.addWidget(self.price_canvas)
        
        self.indicator_chart = QWidget()
        self.indicator_layout = QVBoxLayout(self.indicator_chart)
        
        self.indicator_fig = plt.figure(figsize=(12, 4))
        self.indicator_canvas = FigureCanvas(self.indicator_fig)
        self.indicator_layout.addWidget(self.indicator_canvas)
        
        self.chart_tabs.addTab(self.price_chart, "Price Action")
        self.chart_tabs.addTab(self.indicator_chart, "Indicators")
        
        self.right_layout.addWidget(self.chart_tabs)
    
    def init_log_panel(self):
        group = QGroupBox("Trading Log")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace;")
        
        btn_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("Clear Log")
        self.save_log_btn = QPushButton("Save Log")
        
        btn_layout.addWidget(self.clear_log_btn)
        btn_layout.addWidget(self.save_log_btn)
        
        layout.addWidget(self.log_text)
        layout.addLayout(btn_layout)
        group.setLayout(layout)
        
        self.right_layout.addWidget(group)
    
    def init_status_bar(self):
        self.status_bar = self.statusBar()
        
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: red;")
        
        self.trading_status = QLabel("Not Trading")
        self.trading_status.setStyleSheet("color: orange;")
        
        self.position_status = QLabel("No Position")
        self.position_status.setStyleSheet("color: gray;")
        
        self.pnl_status = QLabel("PnL: $0.00")
        self.pnl_status.setStyleSheet("font-weight: bold;")
        
        self.status_bar.addWidget(self.connection_status)
        self.status_bar.addWidget(self.trading_status)
        self.status_bar.addWidget(self.position_status)
        self.status_bar.addPermanentWidget(self.pnl_status)
    
    def init_signals(self):
        self.connect_btn.clicked.connect(self.connect_to_api)
        self.buy_btn.clicked.connect(self.manual_buy)
        self.sell_btn.clicked.connect(self.manual_sell)
        self.close_btn.clicked.connect(self.close_position)
        self.load_model_btn.clicked.connect(self.load_model)
        self.train_model_btn.clicked.connect(self.train_model)
        self.clear_log_btn.clicked.connect(self.clear_logs)
        self.save_log_btn.clicked.connect(self.save_logs)
        self.start_stop_btn.clicked.connect(self.toggle_trading)
        self.update_perf_btn.clicked.connect(self.update_performance)
        
        self.symbol_combo.currentTextChanged.connect(self.change_symbol)
        self.risk_spin.valueChanged.connect(self.update_risk)
        self.max_loss_spin.valueChanged.connect(self.update_max_loss)
        self.pos_size_spin.valueChanged.connect(self.update_position_size)
        self.auto_trade_check.stateChanged.connect(self.toggle_auto_trading)
        self.confidence_spin.valueChanged.connect(self.update_confidence)
        self.atr_mult_spin.valueChanged.connect(self.update_atr_multiplier)
        self.max_hold_spin.valueChanged.connect(self.update_max_hold_time)
    
    def update_performance(self):
        perf_text = self.strategy.show_performance()
        self.performance_text.setPlainText(perf_text)
    
    def load_settings(self):
        settings = QSettings("EnhancedTradingBot", "ForexTrading")
        
        self.api_key_input.setText(settings.value("api_key", ""))
        self.account_id_input.setText(settings.value("account_id", ""))
        
        self.symbol_combo.setCurrentText(settings.value("symbol", "EUR_USD"))
        self.risk_spin.setValue(float(settings.value("risk", 1.0)))
        self.max_loss_spin.setValue(float(settings.value("max_loss", 2.0)))
        self.pos_size_spin.setValue(float(settings.value("position_size", 2.0)))
        self.auto_trade_check.setChecked(settings.value("auto_trading", "true") == "true")
        
        self.confidence_spin.setValue(float(settings.value("confidence", 0.6)))
        self.atr_mult_spin.setValue(float(settings.value("atr_multiplier", 1.8)))
        self.max_hold_spin.setValue(int(settings.value("max_hold_time", 30)))
        
        self.strategy.symbol = self.symbol_combo.currentText()
        self.strategy.min_confidence = self.confidence_spin.value()
        self.strategy.base_atr_multiplier = self.atr_mult_spin.value()
        self.strategy.auto_trading = self.auto_trade_check.isChecked()
        self.strategy.max_daily_loss_pct = self.max_loss_spin.value()
        self.strategy.max_position_size_pct = self.pos_size_spin.value()
        self.strategy.max_hold_time = timedelta(minutes=self.max_hold_spin.value())
    
    def save_settings(self):
        settings = QSettings("EnhancedTradingBot", "ForexTrading")
        
        settings.setValue("api_key", self.api_key_input.text())
        settings.setValue("account_id", self.account_id_input.text())
        
        settings.setValue("symbol", self.symbol_combo.currentText())
        settings.setValue("risk", self.risk_spin.value())
        settings.setValue("max_loss", self.max_loss_spin.value())
        settings.setValue("position_size", self.pos_size_spin.value())
        settings.setValue("auto_trading", self.auto_trade_check.isChecked())
        
        settings.setValue("confidence", self.confidence_spin.value())
        settings.setValue("atr_multiplier", self.atr_mult_spin.value())
        settings.setValue("max_hold_time", self.max_hold_spin.value())
    
    def connect_to_api(self):
        try:
            api_key = self.api_key_input.text().strip()
            account_id = self.account_id_input.text().strip()
            
            if not api_key or not account_id:
                raise ValueError("API key and Account ID are required")
                
            self.api = API(access_token=api_key, 
                          environment="practice",
                          request_params={'timeout': 10})
            
            endpoint = AccountDetails(accountID=account_id)
            response = self.api.request(endpoint)
            
            if 'account' not in response:
                raise ValueError("Failed to get account details")
                
            self.api_key = api_key
            self.account_id = account_id
            
            if hasattr(self, 'data_fetcher') and self.data_fetcher:
                self.data_fetcher.stop()
                self.data_fetcher.wait()
                
            self.data_fetcher = EnhancedDataFetcher(
                self.api, 
                self.symbol_combo.currentText(),
                granularities=["M5"],
                count=500
            )
            self.data_fetcher.data_updated.connect(self.update_data)
            self.data_fetcher.error_occurred.connect(self.handle_error)
            self.data_fetcher.start()
            
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet("color: green;")
            self.log_message("Successfully connected to OANDA API")
            
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            if hasattr(e, 'response'):
                try:
                    error_data = json.loads(e.response.text)
                    error_msg = error_data.get('errorMessage', str(e))
                except:
                    error_msg = e.response.text
                    
            self.connection_status.setText("Connection Failed")
            self.connection_status.setStyleSheet("color: red;")
            self.log_message(error_msg)
            QMessageBox.critical(self, "Connection Error", error_msg)
    
    def toggle_trading(self):
        if not self.api:
            QMessageBox.warning(self, "Error", "Please connect to API first")
            return
            
        self.strategy.trading_active = not self.strategy.trading_active
        
        if self.strategy.trading_active:
            self.start_stop_btn.setText("Stop Trading")
            self.start_stop_btn.setStyleSheet("background-color: #F44336; color: white;")
            self.trading_status.setText("Trading: ACTIVE")
            self.trading_status.setStyleSheet("color: green;")
            self.log_message("Trading activated")
        else:
            self.start_stop_btn.setText("Start Trading")
            self.start_stop_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.trading_status.setText("Trading: PAUSED")
            self.trading_status.setStyleSheet("color: orange;")
            self.log_message("Trading paused")
    
    def update_data(self, data_dict):
        try:
            if not data_dict or 'M5' not in data_dict:
                return
                
            df = data_dict['M5']
            if len(df) < 200:
                return
                
            signal = self.strategy.generate_signal(data_dict)
            self.update_charts(df, signal)
            
            if self.strategy.auto_trading and signal['signal'] != 'HOLD':
                self.execute_trade(signal['signal'], signal['price'])
                
        except Exception as e:
            self.log_message(f"Error updating data: {str(e)}")
    
    def update_charts(self, df, signal):
        try:
            self.price_fig.clear()
            self.indicator_fig.clear()
            
            df = df.iloc[-200:]
            dates = mdates.date2num(df.index.to_pydatetime())
            ohlc = list(zip(dates, df['open'], df['high'], df['low'], df['close']))
            
            ax1 = self.price_fig.add_subplot(111)
            ax1.grid(True)
            
            candlestick_ohlc(ax1, ohlc, width=0.0004, colorup='g', colordown='r')
            
            ax1.plot(dates, self.strategy.indicators['ema_50'].iloc[-200:], 'b-', label='EMA 50', alpha=0.7)
            ax1.plot(dates, self.strategy.indicators['ema_200'].iloc[-200:], 'm-', label='EMA 200', alpha=0.7)
            
            if 'bollinger_upper' in self.strategy.indicators:
                ax1.plot(dates, self.strategy.indicators['bollinger_upper'].iloc[-200:], 'c--', alpha=0.5)
                ax1.plot(dates, self.strategy.indicators['bollinger_lower'].iloc[-200:], 'c--', alpha=0.5)
            
            if signal['signal'] == 'BUY':
                ax1.axvline(x=dates[-1], color='g', linestyle='--', alpha=0.5)
                ax1.text(dates[-1], df['low'].iloc[-1], 'BUY', color='g')
            elif signal['signal'] == 'SELL':
                ax1.axvline(x=dates[-1], color='r', linestyle='--', alpha=0.5)
                ax1.text(dates[-1], df['high'].iloc[-1], 'SELL', color='r')
            elif signal['signal'] == 'CLOSE':
                ax1.axvline(x=dates[-1], color='orange', linestyle=':', alpha=0.5)
                ax1.text(dates[-1], df['close'].iloc[-1], 'CLOSE', color='orange')
                
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.price_fig.autofmt_xdate()
            ax1.legend()
            
            ax2 = self.indicator_fig.add_subplot(311)
            ax3 = self.indicator_fig.add_subplot(312)
            ax4 = self.indicator_fig.add_subplot(313)
            
            ax2.plot(dates, self.strategy.indicators['rsi'].iloc[-200:], 'b-')
            ax2.axhline(70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
            
            ax3.plot(dates, self.strategy.indicators['macd'].iloc[-200:], 'b-')
            ax3.plot(dates, self.strategy.indicators['macd_signal'].iloc[-200:], 'r-')
            ax3.bar(dates, self.strategy.indicators['macd_diff'].iloc[-200:], color='gray', alpha=0.3)
            ax3.set_ylabel('MACD')
            
            ax4.bar(dates, df['volume'].iloc[-200:], color='purple', alpha=0.5)
            ax4.set_ylabel('Volume')
            
            for ax in [ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.grid(True)
                
            self.indicator_fig.tight_layout()
            
            self.price_canvas.draw()
            self.indicator_canvas.draw()
            
        except Exception as e:
            self.log_message(f"Error updating charts: {str(e)}")
    
    def execute_trade(self, signal_type, price):
        try:
            if not self.api:
                self.log_message("Error: Not connected to API")
                return
                
            endpoint = AccountDetails(accountID=self.account_id)
            response = self.api.request(endpoint)
            
            if 'account' not in response:
                raise ValueError("Invalid account response")
                
            balance = float(response['account']['balance'])
            
            if self.strategy.daily_pnl < -(balance * (self.strategy.max_daily_loss_pct / 100)):
                self.log_message("Daily loss limit reached - stopping trading")
                self.strategy.trading_active = False
                self.toggle_trading()
                return
            
            # Calculate position size with hard cap at 1000 units
            units = min(
                self.strategy.calculate_position_size(
                    price,
                    price * 0.99 if signal_type == "BUY" else price * 1.01,
                    balance
                ),
                1000  # Absolute maximum
            )
            
            if units < 1000:
                self.log_message(f"Position size calculated at {units} units (below max)")
            
            if signal_type == "BUY":
                data = {
                    "order": {
                        "units": str(units),
                        "instrument": self.strategy.symbol,
                        "timeInForce": "FOK",
                        "type": "MARKET",
                        "positionFill": "DEFAULT"
                    }
                }
                
                endpoint = OrderCreate(accountID=self.account_id, data=data)
                response = self.api.request(endpoint)
                
                self.strategy.current_position = "BUY"
                self.strategy.entry_price = price
                self.strategy.entry_time = datetime.now()
                self.strategy.position_size = units
                
                self.position_status.setText("LONG")
                self.position_status.setStyleSheet("color: green;")
                self.log_message(f"BUY executed: {units} units @ {price:.5f}")
                
            elif signal_type == "SELL":
                data = {
                    "order": {
                        "units": str(-units),
                        "instrument": self.strategy.symbol,
                        "timeInForce": "FOK",
                        "type": "MARKET",
                        "positionFill": "DEFAULT"
                    }
                }
                
                endpoint = OrderCreate(accountID=self.account_id, data=data)
                response = self.api.request(endpoint)
                
                self.strategy.current_position = "SELL"
                self.strategy.entry_price = price
                self.strategy.entry_time = datetime.now()
                self.strategy.position_size = units
                
                self.position_status.setText("SHORT")
                self.position_status.setStyleSheet("color: red;")
                self.log_message(f"SELL executed: {units} units @ {price:.5f}")
                
            elif signal_type == "CLOSE":
                endpoint = OpenTrades(accountID=self.account_id)
                response = self.api.request(endpoint)
                
                if 'trades' not in response or len(response['trades']) == 0:
                    QMessageBox.warning(self, "Error", "No open trades found")
                    return
                    
                trade_id = response['trades'][0]['id']
                endpoint = TradeClose(accountID=self.account_id, tradeID=trade_id)
                response = self.api.request(endpoint)
                
                exit_price = float(response['orderFillTransaction']['price'])
                pnl = (exit_price - self.strategy.entry_price) * self.strategy.position_size
                if self.strategy.current_position == "SELL":
                    pnl *= -1
                    
                self.strategy.total_pnl += pnl
                self.strategy.daily_pnl += pnl
                self.strategy.trade_count += 1
                if pnl > 0:
                    self.strategy.win_count += 1
                self.strategy.win_rate = self.strategy.win_count / self.strategy.trade_count
                self.strategy.current_position = None
                self.strategy.entry_price = 0
                self.strategy.entry_time = None
                self.strategy.position_size = 0
                
                self.position_status.setText("FLAT")
                self.position_status.setStyleSheet("color: gray;")
                self.pnl_status.setText(f"PnL: ${self.strategy.total_pnl:.2f}")
                self.log_message(f"Position closed @ {exit_price:.5f}, PnL: ${pnl:.2f}")
                self.update_performance()
                
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                try:
                    error_data = json.loads(e.response.text)
                    error_msg = error_data.get('errorMessage', str(e))
                except:
                    error_msg = e.response.text
                    
            QMessageBox.critical(self, "Error", f"Trade execution failed: {error_msg}")
            self.log_message(f"Trade error: {error_msg}")
    
    def manual_buy(self):
        if not self.api_key_input.text() or not self.account_id_input.text():
            QMessageBox.warning(self, "Error", "Please connect to API first")
            return
            
        if not self.data_fetcher or not self.data_fetcher.last_update:
            QMessageBox.warning(self, "Error", "No market data available")
            return
            
        price = self.data_fetcher.last_update['M5']['close'].iloc[-1]
        self.execute_trade("BUY", price)
    
    def manual_sell(self):
        if not self.api_key_input.text() or not self.account_id_input.text():
            QMessageBox.warning(self, "Error", "Please connect to API first")
            return
            
        if not self.data_fetcher or not self.data_fetcher.last_update:
            QMessageBox.warning(self, "Error", "No market data available")
            return
            
        price = self.data_fetcher.last_update['M5']['close'].iloc[-1]
        self.execute_trade("SELL", price)
    
    def close_position(self):
        if not self.api_key_input.text() or not self.account_id_input.text():
            QMessageBox.warning(self, "Error", "Please connect to API first")
            return
            
        if not self.strategy.current_position:
            QMessageBox.warning(self, "Error", "No position to close")
            return
            
        self.execute_trade("CLOSE", self.strategy.entry_price)
    
    def load_model(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Model", "models", "Model Files (*.joblib)"
            )
            
            if path:
                self.strategy._load_model(path)
                self.log_message(f"Model loaded from {path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.log_message(f"Model load error: {str(e)}")
    
    def train_model(self):
        try:
            reply = QMessageBox.question(
                self, "Confirm", 
                "Training may take several hours. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_message("Starting model training...")
                self.strategy._train_model()
                self.log_message("Model training completed")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
    
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.moveCursor(QTextCursor.End)
    
    def clear_logs(self):
        self.log_text.clear()
    
    def save_logs(self):
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Logs", "logs", "Log Files (*.log)"
            )
            
            if path:
                with open(path, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Logs saved to {path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save logs: {str(e)}")
    
    def handle_error(self, message):
        self.log_message(f"ERROR: {message}")
    
    def change_symbol(self, symbol):
        self.strategy.symbol = symbol
        if self.data_fetcher:
            self.data_fetcher.symbol = symbol
        self.save_settings()
        self.log_message(f"Symbol changed to {symbol}")
    
    def update_risk(self, value):
        self.save_settings()
    
    def update_max_loss(self, value):
        self.strategy.max_daily_loss_pct = value
        self.save_settings()
    
    def update_position_size(self, value):
        self.strategy.max_position_size_pct = value
        self.save_settings()
    
    def toggle_auto_trading(self, state):
        self.strategy.auto_trading = state == Qt.Checked
        self.save_settings()
        status = "ON" if self.strategy.auto_trading else "OFF"
        self.trading_status.setText(f"Auto Trading: {status}")
        color = "green" if self.strategy.auto_trading else "orange"
        self.trading_status.setStyleSheet(f"color: {color};")
        self.log_message(f"Auto trading {status}")
    
    def update_confidence(self, value):
        self.strategy.min_confidence = value
        self.save_settings()
    
    def update_atr_multiplier(self, value):
        self.strategy.base_atr_multiplier = value
        self.save_settings()
    
    def update_max_hold_time(self, value):
        self.strategy.max_hold_time = timedelta(minutes=value)
        self.save_settings()
    
    def closeEvent(self, event):
        try:
            if hasattr(self, 'data_fetcher') and self.data_fetcher:
                self.data_fetcher.stop()
                self.data_fetcher.wait(2000)
                
            self.save_settings()
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            
        finally:
            event.accept()

def main():
    try:
        logger.info("Application starting...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create("Fusion"))
        
        # Set dark theme palette
        palette = app.palette()
        palette.setColor(palette.Window, QColor(53, 53, 53))
        palette.setColor(palette.WindowText, Qt.white)
        palette.setColor(palette.Base, QColor(25, 25, 25))
        palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(palette.ToolTipBase, Qt.white)
        palette.setColor(palette.ToolTipText, Qt.white)
        palette.setColor(palette.Text, Qt.white)
        palette.setColor(palette.Button, QColor(53, 53, 53))
        palette.setColor(palette.ButtonText, Qt.white)
        palette.setColor(palette.BrightText, Qt.red)
        palette.setColor(palette.Link, QColor(42, 130, 218))
        palette.setColor(palette.Highlight, QColor(42, 130, 218))
        palette.setColor(palette.HighlightedText, Qt.black)
        app.setPalette(palette)
        
        window = EnhancedTradingBotGUI()
        window.show()
        
        # Ensure proper application exit
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Fatal error during initialization: {str(e)}")
        traceback.print_exc()
        QMessageBox.critical(None, "Fatal Error", 
                            f"Application failed to start:\n{str(e)}")

if __name__ == "__main__":
    main()