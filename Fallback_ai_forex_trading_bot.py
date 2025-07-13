# ai_forex_trading_bot.py
import sys
import os
import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTextEdit, QDoubleSpinBox, QFormLayout,
    QTabWidget, QCheckBox, QComboBox, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt5.QtGui import QDoubleValidator, QFont, QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, QByteArray

import mplfinance as mpf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from xgboost import XGBClassifier
import joblib
from dotenv import load_dotenv
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.trades import TradeClose, OpenTrades
from oandapyV20.endpoints.positions import OpenPositions
import threading
import time
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(),
        logging.FileHandler('logs/retraining.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("backups/models", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

class DataFetcher(QThread):
    data_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api, symbol="EUR_USD", granularities=["M5", "H1", "D"], count=500):
        super().__init__()
        self.api = api
        self.symbol = symbol
        self.granularities = granularities
        self.count = count
        self.running = False
        self.last_update = {}
        self.update_interval = 60  # seconds
        self.min_data_threshold = int(count * 0.8)  # Require at least 80% of requested data
        
    def process_candles(self, candles):
        """More robust data processing with validation"""
        if not candles:
            raise ValueError("No candle data received")
            
        data = []
        for candle in candles:
            if not candle['complete']:
                continue
            try:
                data.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid candle data skipped: {str(e)}")
                continue
        
        if len(data) < self.min_data_threshold:
            raise ValueError(f"Insufficient valid data: {len(data)}/{self.count} candles")
            
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        return df
    
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
                        "price": "MBA"
                    }
                    
                    try:
                        candles = InstrumentsCandles(instrument=self.symbol, params=params)
                        self.api.request(candles)
                        df = self.process_candles(candles.response['candles'])
                        data[granularity] = df
                    except Exception as e:
                        logger.error(f"Error fetching {granularity} data: {str(e)}")
                        continue
                
                if data and 'M5' in data and not data['M5'].empty:
                    self.data_updated.emit(data)
                    self.last_update = data
                else:
                    raise ValueError("No valid M5 data received")
                
            except Exception as e:
                logger.error(f"Error in data fetcher: {str(e)}")
                self.error_occurred.emit(str(e))
            
            # Wait for next update
            for i in range(self.update_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def stop(self):
        self.running = False
        logger.info("Data fetcher stopped")
        
    def get_last_data(self):
        return self.last_update

class ModelRetrainer(QThread):
    retraining_complete = pyqtSignal(bool, str)
    
    def __init__(self, strategy, data_fetcher, retrain_interval=86400):
        super().__init__()
        self.strategy = strategy
        self.data_fetcher = data_fetcher
        self.retrain_interval = retrain_interval  # Default: 24 hours
        self.running = False
        self.last_retrain_time = None
        self.retry_count = 0
        self.max_retries = 3
        
    def run(self):
        self.running = True
        logger.info("Model retrainer started")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if we should retrain
                if (self.last_retrain_time is None or 
                    (current_time - self.last_retrain_time).total_seconds() >= self.retrain_interval):
                    
                    logger.info("Starting model retraining...")
                    success, message = self.retrain_model()
                    self.retraining_complete.emit(success, message)
                    self.last_retrain_time = current_time
                    self.retry_count = 0
                    
                    if success:
                        logger.info("Model retraining successful")
                    else:
                        logger.error(f"Model retraining failed: {message}")
                
                # Sleep for 1 minute before checking again
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in model retrainer: {str(e)}")
                time.sleep(60)
    
    def retrain_model(self):
        try:
            # Get historical data
            data = self.fetch_retraining_data()
            if not data or not self.validate_retraining_data(data['M5']):
                return False, "Invalid training data"
                
            # Prepare training data
            X, y = self.strategy._prepare_training_data(data['M5'])
            
            if X is None or y is None or X.shape[0] < 1000 or y.shape[0] < 1000:
                return False, "Insufficient training data"
                
            # Backup current model
            if os.path.exists(self.strategy.model_path):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"backups/models/model_backup_{timestamp}.pkl"
                shutil.copy(self.strategy.model_path, backup_path)
                logger.info(f"Model backed up to {backup_path}")
            
            # Train and evaluate new model
            old_score = self.evaluate_model(self.strategy.model, X, y)
            new_model = self.train_new_model(X, y)
            new_score = self.evaluate_model(new_model, X, y)
            
            improvement_pct = (new_score - old_score) / old_score * 100 if old_score != 0 else float('inf')
            
            # Only update if significant improvement
            if improvement_pct > 2.0:
                joblib.dump(new_model, self.strategy.model_path)
                self.strategy.model = new_model
                self.strategy._initialize_model()  # Reinitialize with new model
                return True, f"Model updated with {improvement_pct:.2f}% improvement"
            else:
                return False, f"Model kept (improvement {improvement_pct:.2f}% <= 2% threshold)"
                
        except Exception as e:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                return False, f"Retraining failed after {self.max_retries} attempts: {str(e)}"
            return False, f"Retraining failed (attempt {self.retry_count}): {str(e)}"

    def fetch_retraining_data(self):
        """Fetch extended historical data for retraining"""
        temp_fetcher = DataFetcher(
            api=self.data_fetcher.api,
            symbol=self.data_fetcher.symbol,
            granularities=["M5"],
            count=2000  # More data for retraining
        )
        try:
            temp_fetcher.run()
            time.sleep(10)  # Allow time for data collection
            data = temp_fetcher.get_last_data()
            temp_fetcher.stop()
            return data
        except Exception as e:
            logger.error(f"Error fetching retraining data: {str(e)}")
            return None

    def validate_retraining_data(self, df):
        """Validate data quality before retraining"""
        checks = {
            'min_candles': len(df) >= 1000,
            'no_nulls': not df.isnull().values.any(),
            'sufficient_unique': len(df['close'].unique()) > 100,
            'volatility': df['close'].pct_change().std() > 0.0002
        }
        
        if not all(checks.values()):
            logger.warning(f"Data validation failed: {checks}")
            return False
        return True

    def train_new_model(self, X, y):
        """Train new model with improved parameters"""
        model_params = {
            'n_estimators': 1000,
            'max_depth': 7,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.3,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'early_stopping_rounds': 50
        }
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(**model_params))
        ])
        
        model.fit(X_train, y_train)
        
        # Log performance
        y_pred = model.predict(X_val)
        logger.info(f"Model accuracy: {accuracy_score(y_val, y_pred):.2f}")
        logger.info(classification_report(y_val, y_pred))
        
        return model

    def evaluate_model(self, model, X, y):
        """Evaluate model performance"""
        if model is None:
            return 0
            
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )
        
        # Handle both pipeline and standalone models
        if hasattr(model, 'named_steps'):  # Pipeline
            return model.score(X_val, y_val)
        else:  # Standalone model
            return accuracy_score(y_val, model.predict(X_val))
    
    def stop(self):
        self.running = False
        logger.info("Model retrainer stopped")

class AITradingStrategy:
    def __init__(self, symbol="EUR_USD", model_path="models/ai_model.pkl"):
        self.symbol = symbol
        self.current_position = None
        self.position_size = 0
        self.entry_price = 0
        self.model = None
        self.indicators = {}
        self.trade_history = []
        self.daily_loss_limit = 0.02  # 2% of account balance
        self.max_drawdown = 0.05      # 5% max drawdown
        self.model_path = model_path
        self.initialized = False
        self.min_warmup_candles = 200  # Minimum candles before trading
        self.warmup_complete = False
        self._initialize_indicators()
        self._initialize_model()
        self.daily_pnl = 0
        self.total_pnl = 0
        self.win_rate = 0
        self.trade_count = 0
        self.win_count = 0
        self.last_signal = None
        self.min_confidence = 0.65  # Default minimum confidence threshold

    def _initialize_indicators(self):
        self.indicators = {
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'stoch_k': None,
            'stoch_d': None,
            'adx': None,
            'atr': None,
            'vwap': None,
            'bollinger_upper': None,
            'bollinger_lower': None,
            'ema_50': None,
            'ema_200': None,
            'obv': None
        }

    def _initialize_model(self):
        """Improved model initialization with better error handling"""
        try:
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                    if not hasattr(self.model, 'predict'):
                        raise ValueError("Loaded object is not a valid model")
                    logger.info(f"Model successfully loaded from {self.model_path}")
                    
                    # Test the model with dummy data
                    test_input = np.zeros((1, 11))
                    try:
                        prediction = self.model.predict(test_input)
                        logger.info(f"Model test prediction: {prediction}")
                    except Exception as e:
                        logger.error(f"Model test failed: {str(e)}")
                        raise
                    
                    # Check if it's a pipeline or standalone model
                    if hasattr(self.model, 'named_steps'):  # Pipeline
                        xgb_model = self.model.named_steps['xgb']
                        if hasattr(xgb_model, 'n_features_in_') and xgb_model.n_features_in_ != 11:
                            logger.warning("Model feature mismatch, retraining...")
                            self._train_model()
                    else:  # Standalone model
                        if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != 11:
                            logger.warning("Model feature mismatch, retraining...")
                            self._train_model()
                    
                    self.initialized = True
                    return
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
            
            logger.info("No valid model found, training new model...")
            if self._train_model():
                self.initialized = True
            
        except Exception as e:
            logger.error(f"Critical error during model initialization: {str(e)}")
            raise

    def _train_model(self, X=None, y=None):
        """Enhanced model training with progress feedback"""
        try:
            logger.info("Starting model training...")
            
            if X is None or y is None:
                X, y = self._prepare_training_data()
            
            if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("No valid training data available")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Model pipeline with preprocessing
            model_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.02,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'early_stopping_rounds': 20
            }
            
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBClassifier(**model_params))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            logger.info(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model trained and saved to {self.model_path}")
            
            # Save feature importance
            if hasattr(self.model.named_steps['xgb'], 'feature_importances_'):
                feature_names = [
                    'RSI', 'MACD', 'Stoch_K', 'Stoch_D', 'ADX', 'ATR',
                    'VWAP', 'BB_Upper', 'BB_Lower', 'EMA_Diff', 'OBV'
                ]
                importance_df = pd.DataFrame(
                    self.model.named_steps['xgb'].feature_importances_,
                    index=feature_names,
                    columns=['Importance']
                ).sort_values('Importance', ascending=False)
                importance_df.to_csv('feature_importance.csv')
                logger.info("Feature importance saved to feature_importance.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.model = None
            return False

    def _prepare_training_data(self, df=None):
        try:
            if df is None:
                # Load from saved data if available
                data_file = f"data/{self.symbol}_historical.csv"
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file, parse_dates=['time'], index_col='time')
                else:
                    # Generate random data for initial setup
                    np.random.seed(42)
                    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
                    prices = np.cumprod(1 + np.random.normal(0.001, 0.01, 1000))
                    df = pd.DataFrame({
                        'open': prices,
                        'high': prices * (1 + np.random.uniform(0, 0.005, 1000)),
                        'low': prices * (1 - np.random.uniform(0, 0.005, 1000)),
                        'close': prices,
                        'volume': np.random.randint(100, 1000, 1000)
                    }, index=dates)
            
            if not self.calculate_indicators(df):
                raise ValueError("Indicator calculation failed")
            
            features = []
            targets = []
            
            for i in range(50, len(df)-1):  # Start from 50 to ensure all indicators have values
                try:
                    feature_vector = [
                        self.indicators['rsi'].iloc[i],
                        self.indicators['macd'].iloc[i],
                        self.indicators['stoch_k'].iloc[i],
                        self.indicators['stoch_d'].iloc[i],
                        self.indicators['adx'].iloc[i],
                        self.indicators['atr'].iloc[i],
                        self.indicators['vwap'].iloc[i],
                        self.indicators['bollinger_upper'].iloc[i] - df['close'].iloc[i],
                        df['close'].iloc[i] - self.indicators['bollinger_lower'].iloc[i],
                        self.indicators['ema_50'].iloc[i] - self.indicators['ema_200'].iloc[i],
                        self.indicators['obv'].iloc[i]
                    ]
                    
                    # Target is 1 if next close is higher than current close
                    target = 1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0
                    
                    features.append(feature_vector)
                    targets.append(target)
                except (IndexError, KeyError) as e:
                    logger.warning(f"Skipping incomplete data at index {i}: {str(e)}")
                    continue
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None

    def calculate_indicators(self, df):
        try:
            if df.empty or len(df) < 200:  # Ensure enough data for indicators
                logger.warning(f"Not enough data to calculate indicators ({len(df)} < 200)")
                return False
                
            # Clean data
            close = df['close'].ffill().bfill()
            high = df['high'].ffill().bfill()
            low = df['low'].ffill().bfill()
            volume = df['volume'].ffill().bfill()
            
            # RSI (requires at least 14 periods)
            self.indicators['rsi'] = RSIIndicator(close=close, window=14).rsi().clip(0, 100)
            
            # MACD
            macd = MACD(close=close)
            self.indicators['macd'] = macd.macd()
            self.indicators['macd_signal'] = macd.macd_signal()
            
            # Stochastic (requires at least 14 periods)
            stoch = StochasticOscillator(
                high=high, low=low, close=close, 
                window=14, smooth_window=3
            )
            self.indicators['stoch_k'] = stoch.stoch()
            self.indicators['stoch_d'] = stoch.stoch_signal()
            
            # ADX (requires at least 14 periods)
            self.indicators['adx'] = ADXIndicator(
                high=high, low=low, close=close, window=14
            ).adx()
            
            # Moving Averages
            self.indicators['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator()
            self.indicators['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator()
            
            # ATR (requires at least 14 periods)
            self.indicators['atr'] = AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()
            
            # Bollinger Bands (requires at least 20 periods)
            bb = BollingerBands(close=close, window=20, window_dev=2)
            self.indicators['bollinger_upper'] = bb.bollinger_hband()
            self.indicators['bollinger_lower'] = bb.bollinger_lband()
            
            # VWAP (requires at least 14 periods)
            self.indicators['vwap'] = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=volume, window=14
            ).volume_weighted_average_price()
            
            # OBV
            self.indicators['obv'] = OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume()
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False

    def prepare_features(self, df):
        try:
            # Ensure all indicators have values
            for name, indicator in self.indicators.items():
                if indicator is None or len(indicator) == 0:
                    logger.error(f"Indicator {name} not properly calculated")
                    return None
            
            last_index = -1  # Get most recent values
            
            features = [
                self.indicators['rsi'].iloc[last_index],
                self.indicators['macd'].iloc[last_index],
                self.indicators['stoch_k'].iloc[last_index],
                self.indicators['stoch_d'].iloc[last_index],
                self.indicators['adx'].iloc[last_index],
                self.indicators['atr'].iloc[last_index],
                self.indicators['vwap'].iloc[last_index],
                self.indicators['bollinger_upper'].iloc[last_index] - df['close'].iloc[last_index],
                df['close'].iloc[last_index] - self.indicators['bollinger_lower'].iloc[last_index],
                self.indicators['ema_50'].iloc[last_index] - self.indicators['ema_200'].iloc[last_index],
                self.indicators['obv'].iloc[last_index]
            ]
            
            # Convert to numpy array and reshape
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Handle NaN/inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.error("Features contain NaN or inf values")
                return None
                
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def generate_signal(self, data_dict, risk_percent=1.0, tp_multiplier=1.5, sl_multiplier=1.0):
        signal = {
            "signal": "HOLD",
            "reason": "Initializing",
            "price": 0.0,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 0.0,
            "time": datetime.now().strftime('%H:%M:%S'),
            "indicators": {},
            "h1_trend": False,
            "daily_trend": False,
            "risk_level": risk_percent
        }

        # Check warmup status
        if not self.warmup_complete:
            if 'M5' in data_dict and len(data_dict['M5']) >= self.min_warmup_candles:
                self.warmup_complete = True
                logger.info(f"Warmup complete with {len(data_dict['M5'])} candles")
            else:
                current = len(data_dict['M5']) if 'M5' in data_dict else 0
                signal.update({
                    "signal": "HOLD",
                    "reason": f"Warming up ({current}/{self.min_warmup_candles} candles)",
                    "confidence": 0.0
                })
                return signal

        if not self.initialized or not self.model:
            signal.update({
                "signal": "HOLD",
                "reason": "Model not ready",
                "confidence": 0.0
            })
            return signal

        try:
            df = data_dict['M5']
            if df.empty or len(df) < 100:
                signal.update({
                    "signal": "HOLD",
                    "reason": "Insufficient data",
                    "confidence": 0.0
                })
                return signal

            # Calculate indicators first
            if not self.calculate_indicators(df):
                signal.update({
                    "signal": "HOLD",
                    "reason": "Indicator calculation failed",
                    "confidence": 0.0
                })
                return signal

            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]
            
            # Prepare features for prediction
            features = self.prepare_features(df)
            if features is None or np.all(features == 0):
                signal.update({
                    "signal": "HOLD",
                    "reason": "Feature preparation failed",
                    "confidence": 0.0
                })
                return signal

            # Get prediction probabilities
            try:
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features)[0]
                    confidence = max(proba)
                    prediction = np.argmax(proba)
                else:
                    # Fallback for models without predict_proba
                    prediction = self.model.predict(features)[0]
                    confidence = 0.7  # Default confidence if no probabilities available
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                signal.update({
                    "signal": "HOLD",
                    "reason": "Prediction failed",
                    "confidence": 0.0
                })
                return signal

            # Calculate TP/SL based on ATR
            atr = self.indicators['atr'].iloc[-1] if self.indicators['atr'] is not None else 0
            tp = current_price + (atr * tp_multiplier)
            sl = current_price - (atr * sl_multiplier)

            # Update signal with basic info
            signal.update({
                "price": current_price,
                "tp": tp,
                "sl": sl,
                "confidence": confidence,
                "time": current_time.strftime('%H:%M:%S'),
                "indicators": {
                    k: v.iloc[-1] if v is not None else 0 
                    for k, v in self.indicators.items()
                }
            })

            # Get additional indicator values for signal logic
            rsi_value = self.indicators['rsi'].iloc[-1] if self.indicators['rsi'] is not None else 50
            obv_trend = False
            if self.indicators['obv'] is not None and len(self.indicators['obv']) >= 5:
                obv_trend = self.indicators['obv'].iloc[-1] > self.indicators['obv'].rolling(5).mean().iloc[-1]
            
            macd_hist = self.indicators['macd'].iloc[-1] if self.indicators['macd'] is not None else 0
            stoch_k = self.indicators['stoch_k'].iloc[-1] if self.indicators['stoch_k'] is not None else 50
            stoch_d = self.indicators['stoch_d'].iloc[-1] if self.indicators['stoch_d'] is not None else 50
            
            # Get higher timeframe trends
            df_h1 = data_dict.get('H1', pd.DataFrame())
            df_d = data_dict.get('D', pd.DataFrame())
            
            h1_trend = False
            daily_trend = False
            
            if len(df_h1) >= 50:
                h1_trend = df_h1['close'].iloc[-1] > df_h1['close'].rolling(50).mean().iloc[-1]
            
            if len(df_d) >= 200:
                daily_trend = df_d['close'].iloc[-1] > df_d['close'].rolling(200).mean().iloc[-1]
            
            signal.update({
                "h1_trend": h1_trend,
                "daily_trend": daily_trend
            })

            # Signal generation logic with multiple confirmation
            if prediction == 1 and self.current_position != "BUY":
                # Check multiple confirmations
                confirmations = 0
                
                # RSI confirmation
                if 30 < rsi_value < 70:
                    confirmations += 1
                
                # Stochastic confirmation
                if stoch_k > stoch_d and stoch_k < 80:
                    confirmations += 1
                
                # MACD confirmation
                if macd_hist > 0:
                    confirmations += 1
                
                # OBV confirmation
                if obv_trend:
                    confirmations += 1
                
                # Trend confirmation
                if daily_trend and h1_trend:
                    confirmations += 2  # Higher weight for trend
                
                # Require at least 3 confirmations to trade
                if confirmations >= 3:
                    signal.update({
                        "signal": "BUY",
                        "reason": f"AI Buy Signal ({confirmations} confirmations)",
                        "confidence": min(confidence * (1 + confirmations * 0.1), 0.99)
                    })
                else:
                    signal.update({
                        "signal": "HOLD",
                        "reason": f"Insufficient confirmations ({confirmations}/3)",
                        "confidence": confidence * (confirmations / 3)
                    })
            
            elif prediction == 0 and self.current_position == "BUY":
                # Check exit conditions
                exit_conditions = 0
                
                # RSI exit
                if rsi_value > 70:
                    exit_conditions += 1
                
                # Stochastic exit
                if stoch_k < stoch_d and stoch_k > 20:
                    exit_conditions += 1
                
                # MACD exit
                if macd_hist < 0:
                    exit_conditions += 1
                
                # OBV exit
                if not obv_trend:
                    exit_conditions += 1
                
                # Trend exit
                if not daily_trend or not h1_trend:
                    exit_conditions += 1
                
                # Exit if at least 2 conditions met
                if exit_conditions >= 2:
                    signal.update({
                        "signal": "CLOSE",
                        "reason": f"AI Exit Signal ({exit_conditions} conditions)"
                    })
            
            self.last_signal = signal
            logger.debug(f"Generated signal: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}", exc_info=True)
            signal.update({
                "signal": "ERROR",
                "reason": str(e),
                "confidence": 0.0
            })
            return signal

    def update_position(self, position_type, price, size):
        self.current_position = position_type
        self.entry_price = price
        self.position_size = size
        logger.info(f"Position updated: {position_type} at {price} with size {size}")

    def close_position(self, exit_price):
        if self.current_position is None:
            return 0
            
        pnl = 0
        if self.current_position == "BUY":
            pnl = (exit_price - self.entry_price) * self.position_size
        elif self.current_position == "SELL":
            pnl = (self.entry_price - exit_price) * self.position_size
        
        # Update trade history and statistics
        self.trade_history.append({
            'entry_time': datetime.now() - timedelta(minutes=5),  # Approximate
            'exit_time': datetime.now(),
            'position': self.current_position,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'size': self.position_size,
            'pnl': pnl,
            'win': pnl > 0
        })
        
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        
        if self.trade_count > 0:
            self.win_rate = self.win_count / self.trade_count
        
        # Reset position
        self.current_position = None
        self.entry_price = 0
        self.position_size = 0
        
        logger.info(f"Position closed at {exit_price}, PnL: {pnl:.2f}")
        return pnl

    def get_performance_metrics(self):
        return {
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'trade_count': self.trade_count,
            'win_count': self.win_count
        }

    def reset_daily_metrics(self):
        self.daily_pnl = 0
        logger.info("Daily metrics reset")

class TradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Forex Trading Bot")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize settings
        self.settings = QSettings("AI_Trading_Bot", "Forex_Trader")
        
        # Initialize variables
        self.api = None
        self.strategy = None
        self.data_fetcher = None
        self.retrainer = None
        self.current_symbol = "EUR_USD"
        self.model_path = "models/ai_model.pkl"
        self.account_balance = 0
        self.equity = 0
        self.margin_available = 0
        self.open_positions = []
        self.open_trades = []
        self.last_data = {}
        self.chart_figure = None
        self.chart_canvas = None
        self.indicators_figure = None
        self.indicators_canvas = None
        self.dark_mode = False
        
        # Load environment variables
        load_dotenv()
        
        # Initialize UI
        self.init_ui()
        
        # Load saved settings
        self.load_settings()
        
        # Start with trading stopped
        self.update_button_states(connected=False)
        
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Connection group
        connection_group = QGroupBox("Connection")
        connection_layout = QFormLayout(connection_group)
        
        self.account_label = QLabel("Not connected")
        self.balance_label = QLabel("Balance: -")
        self.equity_label = QLabel("Equity: -")
        self.margin_label = QLabel("Margin: -")
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter OANDA API key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        
        self.account_id_input = QLineEdit()
        self.account_id_input.setPlaceholderText("Enter Account ID")
        
        self.practice_check = QCheckBox("Practice Account")
        self.practice_check.setChecked(True)
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.init_oanda_api)
        
        connection_layout.addRow("API Key:", self.api_key_input)
        connection_layout.addRow("Account ID:", self.account_id_input)
        connection_layout.addRow(self.practice_check)
        connection_layout.addRow(self.connect_button)
        connection_layout.addRow(self.account_label)
        connection_layout.addRow(self.balance_label)
        connection_layout.addRow(self.equity_label)
        connection_layout.addRow(self.margin_label)
        
        # Trading group
        trading_group = QGroupBox("Trading")
        trading_layout = QFormLayout(trading_group)
        
        self.pair_combo = QComboBox()
        self.pair_combo.addItems(["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"])
        
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 5.0)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setSuffix("%")
        
        self.tp_multiplier = QDoubleSpinBox()
        self.tp_multiplier.setRange(1.0, 3.0)
        self.tp_multiplier.setValue(1.5)
        
        self.sl_multiplier = QDoubleSpinBox()
        self.sl_multiplier.setRange(0.5, 2.0)
        self.sl_multiplier.setValue(1.0)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.5, 0.99)
        self.confidence_spin.setValue(0.65)
        self.confidence_spin.setSingleStep(0.01)
        
        self.start_button = QPushButton("Start Trading")
        self.start_button.clicked.connect(self.start_trading)
        
        self.stop_button = QPushButton("Stop Trading")
        self.stop_button.clicked.connect(self.stop_trading)
        
        self.auto_trade_button = QPushButton("Enable Auto Trading")
        self.auto_trade_button.setCheckable(True)
        self.auto_trade_button.clicked.connect(self.toggle_auto_trading)
        
        trading_layout.addRow("Currency Pair:", self.pair_combo)
        trading_layout.addRow("Risk per Trade:", self.risk_spin)
        trading_layout.addRow("TP Multiplier:", self.tp_multiplier)
        trading_layout.addRow("SL Multiplier:", self.sl_multiplier)
        trading_layout.addRow("Min Confidence:", self.confidence_spin)
        trading_layout.addRow(self.start_button)
        trading_layout.addRow(self.stop_button)
        trading_layout.addRow(self.auto_trade_button)
        
        # Signal group
        signal_group = QGroupBox("Current Signal")
        signal_layout = QVBoxLayout(signal_group)
        
        self.signal_label = QLabel("No signal")
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.signal_reason = QLabel("")
        self.signal_reason.setAlignment(Qt.AlignCenter)
        
        self.signal_price = QLabel("Price: -")
        self.signal_tp = QLabel("Take Profit: -")
        self.signal_sl = QLabel("Stop Loss: -")
        self.signal_confidence = QLabel("Confidence: -")
        
        signal_layout.addWidget(self.signal_label)
        signal_layout.addWidget(self.signal_reason)
        signal_layout.addWidget(self.signal_price)
        signal_layout.addWidget(self.signal_tp)
        signal_layout.addWidget(self.signal_sl)
        signal_layout.addWidget(self.signal_confidence)
        
        # Performance group
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.daily_pnl_label = QLabel("Today's PnL: -")
        self.total_pnl_label = QLabel("Total PnL: -")
        self.win_rate_label = QLabel("Win Rate: -")
        self.trade_count_label = QLabel("Trades: -")
        
        perf_layout.addRow(self.daily_pnl_label)
        perf_layout.addRow(self.total_pnl_label)
        perf_layout.addRow(self.win_rate_label)
        perf_layout.addRow(self.trade_count_label)
        
        # Theme toggle
        self.theme_button = QPushButton("Toggle Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        
        # Add groups to left layout
        left_layout.addWidget(connection_group)
        left_layout.addWidget(trading_group)
        left_layout.addWidget(signal_group)
        left_layout.addWidget(perf_group)
        left_layout.addWidget(self.theme_button)
        left_layout.addStretch()
        
        # Right panel (charts and logs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget for charts and logs
        tab_widget = QTabWidget()
        
        # Chart tab
        chart_tab = QWidget()
        chart_tab_layout = QVBoxLayout(chart_tab)
        
        self.chart_figure, self.chart_canvas = self.create_chart()
        chart_tab_layout.addWidget(self.chart_canvas)
        
        # Indicators tab
        indicators_tab = QWidget()
        indicators_tab_layout = QVBoxLayout(indicators_tab)
        
        self.indicators_figure, self.indicators_canvas = self.create_indicators_chart()
        indicators_tab_layout.addWidget(self.indicators_canvas)
        
        # Log tab
        log_tab = QWidget()
        log_tab_layout = QVBoxLayout(log_tab)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-family: monospace;")
        
        log_tab_layout.addWidget(self.log_display)
        
        # Positions tab
        positions_tab = QWidget()
        positions_layout = QVBoxLayout(positions_tab)
        
        self.positions_display = QTextEdit()
        self.positions_display.setReadOnly(True)
        
        positions_layout.addWidget(self.positions_display)
        
        # Add tabs
        tab_widget.addTab(chart_tab, "Price Chart")
        tab_widget.addTab(indicators_tab, "Indicators")
        tab_widget.addTab(log_tab, "Log")
        tab_widget.addTab(positions_tab, "Positions")
        
        right_layout.addWidget(tab_widget)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Timer for periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(5000)  # 5 seconds
        
        # Auto trade timer
        self.auto_trade_timer = QTimer()
        self.auto_trade_timer.timeout.connect(self.auto_trade_cycle)
        self.auto_trade_timer.setInterval(60000)  # 1 minute
        
        # Daily reset timer (at midnight)
        self.reset_timer = QTimer()
        self.reset_timer.timeout.connect(self.check_for_daily_reset)
        self.reset_timer.start(60000)  # 1 minute
        
        # Apply initial theme
        self.apply_theme()
        
    def create_chart(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(bottom=0.2)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return fig, canvas

    def create_indicators_chart(self):
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.5)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return fig, canvas

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2D2D2D;
                    color: #E0E0E0;
                }
                QGroupBox {
                    border: 1px solid #444;
                    border-radius: 5px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
                QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #3D3D3D;
                    color: #E0E0E0;
                    border: 1px solid #444;
                    padding: 3px;
                }
                QPushButton {
                    background-color: #505050;
                    color: #E0E0E0;
                    border: 1px solid #444;
                    padding: 5px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #606060;
                }
                QPushButton:pressed {
                    background-color: #404040;
                }
                QPushButton:checked {
                    background-color: #4CAF50;
                    color: white;
                }
                QTabWidget::pane {
                    border: 1px solid #444;
                    background: #2D2D2D;
                }
                QTabBar::tab {
                    background: #3D3D3D;
                    color: #E0E0E0;
                    padding: 5px;
                    border: 1px solid #444;
                    border-bottom: none;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                }
                QTabBar::tab:selected {
                    background: #505050;
                    border-bottom: 2px solid #4CAF50;
                }
                QLabel {
                    color: #E0E0E0;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #F0F0F0;
                    color: #000000;
                }
                QGroupBox {
                    border: 1px solid #AAA;
                    border-radius: 5px;
                    margin-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
                QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #AAA;
                    padding: 3px;
                }
                QPushButton {
                    background-color: #E0E0E0;
                    color: #000000;
                    border: 1px solid #AAA;
                    padding: 5px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #D0D0D0;
                }
                QPushButton:pressed {
                    background-color: #C0C0C0;
                }
                QPushButton:checked {
                    background-color: #4CAF50;
                    color: white;
                }
                QTabWidget::pane {
                    border: 1px solid #AAA;
                    background: #F0F0F0;
                }
                QTabBar::tab {
                    background: #E0E0E0;
                    color: #000000;
                    padding: 5px;
                    border: 1px solid #AAA;
                    border-bottom: none;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                }
                QTabBar::tab:selected {
                    background: #FFFFFF;
                    border-bottom: 2px solid #4CAF50;
                }
            """)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.log(f"Switched to {'dark' if self.dark_mode else 'light'} mode")

    def init_oanda_api(self):
        api_key = self.api_key_input.text().strip()
        account_id = self.account_id_input.text().strip()
        
        if not api_key or not account_id:
            self.log("Please enter both API key and Account ID", level="error")
            return False
            
        try:
            # Save credentials
            self.settings.setValue("api_key", api_key)
            self.settings.setValue("account_id", account_id)
            self.settings.setValue("practice_account", self.practice_check.isChecked())
            
            # Initialize API
            environment = "practice" if self.practice_check.isChecked() else "live"
            self.api = API(access_token=api_key, environment=environment)
            
            # Test connection
            account_details = AccountDetails(accountID=account_id)
            self.api.request(account_details)
            
            # Update account info
            self.account_balance = float(account_details.response['account']['balance'])
            self.equity = float(account_details.response['account']['balance'])  # Initial equity same as balance
            self.margin_available = float(account_details.response['account']['marginAvailable'])
            
            self.account_label.setText(f"Account: {account_id}")
            self.balance_label.setText(f"Balance: {self.account_balance:.2f}")
            self.equity_label.setText(f"Equity: {self.equity:.2f}")
            self.margin_label.setText(f"Margin: {self.margin_available:.2f}")
            
            self.log(f"Connected to OANDA {environment} account")
            self.update_button_states(connected=True)
            
            # Update open positions and trades
            self.update_account_info()
            self.update_open_positions()
            
            return True
            
        except Exception as e:
            self.log(f"Failed to connect to OANDA API: {str(e)}", level="error")
            self.api = None
            return False

    def start_trading(self):
        if not self.init_oanda_api():
            return
            
        self.current_symbol = self.pair_combo.currentText()
        
        try:
            # Initialize strategy with model path
            self.strategy = AITradingStrategy(
                symbol=self.current_symbol,
                model_path=self.model_path
            )
            
            # Verify model loaded successfully
            if not self.strategy.initialized:
                raise RuntimeError("Failed to initialize trading model")
                
            # Update risk parameters
            self.update_risk_parameters()
            
            # Initialize data fetcher
            self.data_fetcher = DataFetcher(
                api=self.api,
                symbol=self.current_symbol,
                granularities=["M5", "H1", "D"],
                count=500
            )
            
            self.data_fetcher.data_updated.connect(self.on_data_update)
            self.data_fetcher.error_occurred.connect(lambda msg: self.log(msg, level="error"))
            self.data_fetcher.start()
            
            # Initialize retrainer
            self.retrainer = ModelRetrainer(self.strategy, self.data_fetcher)
            self.retrainer.retraining_complete.connect(self.on_retraining_complete)
            self.retrainer.start()
            
            self.log(f"Started trading {self.current_symbol}")
            self.update_button_states(connected=True)
            
            # Initial updates
            self.update_chart()
            self.update_account_info()
            self.update_open_positions()
            self.update_model_info()
            
        except Exception as e:
            self.log(f"Failed to start trading: {str(e)}", level="error")
            self.stop_trading()

    def stop_trading(self):
        if self.data_fetcher:
            self.data_fetcher.stop()
            self.data_fetcher = None
            
        if self.retrainer:
            self.retrainer.stop()
            self.retrainer = None
            
        if self.auto_trade_button.isChecked():
            self.auto_trade_button.setChecked(False)
            self.auto_trade_timer.stop()
            
        self.strategy = None
        self.log("Trading stopped")
        self.update_button_states(connected=False)

    def on_retraining_complete(self, success, message):
        if success:
            self.log(f"Model retraining successful: {message}")
            self.update_model_info()
        else:
            self.log(f"Model retraining failed: {message}", level="warning")

    def update_button_states(self, connected):
        is_trading = connected and self.strategy is not None
        
        self.connect_button.setEnabled(not is_trading)
        self.start_button.setEnabled(connected and not is_trading)
        self.stop_button.setEnabled(is_trading)
        self.auto_trade_button.setEnabled(is_trading)
        self.pair_combo.setEnabled(not is_trading)
        
        if not connected:
            self.account_label.setText("Not connected")
            self.balance_label.setText("Balance: -")
            self.equity_label.setText("Equity: -")
            self.margin_label.setText("Margin: -")

    def update_risk_parameters(self):
        if self.strategy:
            self.strategy.daily_loss_limit = self.risk_spin.value() / 100.0
            self.strategy.min_confidence = self.confidence_spin.value()
            self.log(f"Updated risk parameters: Daily loss limit {self.strategy.daily_loss_limit:.2%}, Min confidence {self.strategy.min_confidence:.2%}")

    def on_data_update(self, data):
        self.last_data = data
        self.update_chart()
        
        if self.auto_trade_button.isChecked():
            self.auto_trade_cycle()

    def update_chart(self):
        if not self.last_data or 'M5' not in self.last_data:
            return
            
        df = self.last_data['M5']
        if len(df) < 20:
            return
            
        try:
            # Clear previous charts
            self.chart_figure.clear()
            self.indicators_figure.clear()
            
            # Initialize indicators dictionary
            indicators = {}
            if self.strategy and self.strategy.indicators:
                indicators = self.strategy.indicators
            
            # Price chart with indicators
            ax1 = self.chart_figure.add_subplot(111)
            
            # Plot candles
            mpf.plot(df, type='candle', style='charles', ax=ax1, volume=False)
            
            # Add indicators if they exist
            if indicators.get('ema_50') is not None:
                ax1.plot(df.index, indicators['ema_50'], label='EMA 50', color='blue', alpha=0.7)
            if indicators.get('ema_200') is not None:
                ax1.plot(df.index, indicators['ema_200'], label='EMA 200', color='red', alpha=0.7)
            if indicators.get('bollinger_upper') is not None and indicators.get('bollinger_lower') is not None:
                ax1.plot(df.index, indicators['bollinger_upper'], label='BB Upper', color='green', alpha=0.5, linestyle='--')
                ax1.plot(df.index, indicators['bollinger_lower'], label='BB Lower', color='green', alpha=0.5, linestyle='--')
            
            ax1.legend()
            ax1.set_title(f"{self.current_symbol} Price and Indicators")
        
            # Indicators chart
            axes = self.indicators_figure.subplots(4, 1)
            
            # RSI
            if indicators.get('rsi') is not None:
                axes[0].plot(df.index, indicators['rsi'], label='RSI', color='purple')
                axes[0].axhline(70, color='red', linestyle='--', alpha=0.3)
                axes[0].axhline(30, color='green', linestyle='--', alpha=0.3)
                axes[0].set_ylim(0, 100)
                axes[0].legend()
                axes[0].set_title('RSI (14)')
            
            # MACD
            if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
                axes[1].plot(df.index, indicators['macd'], label='MACD', color='blue')
                axes[1].plot(df.index, indicators['macd_signal'], label='Signal', color='orange')
                axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
                axes[1].legend()
                axes[1].set_title('MACD')
            
            # Stochastic
            if indicators.get('stoch_k') is not None and indicators.get('stoch_d') is not None:
                axes[2].plot(df.index, indicators['stoch_k'], label='Stoch %K', color='blue')
                axes[2].plot(df.index, indicators['stoch_d'], label='Stoch %D', color='orange')
                axes[2].axhline(80, color='red', linestyle='--', alpha=0.3)
                axes[2].axhline(20, color='green', linestyle='--', alpha=0.3)
                axes[2].set_ylim(0, 100)
                axes[2].legend()
                axes[2].set_title('Stochastic Oscillator')
            
            # Volume
            axes[3].bar(df.index, df['volume'], color='blue', alpha=0.5)
            axes[3].set_title('Volume')
            
            # Redraw
            self.chart_canvas.draw()
            self.indicators_canvas.draw()
            
        except Exception as e:
            self.log(f"Error updating chart: {str(e)}", level="error")

    def auto_trade_cycle(self):
        if not self.strategy or not self.last_data:
            return
            
        signal = self.strategy.generate_signal(
            self.last_data,
            risk_percent=self.risk_spin.value(),
            tp_multiplier=self.tp_multiplier.value(),
            sl_multiplier=self.sl_multiplier.value()
        )
        
        self.display_signal(signal)
        
        if signal['confidence'] < self.confidence_spin.value():
            self.log(f"Signal confidence {signal['confidence']:.2%} below threshold {self.confidence_spin.value():.2%}", level="warning")
            return
            
        if signal['signal'] == "BUY":
            self.execute_trade(signal)
        elif signal['signal'] == "CLOSE":
            self.close_trade(signal)

    def execute_trade(self, signal):
        if not self.api:
            return
            
        account_id = self.account_id_input.text().strip()
        if not account_id:
            return
            
        try:
            # Calculate position size based on risk
            risk_amount = self.account_balance * (self.risk_spin.value() / 100.0)
            stop_loss_pips = abs(signal['price'] - signal['sl'])
            pip_value = 1.0  # Simplified - should be calculated based on pair and lot size
            units = risk_amount / (stop_loss_pips * pip_value)
            
            # Place order
            data = {
                "order": {
                    "instrument": self.current_symbol,
                    "units": str(int(units)),
                    "type": "MARKET",
                    "takeProfitOnFill": {
                        "price": str(round(signal['tp'], 5))
                    },
                    "stopLossOnFill": {
                        "price": str(round(signal['sl'], 5))
                    }
                }
            }
            
            order = OrderCreate(accountID=account_id, data=data)
            self.api.request(order)
            
            # Update position
            self.strategy.update_position("BUY", signal['price'], units)
            self.log(f"BUY order executed at {signal['price']:.5f}, units: {units:.0f}")
            
            # Update account info after trade
            self.update_account_info()
            self.update_open_positions()
            
        except Exception as e:
            self.log(f"Error executing trade: {str(e)}", level="error")

    def close_trade(self, signal):
        if not self.api or not self.strategy.current_position:
            return
            
        account_id = self.account_id_input.text().strip()
        if not account_id:
            return
            
        try:
            trades = OpenTrades(accountID=account_id)
            self.api.request(trades)
            
            for trade in trades.response['trades']:
                if trade['instrument'] == self.current_symbol:
                    close_trade = TradeClose(accountID=account_id, tradeID=trade['id'])
                    self.api.request(close_trade)
                    
                    # Update position
                    pnl = self.strategy.close_position(signal['price'])
                    self.log(f"Trade closed at {signal['price']:.5f}, PnL: {pnl:.2f}")
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Update account info
                    self.update_account_info()
                    self.update_open_positions()
                    break
                    
        except Exception as e:
            self.log(f"Error closing trade: {str(e)}", level="error")

    def update_account_info(self):
        if not self.api:
            return
            
        account_id = self.account_id_input.text().strip()
        if not account_id:
            return
            
        try:
            account_details = AccountDetails(accountID=account_id)
            self.api.request(account_details)
            
            self.account_balance = float(account_details.response['account']['balance'])
            self.equity = float(account_details.response['account']['balance'])  # Simplified
            self.margin_available = float(account_details.response['account']['marginAvailable'])
            
            self.balance_label.setText(f"Balance: {self.account_balance:.2f}")
            self.equity_label.setText(f"Equity: {self.equity:.2f}")
            self.margin_label.setText(f"Margin: {self.margin_available:.2f}")
            
        except Exception as e:
            self.log(f"Error updating account info: {str(e)}", level="error")

    def update_open_positions(self):
        if not self.api:
            return
            
        account_id = self.account_id_input.text().strip()
        if not account_id:
            return
            
        try:
            positions = OpenPositions(accountID=account_id)
            self.api.request(positions)
            
            self.open_positions = positions.response['positions'] if 'positions' in positions.response else []
            
            trades = OpenTrades(accountID=account_id)
            self.api.request(trades)
            
            self.open_trades = trades.response['trades'] if 'trades' in trades.response else []
            
            # Update positions display
            text = "<b>Open Positions:</b>\n"
            if self.open_positions:
                for pos in self.open_positions:
                    if pos['instrument'] == self.current_symbol:
                        text += f"Long: {pos['long']['units']} @ {pos['long']['averagePrice']}\n"
                        text += f"Short: {pos['short']['units']} @ {pos['short']['averagePrice']}\n"
            else:
                text += "No open positions\n"
            
            text += "\n<b>Open Trades:</b>\n"
            if self.open_trades:
                for trade in self.open_trades:
                    text += f"{trade['instrument']} {trade['currentUnits']} @ {trade['price']} (PL: {trade['unrealizedPL']})\n"
            else:
                text += "No open trades\n"
            
            self.positions_display.setHtml(text)
            
            # Update status bar
            self.statusBar().showMessage(f"Open positions: {len(self.open_positions)} | Open trades: {len(self.open_trades)}")
            
        except Exception as e:
            self.log(f"Error updating open positions: {str(e)}", level="error")

    def update_performance_metrics(self):
        if self.strategy:
            metrics = self.strategy.get_performance_metrics()
            self.daily_pnl_label.setText(f"Today's PnL: {metrics['daily_pnl']:.2f}")
            self.total_pnl_label.setText(f"Total PnL: {metrics['total_pnl']:.2f}")
            self.win_rate_label.setText(f"Win Rate: {metrics['win_rate']:.1%}")
            self.trade_count_label.setText(f"Trades: {metrics['trade_count']}")

    def update_model_info(self):
        if self.strategy and self.strategy.model:
            if hasattr(self.strategy.model, 'named_steps'):  # Pipeline
                msg = f"Model loaded: {os.path.basename(self.model_path)} (Pipeline)"
                if hasattr(self.strategy.model.named_steps['xgb'], 'feature_importances_'):
                    msg += " (Feature importance available)"
            else:  # Standalone model
                msg = f"Model loaded: {os.path.basename(self.model_path)} (Standalone)"
                if hasattr(self.strategy.model, 'feature_importances_'):
                    msg += " (Feature importance available)"
            self.log(msg)

    def periodic_update(self):
        if self.api:
            self.update_account_info()
            self.update_open_positions()
            
        if self.strategy:
            self.update_performance_metrics()

    def check_for_daily_reset(self):
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            if self.strategy:
                self.strategy.reset_daily_metrics()
            self.log("Daily metrics reset at midnight")

    def display_signal(self, signal):
        self.signal_label.setText(signal['signal'])
        self.signal_reason.setText(signal['reason'])
        self.signal_price.setText(f"Price: {signal['price']:.5f}")
        self.signal_tp.setText(f"Take Profit: {signal['tp']:.5f}")
        self.signal_sl.setText(f"Stop Loss: {signal['sl']:.5f}")
        self.signal_confidence.setText(f"Confidence: {signal['confidence']:.2%}")
        
        # Color coding
        if signal['signal'] == "BUY":
            self.signal_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        elif signal['signal'] == "CLOSE":
            self.signal_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        else:
            self.signal_label.setStyleSheet("color: black; font-size: 16px; font-weight: bold;")

    def toggle_auto_trading(self, checked):
        if checked:
            self.auto_trade_timer.start()
            self.auto_trade_button.setText("Auto Trading ON")
            self.log("Auto trading enabled")
        else:
            self.auto_trade_timer.stop()
            self.auto_trade_button.setText("Enable Auto Trading")
            self.log("Auto trading disabled")

    def log(self, message, level="info"):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if level == "error":
            logger.error(message)
            self.log_display.setTextColor(QColor(255, 0, 0))  # Red
        elif level == "warning":
            logger.warning(message)
            self.log_display.setTextColor(QColor(255, 165, 0))  # Orange
        else:
            logger.info(message)
            self.log_display.setTextColor(QColor(0, 0, 0))  # Black
        
        self.log_display.append(log_message)
        self.log_display.ensureCursorVisible()

    def load_settings(self):
        api_key = self.settings.value("api_key", "")
        account_id = self.settings.value("account_id", "")
        practice = self.settings.value("practice_account", True, type=bool)
        
        self.api_key_input.setText(api_key)
        self.account_id_input.setText(account_id)
        self.practice_check.setChecked(practice)
        
        # Window geometry
        geometry = self.settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        # Save settings
        self.settings.setValue("window_geometry", self.saveGeometry())
        
        # Stop trading
        self.stop_trading()
        
        # Close
        event.accept()

def main():
    load_dotenv()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(10)
    app.setFont(font)
    
    window = TradingGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()