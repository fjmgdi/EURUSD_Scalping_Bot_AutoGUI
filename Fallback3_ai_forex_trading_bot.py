import sys
import os
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
    QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog,
    QSpinBox
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings, QObject, pyqtSlot
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
import time
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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
    
    def __init__(self, api, symbol="EUR_USD", granularities=["M5"], count=300):
        super().__init__()
        self.api = api
        self.symbol = symbol
        self.granularities = granularities
        self.count = count
        self.running = False
        self.last_update = {}
        self.update_interval = 300  # 5 minutes between updates
        self.min_data_threshold = int(count * 0.8)
        
    def process_candles(self, candles):
        """Process candle data with robust validation"""
        logger.debug(f"Processing candles for {self.symbol}")
        
        if not candles or 'candles' not in candles:
            logger.error("No candle data in response")
            raise ValueError("No candle data received")
            
        if len(candles['candles']) < self.min_data_threshold:
            logger.error(f"Insufficient candles: {len(candles['candles'])} (minimum {self.min_data_threshold})")
            raise ValueError(f"Insufficient candle data (received {len(candles['candles'])}, need {self.min_data_threshold})")
            
        data = []
        for candle in candles['candles']:
            if not candle['complete']:
                continue
            try:
                # Validate and convert all fields
                candle_time = pd.to_datetime(candle['time'])
                if pd.isna(candle_time):
                    raise ValueError("Invalid time format")
                    
                candle_data = {
                    'time': candle_time,
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                }
                
                # Validate price values
                if any(v <= 0 for v in [candle_data['open'], candle_data['high'], 
                                        candle_data['low'], candle_data['close']]):
                    raise ValueError("Invalid price value")
                    
                if candle_data['high'] < candle_data['low']:
                    raise ValueError("High price lower than low price")
                    
                data.append(candle_data)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid candle data skipped: {str(e)}")
                continue
        
        if len(data) < self.min_data_threshold:
            raise ValueError(f"Insufficient valid data: {len(data)}/{self.count} candles")
            
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Additional validation
        if df.isnull().values.any():
            logger.warning("NaN values detected, attempting to fill")
            df = df.ffill().bfill()
            if df.isnull().values.any():
                raise ValueError("NaN values remain after filling")
        
        # Verify price integrity
        if not (df['high'] >= df['low']).all():
            raise ValueError("Price integrity check failed (high < low)")
        
        logger.debug(f"Successfully processed {len(df)} candles")
        return df
    
    def force_refresh(self):
        """Force immediate data refresh with enhanced error handling"""
        try:
            logger.debug("Force refreshing data...")
            data = {}
            for granularity in self.granularities:
                params = {
                    "granularity": granularity,
                    "count": self.count,
                    "price": "M"
                }
                
                logger.debug(f"Fetching {granularity} data for {self.symbol} with params: {params}")
                
                endpoint = InstrumentsCandles(instrument=self.symbol, params=params)
                response = self.api.request(endpoint)
                
                logger.debug(f"Raw API response for {granularity}: {response}")
                
                if 'errorMessage' in response:
                    error_msg = response['errorMessage']
                    logger.error(f"API error: {error_msg}")
                    if "authorization" in error_msg.lower():
                        self.error_occurred.emit("API Authorization Failed - Check Credentials")
                    return False
                
                if 'candles' not in response or len(response['candles']) == 0:
                    logger.error(f"No candle data received for {granularity}")
                    return False
                    
                df = self.process_candles(response)
                data[granularity] = df
            
            if data and 'M5' in data:
                self.data_updated.emit(data)
                return True
                
            logger.error("No valid M5 data after processing")
            return False
            
        except Exception as e:
            logger.error(f"Force refresh failed: {str(e)}", exc_info=True)
            self.error_occurred.emit(str(e))
            return False
    
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
                    
                    try:
                        endpoint = InstrumentsCandles(instrument=self.symbol, params=params)
                        response = self.api.request(endpoint)
                        
                        if 'errorMessage' in response:
                            if "authorization" in response['errorMessage'].lower():
                                self.error_occurred.emit("API Authorization Failed - Check Credentials")
                                self.running = False
                                break
                        
                        df = self.process_candles(response)
                        data[granularity] = df
                    except Exception as e:
                        logger.error(f"Error fetching {granularity} data: {str(e)}")
                        continue
                
                if not self.running:
                    break
                    
                if data and 'M5' in data and not data['M5'].empty:
                    self.data_updated.emit(data)
                    self.last_update = data
                else:
                    logger.warning("No valid M5 data to emit")
                    
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
        self.retrain_interval = retrain_interval
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
        """Retrain model with enhanced validation"""
        try:
            # Get historical data
            data = self.fetch_retraining_data()
            if not data or not self.validate_retraining_data(data['M5']):
                return False, "Invalid training data"
                
            # Prepare training data
            X, y = self.strategy._prepare_training_data(data['M5'])
            
            if X is None or y is None or X.shape[0] < 1000 or y.shape[0] < 1000:
                return False, "Insufficient training data"
                
            # Verify feature dimensions
            if X.shape[1] != 11:
                return False, f"Unexpected feature count: {X.shape[1]} (expected 11)"
                
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
            
            if improvement_pct > 2.0:
                joblib.dump(new_model, self.strategy.model_path)
                self.strategy.model = new_model
                self.strategy._initialize_model()
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
            count=2000
        )
        try:
            temp_fetcher.run()
            time.sleep(10)
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
            'volatility': df['close'].pct_change().std() > 0.0002,
            'price_range': (df['close'].max() - df['close'].min()) > 0.001
        }
        
        if not all(checks.values()):
            logger.warning(f"Data validation failed: {checks}")
            return False
        return True

    def train_new_model(self, X, y):
        """Train new model with improved parameters and validation"""
        # Split data with validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
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
            'early_stopping_rounds': 50
        }
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(**model_params))
        ])
        
        model.fit(X_train, y_train, xgb__eval_set=[(X_val, y_val)])
        
        # Validate model performance
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")
        logger.info(classification_report(y_val, y_pred))
        
        return model

    def evaluate_model(self, model, X, y):
        """Evaluate model performance with validation"""
        if model is None:
            return 0
            
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )
        
        if hasattr(model, 'named_steps'):
            return model.score(X_val, y_val)
        else:
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
        self.daily_loss_limit = 0.02
        self.max_drawdown = 0.05
        self.model_path = model_path.replace(".pkl", f"_{symbol}.pkl")
        self.initialized = False
        self.min_warmup_candles = 50
        self.warmup_complete = False
        self._initialize_indicators()
        self._initialize_model()
        self.daily_pnl = 0
        self.total_pnl = 0
        self.win_rate = 0
        self.trade_count = 0
        self.win_count = 0
        self.last_signal = None
        self.min_confidence = 0.50
        self.atr_multiplier = 2.0
        self.auto_trading = True
        self.data = None  # Store reference to current data
        self.last_trade_time = None
        self.min_trade_interval = 300  # 5 minutes between trades

    def set_auto_trading(self, enabled):
        """Set auto trading mode"""
        self.auto_trading = enabled
        logger.info(f"Auto trading set to: {enabled}")

    def get_system_status(self):
        """Return comprehensive system status"""
        return {
            'model_ready': self.initialized,
            'data_ready': self.warmup_complete,
            'last_signal': self.last_signal,
            'position': self.current_position,
            'data_points': len(self.data) if self.data is not None else 0,
            'auto_trading': self.auto_trading
        }

    def _initialize_indicators(self):
        self.indicators = {
            'rsi': None, 'macd': None, 'macd_signal': None,
            'stoch_k': None, 'stoch_d': None, 'adx': None,
            'atr': None, 'vwap': None, 'bollinger_upper': None,
            'bollinger_lower': None, 'ema_50': None, 'ema_200': None,
            'obv': None
        }

    def _initialize_model(self):
        """Initialize model with robust validation"""
        logger.info(f"Attempting to initialize model from {self.model_path}")
        
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}")
                logger.info("Generating synthetic data for initial training...")
                
                # Generate synthetic data if no historical data exists
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
                prices = np.cumprod(1 + np.random.normal(0.0001, 0.001, 1000))
                df = pd.DataFrame({
                    'time': dates,
                    'open': prices,
                    'high': prices * (1 + np.random.uniform(0, 0.001, 1000)),
                    'low': prices * (1 - np.random.uniform(0, 0.001, 1000)),
                    'close': prices,
                    'volume': np.random.randint(100, 1000, 1000)
                })
                
                # Save the synthetic data
                os.makedirs(os.path.dirname(f"data/{self.symbol}_historical.csv"), exist_ok=True)
                df.to_csv(f"data/{self.symbol}_historical.csv", index=False)
                logger.info(f"Generated synthetic data saved to data/{self.symbol}_historical.csv")
                
                # Train initial model
                return self._train_model()
            
            logger.info(f"Model file exists, size: {os.path.getsize(self.model_path)} bytes")
            
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                
                # Verify the loaded model is valid
                if not hasattr(self.model, 'predict'):
                    logger.warning("Invalid model format, retraining...")
                    return self._train_model()
                
                # Test the model with dummy data
                dummy_data = np.zeros((1, 11))  # 11 features
                try:
                    prediction = self.model.predict(dummy_data)
                    proba = self.model.predict_proba(dummy_data) if hasattr(self.model, 'predict_proba') else None
                    
                    if prediction.shape != (1,) or (proba is not None and proba.shape != (1, 2)):
                        raise ValueError("Model prediction shape mismatch")
                        
                    logger.info("Model loaded and validated successfully")
                    self.initialized = True
                    return True
                except Exception as e:
                    logger.error(f"Model validation failed: {str(e)}")
                    return self._train_model()
                    
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.info("Attempting to train new model...")
                return self._train_model()
            
        except Exception as e:
            logger.error(f"Critical error during model initialization: {str(e)}")
            self.initialized = False
            return False

    def _train_model(self, X=None, y=None):
        """Train model with progress feedback and validation"""
        try:
            logger.info("Starting model training...")
            
            if X is None or y is None:
                X, y = self._prepare_training_data()
            
            if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("No valid training data available")
            
            # Verify feature dimensions
            if X.shape[1] != 11:
                raise ValueError(f"Unexpected feature count: {X.shape[1]} (expected 11)")
            
            # Split data with validation set
            X_train, X_val, y_train, y_val = train_test_split(
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
                'early_stopping_rounds': 20
            }
            
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBClassifier(**model_params))
            ])
            
            # Fit with validation set
            self.model.fit(X_train, y_train, xgb__eval_set=[(X_val, y_val)])
            
            # Evaluate model
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            logger.info(classification_report(y_val, y_pred))
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
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
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.model = None
            self.initialized = False
            return False

    def _prepare_training_data(self, df=None):
        """Prepare training data with validation"""
        try:
            if df is None:
                data_file = f"data/{self.symbol}_historical.csv"
                if os.path.exists(data_file):
                    df = pd.read_csv(data_file)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                    if len(df) < 200:
                        raise ValueError(f"Insufficient data in file: {len(df)} rows")
            
            if not self.calculate_indicators(df):
                raise ValueError("Indicator calculation failed")
            
            features = []
            targets = []
            
            # Skip the warmup period where indicators might be unstable
            start_idx = max(50, self.min_warmup_candles)
            
            for i in range(start_idx, len(df)-1):
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
                    
                    # Validate feature values
                    if any(np.isnan(v) or np.isinf(v) for v in feature_vector):
                        logger.debug(f"Invalid feature at index {i}: {feature_vector}")
                        continue
                    
                    target = 1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0
                    
                    features.append(feature_vector)
                    targets.append(target)
                except Exception as e:
                    logger.warning(f"Skipping index {i}: {str(e)}")
                    continue
            
            if len(features) < 100:
                raise ValueError(f"Only {len(features)} valid samples - need at least 100")
            
            return np.array(features, dtype=np.float32), np.array(targets, dtype=np.int32)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None

    def calculate_indicators(self, df):
        """Calculate technical indicators with validation"""
        try:
            if df.empty or len(df) < 200:
                logger.warning(f"Not enough data to calculate indicators ({len(df)} < 200)")
                return False
                
            # Validate input data
            if df.isnull().values.any():
                df = df.ffill().bfill()
                if df.isnull().values.any():
                    raise ValueError("Input data contains NaN values after filling")
                    
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # RSI
            self.indicators['rsi'] = RSIIndicator(close=close, window=14).rsi().ffill().bfill().clip(0, 100)
            
            # MACD
            macd = MACD(close=close)
            self.indicators['macd'] = macd.macd().ffill().bfill()
            self.indicators['macd_signal'] = macd.macd_signal().ffill().bfill()
            
            # Stochastic
            stoch = StochasticOscillator(
                high=high, low=low, close=close, 
                window=14, smooth_window=3
            )
            self.indicators['stoch_k'] = stoch.stoch().ffill().bfill()
            self.indicators['stoch_d'] = stoch.stoch_signal().ffill().bfill()
            
            # ADX
            self.indicators['adx'] = ADXIndicator(
                high=high, low=low, close=close, window=14
            ).adx().ffill().bfill()
            
            # Moving Averages
            self.indicators['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator().ffill().bfill()
            self.indicators['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator().ffill().bfill()
            
            # ATR
            self.indicators['atr'] = AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range().ffill().bfill()
            
            # Bollinger Bands
            bb = BollingerBands(close=close, window=20, window_dev=2)
            self.indicators['bollinger_upper'] = bb.bollinger_hband().ffill().bfill()
            self.indicators['bollinger_lower'] = bb.bollinger_lband().ffill().bfill()
            
            # VWAP
            self.indicators['vwap'] = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=volume, window=14
            ).volume_weighted_average_price().ffill().bfill()
            
            # OBV
            self.indicators['obv'] = OnBalanceVolumeIndicator(
                close=close, volume=volume
            ).on_balance_volume().ffill().bfill()
            
            # Validate indicators
            for name, indicator in self.indicators.items():
                if indicator is None:
                    raise ValueError(f"Indicator {name} calculation failed")
                if len(indicator) != len(df):
                    raise ValueError(f"Indicator {name} length mismatch")
                if indicator.isnull().values.any():
                    raise ValueError(f"Indicator {name} contains NaN values")
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False

    def prepare_features(self, df):
        """Prepare features for prediction with validation"""
        try:
            # Store reference to current data
            self.data = df
            
            # Validate indicators
            for name, indicator in self.indicators.items():
                if indicator is None or len(indicator) == 0:
                    logger.error(f"Indicator {name} not properly calculated")
                    return None
            
            last_index = -1
            
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
            
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.error("Features contain NaN or inf values")
                return None
                
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def dynamic_risk_adjustment(self, atr):
        """Adjust risk based on market volatility"""
        base_risk = 1.0  # Standard 1% risk
        if atr < 0.0002:  # Very low volatility
            return base_risk * 0.5  # Reduce risk
        elif atr > 0.0015:  # High volatility
            return base_risk * 1.5  # Increase risk
        return base_risk

    def generate_signal(self, data_dict, risk_percent=1.0, tp_multiplier=2.0, sl_multiplier=1.0):
        """Generate trading signal with enhanced validation"""
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
            "risk_level": risk_percent,
            "auto_trading": self.auto_trading
        }

        # Check if in manual trading mode
        if not self.auto_trading:
            signal.update({
                "signal": "HOLD",
                "reason": "Manual trading mode active",
                "confidence": 0.0
            })
            return signal

        # Check warmup status
        if not self.warmup_complete:
            current = len(data_dict['M5']) if 'M5' in data_dict else 0
            if current >= self.min_warmup_candles:
                self.warmup_complete = True
                logger.info(f"Warmup complete with {current} candles")
                signal["reason"] = "Analyzing market conditions"
            else:
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

            # Get current market conditions
            current_time = df.index[-1]
            current_price = df['close'].iloc[-1]
            atr = self.indicators['atr'].iloc[-1] if self.indicators['atr'] is not None else 0
            adx = self.indicators['adx'].iloc[-1] if self.indicators['adx'] is not None else 0
            
            # Check trading hours (05:00-09:00 PST)
            current_utc = datetime.utcnow()
            pst_offset = -8 if current_utc.month in (11, 12, 1, 2, 3) else -7  # PST/PDT adjustment
            current_pst = current_utc + timedelta(hours=pst_offset)
            current_pst_hour = current_pst.hour
            
            if not (5 <= current_pst_hour < 9):  # 05:00-09:00 PST
                signal.update({
                    "signal": "HOLD",
                    "reason": f"Outside active trading hours (Current PST: {current_pst.strftime('%H:%M')})",
                    "confidence": 0.0
                })
                return signal
                
            # Check price movement
            price_change = df['close'].iloc[-1] / df['close'].iloc[-20] - 1  # 20-period change
            if abs(price_change) < 0.001:  # Less than 0.1% movement
                signal.update({
                    "signal": "HOLD", 
                    "reason": "Insufficient price movement",
                    "confidence": 0.0
                })
                return signal

            # Check market conditions
            logger.info(f"Market Conditions - ATR: {atr:.6f}, ADX: {adx:.2f}, RSI: {self.indicators['rsi'].iloc[-1]:.2f}")

            if atr < 0.0002:  # Very low volatility
                signal.update({
                    "signal": "HOLD",
                    "reason": "Low volatility market",
                    "confidence": 0.0
                })
                return signal

            if adx < 20:  # Weak trend
                signal.update({
                    "signal": "HOLD",
                    "reason": "Weak trend market",
                    "confidence": 0.0
                })
                return signal

            # Check daily loss limit
            if self.daily_pnl < -self.daily_loss_limit:
                signal.update({
                    "signal": "HOLD",
                    "reason": "Daily loss limit reached",
                    "confidence": 0.0
                })
                return signal

            # Check minimum trade interval
            if self.last_trade_time and (datetime.now() - self.last_trade_time).total_seconds() < self.min_trade_interval:
                signal.update({
                    "signal": "HOLD",
                    "reason": "Waiting between trades",
                    "confidence": 0.0
                })
                return signal

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
                    prediction = self.model.predict(features)[0]
                    confidence = 0.7
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                signal.update({
                    "signal": "HOLD",
                    "reason": "Prediction failed",
                    "confidence": 0.0
                })
                return signal

            # Calculate TP/SL based on ATR with dynamic multiplier
            tp = current_price + (atr * tp_multiplier)
            sl = current_price - (atr * sl_multiplier)

            # Dynamic risk adjustment
            adjusted_risk = self.dynamic_risk_adjustment(atr)
            
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
                },
                "risk_level": adjusted_risk
            })

            # Get additional indicator values
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

            # Weighted confirmation scoring
            confirmation_score = 0
            confirmation_score += 0.3 if (40 < rsi_value < 60) else 0
            confirmation_score += 0.2 if (stoch_k > stoch_d) else 0
            confirmation_score += 0.2 if (macd_hist > 0) else 0
            confirmation_score += 0.2 if obv_trend else 0
            confirmation_score += 0.1 if (daily_trend or h1_trend) else 0

            # Signal generation logic with weighted scoring
            if prediction == 1 and self.current_position != "BUY":
                if confirmation_score >= 0.6 and confidence >= self.min_confidence:
                    signal.update({
                        "signal": "BUY",
                        "reason": f"AI Buy Signal (Score: {confirmation_score:.1f}/1.0)",
                        "confidence": min(confidence * (1 + confirmation_score), 0.99)
                    })
                else:
                    signal.update({
                        "signal": "HOLD",
                        "reason": f"Insufficient score ({confirmation_score:.1f}/0.6) or confidence ({confidence:.2f})",
                        "confidence": confidence * (confirmation_score / 0.6)
                    })
            
            elif prediction == 0 and self.current_position == "BUY":
                exit_conditions = 0
                
                # RSI exit
                if rsi_value > 70 or rsi_value < 30:
                    exit_conditions += 1
                
                # Stochastic exit
                if stoch_k < stoch_d:
                    exit_conditions += 1
                
                # MACD exit
                if macd_hist < -0.0005:
                    exit_conditions += 1
                
                # OBV exit
                if not obv_trend:
                    exit_conditions += 1
                
                # Trend exit
                if not (daily_trend or h1_trend):
                    exit_conditions += 1
                
                # Exit if at least 1 condition met
                if exit_conditions >= 1:
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
        """Update the current position with validation"""
        if position_type not in ["BUY", "SELL", "CLOSE", None]:
            raise ValueError("Invalid position type")
            
        if position_type == "CLOSE":
            if self.current_position is None:
                logger.warning("Attempt to close non-existent position")
                return
                
            # Calculate PnL
            if self.current_position == "BUY":
                pnl = (price - self.entry_price) * size
            else:
                pnl = (self.entry_price - price) * size
                
            self.daily_pnl += pnl
            self.total_pnl += pnl
            self.trade_count += 1
            
            if pnl > 0:
                self.win_count += 1
                
            self.win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
            self.last_trade_time = datetime.now()
            
            logger.info(f"Position closed at {price}. PnL: {pnl:.2f}")
            
            # Reset position
            self.current_position = None
            self.position_size = 0
            self.entry_price = 0
            
        else:
            if position_type == self.current_position:
                logger.warning(f"Attempt to open duplicate {position_type} position")
                return
                
            if self.current_position is not None:
                logger.warning(f"Attempt to open {position_type} position while {self.current_position} is active")
                return
                
            if price <= 0 or size <= 0:
                raise ValueError("Invalid price or size")
                
            self.current_position = position_type
            self.position_size = size
            self.entry_price = price
            self.last_trade_time = datetime.now()
            logger.info(f"{position_type} position opened at {price} with size {size}")

    def check_risk_limits(self):
        """Check if daily loss or max drawdown limits are exceeded"""
        if self.daily_pnl < -self.daily_loss_limit:
            logger.warning(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            return False
            
        if self.total_pnl < -self.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {self.total_pnl:.2f}")
            return False
            
        return True

    def get_trade_stats(self):
        """Return trading performance statistics"""
        return {
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "current_position": self.current_position,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "auto_trading": self.auto_trading
        }

class TradingBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Forex Trading Bot")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load settings
        self.settings = QSettings("AI_Trading_Bot", "Forex_Trading")
        
        # Initialize components
        self.init_ui()
        self.init_api()
        self.init_strategy()
        self.init_data_fetcher()
        self.init_model_retrainer()
        
        # Connect signals
        self.connect_signals()
        
        # Start with default settings
        self.load_settings()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Left panel (controls)
        self.left_panel = QWidget()
        self.left_panel.setMaximumWidth(400)
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        
        # Right panel (charts and logs)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)
        
        # Create tabs for left panel
        self.left_tabs = QTabWidget()
        self.left_layout.addWidget(self.left_tabs)
        
        # Create tabs
        self.create_settings_tab()
        self.create_strategy_tab()
        self.create_account_tab()
        self.create_logs_tab()
        
        # Create chart area
        self.create_chart_area()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Initializing...")
        self.status_bar.addWidget(self.status_label)
        
    def create_settings_tab(self):
        """Create settings tab"""
        self.settings_tab = QWidget()
        self.settings_layout = QVBoxLayout()
        self.settings_tab.setLayout(self.settings_layout)
        
        # API Settings Group
        api_group = QGroupBox("API Settings")
        api_form = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.account_id_input = QLineEdit()
        self.environment_combo = QComboBox()
        self.environment_combo.addItems(["Practice", "Live"])
        
        api_form.addRow("API Key:", self.api_key_input)
        api_form.addRow("Account ID:", self.account_id_input)
        api_form.addRow("Environment:", self.environment_combo)
        
        api_group.setLayout(api_form)
        self.settings_layout.addWidget(api_group)
        
        # Trading Settings Group
        trading_group = QGroupBox("Trading Settings")
        trading_form = QFormLayout()
        
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"])
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 5.0)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setSuffix("%")
        
        trading_form.addRow("Symbol:", self.symbol_combo)
        trading_form.addRow("Risk Percentage:", self.risk_spin)
        
        trading_group.setLayout(trading_form)
        self.settings_layout.addWidget(trading_group)
        
        # Trading Mode Group
        mode_group = QGroupBox("Trading Mode")
        mode_layout = QHBoxLayout()
        
        self.auto_trading_btn = QPushButton("Auto Trading")
        self.auto_trading_btn.setCheckable(True)
        self.auto_trading_btn.setChecked(True)
        self.auto_trading_btn.setStyleSheet("QPushButton:checked { background-color: green; }")
        
        self.manual_trading_btn = QPushButton("Manual Trading")
        self.manual_trading_btn.setCheckable(True)
        self.manual_trading_btn.setChecked(False)
        self.manual_trading_btn.setStyleSheet("QPushButton:checked { background-color: blue; }")
        
        mode_layout.addWidget(self.auto_trading_btn)
        mode_layout.addWidget(self.manual_trading_btn)
        mode_group.setLayout(mode_layout)
        self.settings_layout.addWidget(mode_group)
        
        # Manual Trading Controls
        self.manual_controls_group = QGroupBox("Manual Controls")
        self.manual_controls_group.setVisible(False)
        manual_controls_layout = QHBoxLayout()
        
        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet("background-color: green; color: white;")
        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet("background-color: red; color: white;")
        self.close_btn = QPushButton("CLOSE")
        self.close_btn.setStyleSheet("background-color: blue; color: white;")
        
        manual_controls_layout.addWidget(self.buy_btn)
        manual_controls_layout.addWidget(self.sell_btn)
        manual_controls_layout.addWidget(self.close_btn)
        self.manual_controls_group.setLayout(manual_controls_layout)
        self.settings_layout.addWidget(self.manual_controls_group)
        
        # Buttons
        self.save_btn = QPushButton("Save Settings")
        self.start_btn = QPushButton("Start Bot")
        self.stop_btn = QPushButton("Stop Bot")
        self.stop_btn.setEnabled(False)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        
        self.settings_layout.addLayout(button_layout)
        
        # Add tab
        self.left_tabs.addTab(self.settings_tab, "Settings")
        
    def create_strategy_tab(self):
        """Create strategy configuration tab"""
        self.strategy_tab = QWidget()
        self.strategy_layout = QVBoxLayout()
        self.strategy_tab.setLayout(self.strategy_layout)
        
        # Strategy Parameters Group
        params_group = QGroupBox("Strategy Parameters")
        params_form = QFormLayout()
        
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.1, 0.99)
        self.min_confidence_spin.setValue(0.50)
        self.min_confidence_spin.setSingleStep(0.01)
        
        self.atr_multiplier_spin = QDoubleSpinBox()
        self.atr_multiplier_spin.setRange(1.0, 5.0)
        self.atr_multiplier_spin.setValue(2.0)
        self.atr_multiplier_spin.setSingleStep(0.1)
        
        self.warmup_candles_spin = QSpinBox()
        self.warmup_candles_spin.setRange(10, 200)
        self.warmup_candles_spin.setValue(50)
        
        params_form.addRow("Min Confidence:", self.min_confidence_spin)
        params_form.addRow("ATR Multiplier:", self.atr_multiplier_spin)
        params_form.addRow("Warmup Candles:", self.warmup_candles_spin)
        
        params_group.setLayout(params_form)
        self.strategy_layout.addWidget(params_group)
        
        # Risk Management Group
        risk_group = QGroupBox("Risk Management")
        risk_form = QFormLayout()
        
        self.daily_loss_spin = QDoubleSpinBox()
        self.daily_loss_spin.setRange(0.1, 10.0)
        self.daily_loss_spin.setValue(2.0)
        self.daily_loss_spin.setSuffix("%")
        
        self.max_drawdown_spin = QDoubleSpinBox()
        self.max_drawdown_spin.setRange(1.0, 20.0)
        self.max_drawdown_spin.setValue(5.0)
        self.max_drawdown_spin.setSuffix("%")
        
        risk_form.addRow("Daily Loss Limit:", self.daily_loss_spin)
        risk_form.addRow("Max Drawdown:", self.max_drawdown_spin)
        
        risk_group.setLayout(risk_form)
        self.strategy_layout.addWidget(risk_group)
        
        # Model Management Group
        model_group = QGroupBox("Model Management")
        model_layout = QVBoxLayout()
        
        self.retrain_check = QCheckBox("Enable Auto-Retraining")
        self.retrain_check.setChecked(True)
        
        self.retrain_btn = QPushButton("Retrain Model Now")
        self.save_model_btn = QPushButton("Save Model As...")
        self.load_model_btn = QPushButton("Load Model...")
        
        model_layout.addWidget(self.retrain_check)
        model_layout.addWidget(self.retrain_btn)
        model_layout.addWidget(self.save_model_btn)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        self.strategy_layout.addWidget(model_group)
        
        # Add tab
        self.left_tabs.addTab(self.strategy_tab, "Strategy")
        
    def create_account_tab(self):
        """Create account information tab"""
        self.account_tab = QWidget()
        self.account_layout = QVBoxLayout()
        self.account_tab.setLayout(self.account_layout)
        
        # Account Info Group
        info_group = QGroupBox("Account Information")
        info_form = QFormLayout()
        
        self.balance_label = QLabel("N/A")
        self.equity_label = QLabel("N/A")
        self.margin_label = QLabel("N/A")
        self.free_margin_label = QLabel("N/A")
        self.leverage_label = QLabel("N/A")
        
        info_form.addRow("Balance:", self.balance_label)
        info_form.addRow("Equity:", self.equity_label)
        info_form.addRow("Margin:", self.margin_label)
        info_form.addRow("Free Margin:", self.free_margin_label)
        info_form.addRow("Leverage:", self.leverage_label)
        
        info_group.setLayout(info_form)
        self.account_layout.addWidget(info_group)
        
        # Positions Group
        self.positions_group = QGroupBox("Open Positions")
        self.positions_layout = QVBoxLayout()
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(["ID", "Symbol", "Side", "Size", "PnL"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.positions_layout.addWidget(self.positions_table)
        self.positions_group.setLayout(self.positions_layout)
        self.account_layout.addWidget(self.positions_group)
        
        # System Status Group
        self.status_group = QGroupBox("System Status")
        self.status_layout = QFormLayout()
        
        self.model_status_label = QLabel("Not Ready")
        self.data_status_label = QLabel("No Data")
        self.trading_mode_label = QLabel("Auto Trading")
        self.signal_status_label = QLabel("No Signal")
        
        self.status_layout.addRow("Model Status:", self.model_status_label)
        self.status_layout.addRow("Data Status:", self.data_status_label)
        self.status_layout.addRow("Trading Mode:", self.trading_mode_label)
        self.status_layout.addRow("Last Signal:", self.signal_status_label)
        
        self.status_group.setLayout(self.status_layout)
        self.account_layout.addWidget(self.status_group)
        
        # Add tab
        self.left_tabs.addTab(self.account_tab, "Account")
        
    def create_logs_tab(self):
        """Create logs tab"""
        self.logs_tab = QWidget()
        self.logs_layout = QVBoxLayout()
        self.logs_tab.setLayout(self.logs_layout)
        
        # Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace;")
        
        # Clear button
        self.clear_logs_btn = QPushButton("Clear Logs")
        
        self.logs_layout.addWidget(self.log_text)
        self.logs_layout.addWidget(self.clear_logs_btn)
        
        # Add tab
        self.left_tabs.addTab(self.logs_tab, "Logs")
        
    def create_chart_area(self):
        """Create chart display area"""
        # Chart container
        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout()
        self.chart_container.setLayout(self.chart_layout)
        
        # Chart type selector
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Candlestick", "Line", "OHLC"])
        self.chart_layout.addWidget(self.chart_type_combo)
        
        # Chart canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.chart_layout.addWidget(self.canvas)
        
        # Signal display
        self.signal_display = QLabel("No signal generated yet")
        self.signal_display.setAlignment(Qt.AlignCenter)
        self.signal_display.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.chart_layout.addWidget(self.signal_display)
        
        # Add to right panel
        self.right_layout.addWidget(self.chart_container)
        
    def init_api(self):
        """Initialize API connection with validation"""
        try:
            # First try to load from .env file
            load_dotenv()  # Load environment variables from .env file
            
            api_key = os.getenv('OANDA_API_KEY') or self.api_key_input.text().strip()
            account_id = os.getenv('OANDA_ACCOUNT_ID') or self.account_id_input.text().strip()
            
            if not api_key or not account_id:
                raise ValueError("API Key and Account ID are required")
                
            # Update the UI fields with the values from .env
            self.api_key_input.setText(api_key)
            self.account_id_input.setText(account_id)
                
            environment = "practice" if self.environment_combo.currentText() == "Practice" else "live"
            self.api = API(access_token=api_key, environment=environment)
            
            # Test API connection
            endpoint = AccountDetails(accountID=account_id)
            response = self.api.request(endpoint)
            
            logger.debug(f"Account details response: {response}")
            
            if 'account' not in response:
                raise ValueError("API connection test failed")
                
            self.api_connected = True
            logger.info("API connection established successfully")
        except Exception as e:
            logger.error(f"API initialization failed: {str(e)}", exc_info=True)
            self.api_connected = False
            raise
        
    def test_instruments_endpoint(self):
        """Test the instruments endpoint directly"""
        try:
            params = {
                "granularity": "M5",
                "count": 10,
                "price": "M"
            }
            endpoint = InstrumentsCandles(instrument=self.symbol_combo.currentText(), params=params)
            response = self.api.request(endpoint)
            
            logger.debug(f"Instruments test response: {response}")
            
            if 'candles' not in response:
                raise ValueError("No candles in response")
                
            if len(response['candles']) == 0:
                raise ValueError("Empty candles array")
                
            return True
            
        except Exception as e:
            logger.error(f"Instruments endpoint test failed: {str(e)}")
            return False
        
    def init_strategy(self):
        """Initialize trading strategy"""
        self.strategy = None
        self.strategy_active = False
        
    def init_data_fetcher(self):
        """Initialize data fetcher"""
        self.data_fetcher = None
        self.data_timer = QTimer()
        self.data_timer.setInterval(1000)  # 1 second
        
    def init_model_retrainer(self):
        """Initialize model retrainer"""
        self.model_retrainer = None
        
    def connect_signals(self):
        """Connect UI signals to slots"""
        # Buttons
        self.save_btn.clicked.connect(self.save_settings)
        self.start_btn.clicked.connect(self.start_bot)
        self.stop_btn.clicked.connect(self.stop_bot)
        self.retrain_btn.clicked.connect(self.manual_retrain)
        self.save_model_btn.clicked.connect(self.save_model)
        self.load_model_btn.clicked.connect(self.load_model)
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        
        # Trading mode buttons
        self.auto_trading_btn.clicked.connect(self.set_auto_trading)
        self.manual_trading_btn.clicked.connect(self.set_manual_trading)
        
        # Manual trading buttons
        self.buy_btn.clicked.connect(self.manual_buy)
        self.sell_btn.clicked.connect(self.manual_sell)
        self.close_btn.clicked.connect(self.manual_close)
        
        # Comboboxes
        self.symbol_combo.currentTextChanged.connect(self.symbol_changed)
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        
        # Timers
        self.data_timer.timeout.connect(lambda: self.handle_new_data(self.data_fetcher.get_last_data() if self.data_fetcher else None))
        
    def set_auto_trading(self):
        """Set trading mode to auto"""
        self.auto_trading_btn.setChecked(True)
        self.manual_trading_btn.setChecked(False)
        self.manual_controls_group.setVisible(False)
        
        if self.strategy is not None:
            self.strategy.set_auto_trading(True)
            
        self.trading_mode_label.setText("Auto Trading")
        self.log_message("Switched to Auto Trading mode")
        
    def set_manual_trading(self):
        """Set trading mode to manual"""
        self.auto_trading_btn.setChecked(False)
        self.manual_trading_btn.setChecked(True)
        self.manual_controls_group.setVisible(True)
        
        if self.strategy is not None:
            self.strategy.set_auto_trading(False)
            
        self.trading_mode_label.setText("Manual Trading")
        self.log_message("Switched to Manual Trading mode")
        
    def manual_buy(self):
        """Execute manual buy"""
        if self.strategy is None or not self.strategy_active:
            self.log_message("Cannot execute manual trade - bot not running", level="warning")
            return
            
        # Get current price from last signal or data
        current_price = 0.0
        if self.strategy.last_signal:
            current_price = self.strategy.last_signal['price']
        elif self.data_fetcher and self.data_fetcher.get_last_data():
            current_price = self.data_fetcher.get_last_data()['M5']['close'].iloc[-1]
            
        if current_price <= 0:
            self.log_message("Cannot determine current price", level="error")
            return
            
        # Calculate position size based on risk
        account_info = self.get_account_info()
        if account_info is None:
            return
            
        risk_amount = account_info['balance'] * (self.risk_spin.value() / 100)
        stop_loss_pips = 0.0020  # Default 20 pips if we can't calculate
        if self.strategy.last_signal:
            stop_loss_pips = abs(self.strategy.last_signal['price'] - self.strategy.last_signal['sl'])
        pip_value = 10  # Standard for FX pairs with USD as quote
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Create signal
        signal = {
            "signal": "BUY",
            "reason": "Manual Buy",
            "price": current_price,
            "tp": current_price * 1.005,  # 50 pips TP
            "sl": current_price * 0.995,  # 50 pips SL
            "confidence": 1.0,
            "time": datetime.now().strftime('%H:%M:%S')
        }
        
        self.display_signal(signal)
        self.execute_trade(signal)
        
    def manual_sell(self):
        """Execute manual sell"""
        if self.strategy is None or not self.strategy_active:
            self.log_message("Cannot execute manual trade - bot not running", level="warning")
            return
            
        # Get current price from last signal or data
        current_price = 0.0
        if self.strategy.last_signal:
            current_price = self.strategy.last_signal['price']
        elif self.data_fetcher and self.data_fetcher.get_last_data():
            current_price = self.data_fetcher.get_last_data()['M5']['close'].iloc[-1]
            
        if current_price <= 0:
            self.log_message("Cannot determine current price", level="error")
            return
            
        # Calculate position size based on risk
        account_info = self.get_account_info()
        if account_info is None:
            return
            
        risk_amount = account_info['balance'] * (self.risk_spin.value() / 100)
        stop_loss_pips = 0.0020  # Default 20 pips if we can't calculate
        if self.strategy.last_signal:
            stop_loss_pips = abs(self.strategy.last_signal['price'] - self.strategy.last_signal['sl'])
        pip_value = 10  # Standard for FX pairs with USD as quote
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Create signal
        signal = {
            "signal": "SELL",
            "reason": "Manual Sell",
            "price": current_price,
            "tp": current_price * 0.995,  # 50 pips TP
            "sl": current_price * 1.005,  # 50 pips SL
            "confidence": 1.0,
            "time": datetime.now().strftime('%H:%M:%S')
        }
        
        self.display_signal(signal)
        self.execute_trade(signal)
        
    def manual_close(self):
        """Execute manual close"""
        if self.strategy is None or not self.strategy_active:
            self.log_message("Cannot execute manual trade - bot not running", level="warning")
            return
            
        if self.strategy.current_position is None:
            self.log_message("No position to close", level="warning")
            return
            
        # Get current price from last signal or data
        current_price = 0.0
        if self.strategy.last_signal:
            current_price = self.strategy.last_signal['price']
        elif self.data_fetcher and self.data_fetcher.get_last_data():
            current_price = self.data_fetcher.get_last_data()['M5']['close'].iloc[-1]
            
        if current_price <= 0:
            self.log_message("Cannot determine current price", level="error")
            return
            
        # Create signal
        signal = {
            "signal": "CLOSE",
            "reason": "Manual Close",
            "price": current_price,
            "tp": 0.0,
            "sl": 0.0,
            "confidence": 1.0,
            "time": datetime.now().strftime('%H:%M:%S')
        }
        
        self.display_signal(signal)
        self.execute_trade(signal)
        
    def load_settings(self):
        """Load settings from QSettings"""
        self.api_key_input.setText(self.settings.value("api_key", ""))
        self.account_id_input.setText(self.settings.value("account_id", ""))
        self.environment_combo.setCurrentText(self.settings.value("environment", "Practice"))
        self.symbol_combo.setCurrentText(self.settings.value("symbol", "EUR_USD"))
        self.risk_spin.setValue(float(self.settings.value("risk_percent", 1.0)))
        
        # Strategy settings
        self.min_confidence_spin.setValue(float(self.settings.value("min_confidence", 0.50)))
        self.atr_multiplier_spin.setValue(float(self.settings.value("atr_multiplier", 2.0)))
        self.warmup_candles_spin.setValue(int(self.settings.value("warmup_candles", 50)))
        self.daily_loss_spin.setValue(float(self.settings.value("daily_loss_limit", 2.0)))
        self.max_drawdown_spin.setValue(float(self.settings.value("max_drawdown", 5.0)))
        self.retrain_check.setChecked(self.settings.value("auto_retrain", True, type=bool))
        
        # Trading mode
        if self.settings.value("trading_mode", "auto") == "manual":
            self.set_manual_trading()
        else:
            self.set_auto_trading()
        
    def save_settings(self):
        """Save settings to QSettings"""
        self.settings.setValue("api_key", self.api_key_input.text())
        self.settings.setValue("account_id", self.account_id_input.text())
        self.settings.setValue("environment", self.environment_combo.currentText())
        self.settings.setValue("symbol", self.symbol_combo.currentText())
        self.settings.setValue("risk_percent", self.risk_spin.value())
        
        # Strategy settings
        self.settings.setValue("min_confidence", self.min_confidence_spin.value())
        self.settings.setValue("atr_multiplier", self.atr_multiplier_spin.value())
        self.settings.setValue("warmup_candles", self.warmup_candles_spin.value())
        self.settings.setValue("daily_loss_limit", self.daily_loss_spin.value())
        self.settings.setValue("max_drawdown", self.max_drawdown_spin.value())
        self.settings.setValue("auto_retrain", self.retrain_check.isChecked())
        
        # Trading mode
        mode = "manual" if self.manual_trading_btn.isChecked() else "auto"
        self.settings.setValue("trading_mode", mode)
        
        self.log_message("Settings saved successfully")
        
    def symbol_changed(self, symbol):
        """Handle symbol change"""
        if self.strategy is not None:
            self.strategy.symbol = symbol
            if self.data_fetcher is not None:
                self.data_fetcher.symbol = symbol
                self.data_fetcher.force_refresh()
                
        self.update_chart()
        
    def start_bot(self):
        """Start the trading bot with enhanced validation"""
        try:
            logger.debug("Start button clicked - beginning startup sequence")
            
            # Validate API settings
            api_key = self.api_key_input.text().strip()
            account_id = self.account_id_input.text().strip()
            
            if not api_key or not account_id:
                raise ValueError("API Key and Account ID are required")
                
            # Initialize API
            environment = "practice" if self.environment_combo.currentText() == "Practice" else "live"
            self.api = API(access_token=api_key, environment=environment)
            
            # Verify API connection first
            try:
                logger.debug("Testing API connection...")
                endpoint = AccountDetails(accountID=account_id)
                account_info = self.api.request(endpoint)
                logger.debug(f"Account info: {account_info}")
                if 'account' not in account_info:
                    raise ValueError("Invalid account response")
            except Exception as e:
                raise RuntimeError(f"API connection test failed: {str(e)}")
            
            # Test instruments endpoint
            logger.debug("Testing instruments endpoint...")
            if not self.test_instruments_endpoint():
                raise RuntimeError("Failed to fetch candle data from instruments endpoint")
            
            # Initialize strategy
            symbol = self.symbol_combo.currentText()
            model_path = f"models/ai_model_{symbol}.pkl"
            
            logger.debug(f"Initializing strategy for {symbol} with model at {model_path}")
            self.strategy = AITradingStrategy(
                symbol=symbol,
                model_path=model_path
            )
            
            # Set trading mode
            if self.manual_trading_btn.isChecked():
                self.strategy.set_auto_trading(False)
                logger.debug("Manual trading mode set")
            else:
                self.strategy.set_auto_trading(True)
                logger.debug("Auto trading mode set")
            
            # Update strategy parameters
            self.update_strategy_params()
            logger.debug("Strategy parameters updated")
            
            # Verify model initialized
            if not self.strategy.initialized:
                raise RuntimeError("Failed to initialize trading model")
            
            # Initialize data fetcher
            logger.debug("Initializing data fetcher...")
            self.data_fetcher = DataFetcher(
                api=self.api,
                symbol=symbol,
                granularities=["M5", "H1", "D"],
                count=500
            )
            
            # Connect data fetcher signals
            self.data_fetcher.data_updated.connect(self.handle_new_data)
            self.data_fetcher.error_occurred.connect(self.handle_error)
            
            # Initialize model retrainer if enabled
            if self.retrain_check.isChecked():
                logger.debug("Initializing model retrainer...")
                self.model_retrainer = ModelRetrainer(
                    strategy=self.strategy,
                    data_fetcher=self.data_fetcher,
                    retrain_interval=86400  # 24 hours
                )
                self.model_retrainer.retraining_complete.connect(self.handle_retraining_result)
                self.model_retrainer.start()
            
            # Start components
            logger.debug("Starting data fetcher...")
            self.data_fetcher.start()
            self.data_timer.start()
            
            # Update UI state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.strategy_active = True
            
            # Initial data fetch
            logger.debug("Performing initial data fetch...")
            if not self.data_fetcher.force_refresh():
                raise RuntimeError("Initial data fetch failed")
            
            self.log_message("Trading bot started successfully")
            self.status_label.setText("Running")
            
        except Exception as e:
            error_msg = f"Error starting bot: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            QMessageBox.critical(self, "Start Error", f"Failed to start trading bot:\n{str(e)}")
            
            # Reset state if startup failed
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.strategy_active = False
            
    def stop_bot(self):
        """Stop the trading bot"""
        try:
            logger.debug("Stopping bot...")
            
            # Stop components
            if self.data_fetcher:
                self.data_fetcher.stop()
                self.data_fetcher.wait()
                
            if self.model_retrainer:
                self.model_retrainer.stop()
                self.model_retrainer.wait()
                
            self.data_timer.stop()
            
            # Update UI state
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.strategy_active = False
            
            self.log_message("Trading bot stopped")
            self.status_label.setText("Stopped")
            
        except Exception as e:
            error_msg = f"Error stopping bot: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            
    def update_strategy_params(self):
        """Update strategy parameters from UI"""
        if self.strategy is None:
            return
            
        self.strategy.min_confidence = self.min_confidence_spin.value()
        self.strategy.atr_multiplier = self.atr_multiplier_spin.value()
        self.strategy.min_warmup_candles = self.warmup_candles_spin.value()
        self.strategy.daily_loss_limit = self.daily_loss_spin.value() / 100
        self.strategy.max_drawdown = self.max_drawdown_spin.value() / 100
        
        self.log_message("Strategy parameters updated")
        
    def handle_new_data(self, data_dict):
        """Handle new data from fetcher"""
        try:
            if not data_dict or 'M5' not in data_dict:
                logger.debug("No valid M5 data in update")
                return
                
            # Update chart
            self.update_chart(data_dict['M5'])
            
            # Generate signal if we have a strategy
            if self.strategy and self.strategy_active:
                signal = self.strategy.generate_signal(
                    data_dict,
                    risk_percent=self.risk_spin.value(),
                    tp_multiplier=self.strategy.atr_multiplier,
                    sl_multiplier=self.strategy.atr_multiplier * 0.8
                )
                
                # Display signal
                self.display_signal(signal)
                
                # Execute trade if in auto mode
                if self.strategy.auto_trading:
                    self.execute_trade(signal)
                
            # Update status
            self.data_status_label.setText(f"Data: {len(data_dict['M5'])} candles")
            if self.strategy:
                self.model_status_label.setText("Model: Ready" if self.strategy.initialized else "Model: Not Ready")
                
        except Exception as e:
            error_msg = f"Error processing new data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            
    def handle_error(self, error_msg):
        """Handle errors from components"""
        self.log_message(f"Error: {error_msg}", level="error")
        self.status_label.setText(f"Error: {error_msg[:30]}...")
        
    def handle_retraining_result(self, success, message):
        """Handle model retraining results"""
        if success:
            self.log_message(f"Model retraining successful: {message}")
            self.model_status_label.setText("Model: Ready (Retrained)")
        else:
            self.log_message(f"Model retraining failed: {message}", level="warning")
            
    def manual_retrain(self):
        """Manually trigger model retraining"""
        if self.model_retrainer is None:
            self.log_message("Model retrainer not initialized", level="warning")
            return
            
        self.log_message("Starting manual model retraining...")
        self.model_retrainer.retrain_model()
        
    def save_model(self):
        """Save the current model to a file"""
        if self.strategy is None or self.strategy.model is None:
            self.log_message("No model to save", level="warning")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "models", "Model Files (*.pkl)"
        )
        
        if file_path:
            try:
                joblib.dump(self.strategy.model, file_path)
                self.log_message(f"Model saved to {file_path}")
            except Exception as e:
                error_msg = f"Error saving model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.log_message(error_msg, level="error")
                
    def load_model(self):
        """Load a model from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "models", "Model Files (*.pkl)"
        )
        
        if file_path:
            try:
                model = joblib.load(file_path)
                if not hasattr(model, 'predict'):
                    raise ValueError("Invalid model format")
                    
                if self.strategy is None:
                    symbol = self.symbol_combo.currentText()
                    self.strategy = AITradingStrategy(symbol=symbol)
                    
                self.strategy.model = model
                self.strategy._initialize_model()
                self.log_message(f"Model loaded from {file_path}")
                self.model_status_label.setText("Model: Ready (Loaded)")
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.log_message(error_msg, level="error")
                
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.clear()
        
    def log_message(self, message, level="info"):
        """Log a message to the UI and log file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] {message}"
        
        # Color coding based on level
        if level == "error":
            self.log_text.setTextColor(QColor(255, 0, 0))
        elif level == "warning":
            self.log_text.setTextColor(QColor(255, 165, 0))
        else:
            self.log_text.setTextColor(QColor(0, 0, 0))
            
        self.log_text.append(formatted_msg)
        
        # Also log to file
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
            
    def update_chart(self, df=None):
        """Update the price chart"""
        try:
            self.figure.clear()
            
            if df is None and self.data_fetcher:
                data = self.data_fetcher.get_last_data()
                df = data['M5'] if data and 'M5' in data else None
                
            if df is None or df.empty:
                logger.debug("No data available for chart update")
                return
                
            # Prepare data for mplfinance
            plot_df = df.copy()
            plot_df.columns = [c.capitalize() for c in plot_df.columns]
            
            # Determine chart style
            chart_type = self.chart_type_combo.currentText().lower()
            style = 'yahoo'
            
            if chart_type == 'candlestick':
                mtype = 'candle'
            elif chart_type == 'ohlc':
                mtype = 'ohlc'
            else:
                mtype = 'line'
                
            # Create subplots
            ax1 = self.figure.add_subplot(3, 1, (1, 2))
            ax2 = self.figure.add_subplot(3, 1, 3, sharex=ax1)
            
            # Plot price data
            mpf.plot(
                plot_df[-200:],  # Show last 200 candles
                type=mtype,
                style=style,
                ax=ax1,
                volume=ax2,
                show_nontrading=False,
                returnfig=True
            )
            
            # Add indicators if available
            if hasattr(self.strategy, 'indicators'):
                indicators = self.strategy.indicators
                
                # Plot RSI
                if indicators.get('rsi') is not None:
                    ax1r = ax1.twinx()
                    ax1r.plot(indicators['rsi'][-200:], label='RSI', color='purple', alpha=0.5)
                    ax1r.axhline(70, color='red', linestyle='--', alpha=0.3)
                    ax1r.axhline(30, color='green', linestyle='--', alpha=0.3)
                    ax1r.set_ylim(0, 100)
                    ax1r.legend(loc='upper right')
                
                # Plot MACD
                if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
                    ax2.plot(indicators['macd'][-200:], label='MACD', color='blue')
                    ax2.plot(indicators['macd_signal'][-200:], label='Signal', color='orange')
                    ax2.legend()
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            error_msg = f"Error updating chart: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            
    def display_signal(self, signal):
        """Display the trading signal in the UI"""
        if not signal:
            return
            
        # Update signal display
        signal_text = f"{signal['time']} - {signal['signal']}: {signal['reason']}"
        self.signal_status_label.setText(f"Last Signal: {signal_text}")
        
        # Color code based on signal type
        if signal['signal'] == "BUY":
            self.signal_display.setStyleSheet("background-color: green; color: white; font-size: 16px;")
            self.signal_display.setText(f"BUY SIGNAL\nConfidence: {signal['confidence']:.2f}")
        elif signal['signal'] == "SELL":
            self.signal_display.setStyleSheet("background-color: red; color: white; font-size: 16px;")
            self.signal_display.setText(f"SELL SIGNAL\nConfidence: {signal['confidence']:.2f}")
        elif signal['signal'] == "CLOSE":
            self.signal_display.setStyleSheet("background-color: blue; color: white; font-size: 16px;")
            self.signal_display.setText(f"CLOSE POSITION\n{signal['reason']}")
        else:
            self.signal_display.setStyleSheet("font-size: 16px;")
            self.signal_display.setText(f"NO TRADE\n{signal['reason']}")
            
        # Log the signal
        self.log_message(f"Signal generated: {signal_text}")
        
    def execute_trade(self, signal):
        """Execute a trade based on the signal"""
        if not signal or signal['signal'] == "HOLD":
            return
            
        try:
            # Get account info for position sizing
            account_info = self.get_account_info()
            if not account_info:
                raise ValueError("Could not get account information")
                
            # Calculate position size based on risk
            risk_amount = account_info['balance'] * (signal['risk_level'] / 100)
            stop_loss_pips = abs(signal['price'] - signal['sl'])
            pip_value = 10  # Standard for FX pairs with USD as quote
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Execute based on signal
            if signal['signal'] == "BUY":
                self.create_order("BUY", signal['price'], position_size, signal['tp'], signal['sl'])
                self.strategy.update_position("BUY", signal['price'], position_size)
                
            elif signal['signal'] == "SELL":
                self.create_order("SELL", signal['price'], position_size, signal['tp'], signal['sl'])
                self.strategy.update_position("SELL", signal['price'], position_size)
                
            elif signal['signal'] == "CLOSE" and self.strategy.current_position:
                self.close_position()
                self.strategy.update_position("CLOSE", signal['price'], self.strategy.position_size)
                
            # Update account info
            self.update_account_display()
            
        except Exception as e:
            error_msg = f"Trade execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            
    def create_order(self, side, price, units, take_profit, stop_loss):
        """Create an order with OANDA API"""
        try:
            account_id = self.account_id_input.text().strip()
            if not account_id:
                raise ValueError("Account ID not set")
                
            # Prepare order request
            data = {
                "order": {
                    "instrument": self.strategy.symbol,
                    "units": str(round(units)) if side == "BUY" else str(-round(units)),
                    "type": "MARKET",
                    "takeProfitOnFill": {
                        "price": str(round(take_profit, 5))
                    },
                    "stopLossOnFill": {
                        "price": str(round(stop_loss, 5))
                    }
                }
            }
            
            endpoint = OrderCreate(accountID=account_id, data=data)
            response = self.api.request(endpoint)
            
            if 'orderFillTransaction' in response:
                self.log_message(f"{side} order executed: {response['orderFillTransaction']['id']}")
                return True
            else:
                raise ValueError(f"Order creation failed: {response}")
                
        except Exception as e:
            error_msg = f"Error creating {side} order: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            return False
            
    def close_position(self):
        """Close the current position with OANDA API"""
        try:
            account_id = self.account_id_input.text().strip()
            if not account_id:
                raise ValueError("Account ID not set")
                
            # Get open trades
            endpoint = TradeOpen(accountID=account_id)
            response = self.api.request(endpoint)
            
            if 'trades' not in response or len(response['trades']) == 0:
                raise ValueError("No open trades found")
                
            # Close all trades for our symbol
            for trade in response['trades']:
                if trade['instrument'] == self.strategy.symbol:
                    close_endpoint = TradeClose(
                        accountID=account_id,
                        tradeID=trade['id'],
                        data={"units": "ALL"}
                    )
                    close_response = self.api.request(close_endpoint)
                    self.log_message(f"Trade closed: {close_response}")
                    
            return True
            
        except Exception as e:
            error_msg = f"Error closing position: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            return False
            
    def get_account_info(self):
        """Get account information from OANDA"""
        try:
            account_id = self.account_id_input.text().strip()
            if not account_id:
                return None
                
            endpoint = AccountDetails(accountID=account_id)
            response = self.api.request(endpoint)
            
            if 'account' not in response:
                return None
                
            return {
                'balance': float(response['account']['balance']),
                'equity': float(response['account']['NAV']),
                'margin': float(response['account']['marginUsed']),
                'free_margin': float(response['account']['marginAvailable']),
                'leverage': int(response['account']['marginRate'])
            }
            
        except Exception as e:
            error_msg = f"Error getting account info: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")
            return None
            
    def update_account_display(self):
        """Update the account information display"""
        account_info = self.get_account_info()
        if not account_info:
            return
            
        self.balance_label.setText(f"{account_info['balance']:.2f}")
        self.equity_label.setText(f"{account_info['equity']:.2f}")
        self.margin_label.setText(f"{account_info['margin']:.2f}")
        self.free_margin_label.setText(f"{account_info['free_margin']:.2f}")
        self.leverage_label.setText(f"1:{account_info['leverage']}")
        
        # Update positions table
        self.update_positions_table()
        
    def update_positions_table(self):
        """Update the positions table with current trades"""
        try:
            account_id = self.account_id_input.text().strip()
            if not account_id:
                return
                
            endpoint = TradeOpen(accountID=account_id)
            response = self.api.request(endpoint)
            
            self.positions_table.setRowCount(0)
            
            if 'trades' not in response:
                return
                
            for trade in response['trades']:
                if trade['instrument'] != self.strategy.symbol:
                    continue
                    
                row = self.positions_table.rowCount()
                self.positions_table.insertRow(row)
                
                # Calculate PnL
                current_price = float(trade['currentPrice'])
                open_price = float(trade['price'])
                units = float(trade['currentUnits'])
                pnl = (current_price - open_price) * units if units > 0 else (open_price - current_price) * abs(units)
                
                # Color based on PnL
                color = QColor(0, 255, 0) if pnl >= 0 else QColor(255, 0, 0)
                
                # Add items to table
                self.positions_table.setItem(row, 0, QTableWidgetItem(trade['id']))
                self.positions_table.setItem(row, 1, QTableWidgetItem(trade['instrument']))
                self.positions_table.setItem(row, 2, QTableWidgetItem("LONG" if float(trade['currentUnits']) > 0 else "SHORT"))
                self.positions_table.setItem(row, 3, QTableWidgetItem(str(abs(units))))  # Fixed line
                
                pnl_item = QTableWidgetItem(f"{pnl:.2f}")
                pnl_item.setForeground(color)
                self.positions_table.setItem(row, 4, pnl_item)
                
        except Exception as e:
            error_msg = f"Error updating positions table: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_message(error_msg, level="error")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = TradingBotGUI()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()