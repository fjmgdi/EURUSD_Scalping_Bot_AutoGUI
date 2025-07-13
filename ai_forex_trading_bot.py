import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Required for Render
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from oandapyV20 import API
import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from flask import Flask, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Port configuration for Render
port = int(os.environ.get('PORT', 10000))

# Initialize Flask app
app = Flask(__name__)

class TradingBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OANDA_API_KEY")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.api = None
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize or load the trading model"""
        try:
            # Try to load existing model
            model_path = "models/latest_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("Loaded existing model")
            else:
                self.train_model()
                joblib.dump(self.model, model_path)
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def train_model(self):
        """Train a new LightGBM model"""
        logger.info("Training new model...")
        
        # Create synthetic training data
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        # Create model pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lgbm', lgb.LGBMClassifier())
        ])
        
        # Train model
        self.model.fit(X, y)
        logger.info("Model training complete")

    def run(self):
        """Main trading loop"""
        try:
            logger.info(f"Starting trading bot on port {port}")
            
            # Initialize API connection
            self.api = API(access_token=self.api_key, environment="practice")
            
            # Main trading loop
            while True:
                try:
                    # Get market data and generate signals
                    signal = self.generate_signal()
                    
                    # Execute trading logic
                    if signal == "BUY":
                        self.execute_trade("BUY")
                    elif signal == "SELL":
                        self.execute_trade("SELL")
                        
                    # Sleep to avoid rate limits
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            logger.info("Shutting down trading bot")
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            raise

    def generate_signal(self):
        """Generate trading signals"""
        # In a real implementation, this would use actual market data
        return "BUY" if np.random.rand() > 0.5 else "SELL"

    def execute_trade(self, signal):
        """Execute a trade through OANDA API"""
        logger.info(f"Executing {signal} trade")
        # Actual trade execution logic would go here

# Initialize the trading bot
bot = TradingBot()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint for receiving trading signals via webhook"""
    try:
        data = request.json
        logger.info(f"Received webhook data: {data}")
        
        signal = data.get('signal', '').upper()
        if signal not in ['BUY', 'SELL']:
            return "Invalid signal", 400
            
        bot.execute_trade(signal)
        return "OK", 200
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return "Error processing webhook", 500

def run_bot():
    """Run the trading bot in a separate thread"""
    import threading
    thread = threading.Thread(target=bot.run)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    print(f"Application starting on port {port}")
    try:
        # Create required directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Start the trading bot in background
        run_bot()
        
        # Start the Flask web server
        app.run(host="0.0.0.0", port=port)
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)