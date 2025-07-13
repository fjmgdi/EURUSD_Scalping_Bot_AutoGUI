import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_price_data(csv_file):
    """Load historical candle data from a CSV (with OHLCV columns)."""
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    return df

def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    return df

def create_labels(df):
    """Label: 1 if next candle closes higher, else 0"""
    df['next_close'] = df['close'].shift(-1)
    df['label'] = (df['next_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 'atr']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.2%}")

    joblib.dump(model, "xgb_next_candle.model")
    print("Model saved as 'xgb_next_candle.model'")

if __name__ == "__main__":
    # Replace this path with your historical candle CSV file (M5 or M1 recommended)
    df = load_price_data("historical_data.csv")
    df = add_indicators(df)
    df = create_labels(df)
    train_model(df)
