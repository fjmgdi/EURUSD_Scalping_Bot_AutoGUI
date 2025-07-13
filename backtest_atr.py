import pandas as pd
import numpy as np
from datetime import datetime

def calculate_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, atr_period=14):
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(atr_period).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()

    df.dropna(inplace=True)
    return df

def generate_signals(df):
    df['signal'] = 0
    # Buy signal: MACD crosses above signal line, RSI > 50, close above previous close
    buy_condition = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift()) & (df['rsi'] > 50)
    df.loc[buy_condition, 'signal'] = 1
    # Sell signal: MACD crosses below signal line, RSI < 50, close below previous close
    sell_condition = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift()) & (df['rsi'] < 50)
    df.loc[sell_condition, 'signal'] = -1
    return df

def backtest(df, initial_balance=10000, units=1000):
    balance = initial_balance
    position = 0  # +1 for long, -1 for short, 0 for flat
    entry_price = 0
    trades = []

    for i in range(1, len(df)):
        signal = df['signal'].iloc[i]

        # Enter long
        if signal == 1 and position == 0:
            position = 1
            entry_price = df['close'].iloc[i]
            trades.append(('BUY', df.index[i], entry_price))
            print(f"Buy at {entry_price} on {df.index[i]}")

        # Enter short
        elif signal == -1 and position == 0:
            position = -1
            entry_price = df['close'].iloc[i]
            trades.append(('SELL', df.index[i], entry_price))
            print(f"Sell at {entry_price} on {df.index[i]}")

        # Exit long
        elif signal == -1 and position == 1:
            exit_price = df['close'].iloc[i]
            profit = (exit_price - entry_price) * units
            balance += profit
            trades.append(('CLOSE BUY', df.index[i], exit_price, profit))
            print(f"Close Buy at {exit_price} on {df.index[i]} Profit: {profit:.2f}")
            position = 0

        # Exit short
        elif signal == 1 and position == -1:
            exit_price = df['close'].iloc[i]
            profit = (entry_price - exit_price) * units
            balance += profit
            trades.append(('CLOSE SELL', df.index[i], exit_price, profit))
            print(f"Close Sell at {exit_price} on {df.index[i]} Profit: {profit:.2f}")
            position = 0

    print(f"Final Balance: {balance:.2f}")
    return trades, balance

if __name__ == "__main__":
    # Assume df is your DataFrame with candle data including 'time', 'close', 'high', 'low', 'volume'
    # Make sure your 'time' column is datetime and set as index
    df = pd.read_csv("your_candle_data.csv", parse_dates=['time'], index_col='time')

    df = calculate_indicators(df)
    df = generate_signals(df)
    trades, final_balance = backtest(df)
