import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Load training data
df = pd.read_csv("training_data.csv")

target_col = "label"  # Use the label column from your training data

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Features and target column
feature_cols = ["open", "high", "low", "close", "rsi", "macd", "macd_signal", "atr", "macd_diff", "candle_range"]
target_col = "signal"  # Assuming 'signal' is categorical: e.g. 'buy', 'sell', 'hold'

# Encode target to numeric if it's categorical strings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

X = df[feature_cols]
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model to file
model.save_model("xgb_next_candle.model")
print("[Training] Model saved as xgb_next_candle.model")
