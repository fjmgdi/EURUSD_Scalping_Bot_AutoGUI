import xgboost as xgb
import numpy as np

def create_dummy_model():
    # Create random training data (100 samples, 10 features)
    X = np.random.rand(100, 10)
    # Random binary labels (0 or 1)
    y = np.random.randint(0, 2, 100)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train the model
    model.fit(X, y)

    # Save the model to file
    model.save_model("xgb_next_candle.model")
    print("Dummy XGBoost model saved as 'xgb_next_candle.model'.")

if __name__ == "__main__":
    create_dummy_model()
