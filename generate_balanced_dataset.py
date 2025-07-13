# === generate_balanced_dataset.py ===
import pandas as pd

try:
    # Load original dataset
    df = pd.read_csv("training_data.csv")

    # Check for 'signal' column
    if 'signal' not in df.columns:
        raise ValueError("Missing 'signal' column in training_data.csv")

    # Split by class
    buy_df = df[df["signal"] == "buy"]
    sell_df = df[df["signal"] == "sell"]

    # Match the size of the smaller class
    min_len = min(len(buy_df), len(sell_df))
    buy_df_balanced = buy_df.sample(n=min_len, random_state=42)
    sell_df_balanced = sell_df.sample(n=min_len, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([buy_df_balanced, sell_df_balanced]).sample(frac=1, random_state=42)

    # Save to new file
    balanced_df.to_csv("training_data_balanced.csv", index=False)
    print(f"[✅] Balanced dataset saved as 'training_data_balanced.csv' with {len(balanced_df)} rows.")

except FileNotFoundError:
    print("[❌] File 'training_data.csv' not found.")
except Exception as e:
    print(f"[❌] Error: {e}")
