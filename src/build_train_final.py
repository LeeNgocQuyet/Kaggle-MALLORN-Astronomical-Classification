import pandas as pd
import os

def run():

    print("🔹 Loading logs & features...")
    log = pd.read_csv("data/raw/train_log.csv")
    feats = pd.read_csv("data/processed/train_features.csv")

    print("🔹 Merging files...")
    df = log.merge(feats, on="object_id", how="left")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/train_final.csv", index=False)

    print("✅ Saved: data/processed/train_final.csv")

if __name__ == "__main__":
    run()
