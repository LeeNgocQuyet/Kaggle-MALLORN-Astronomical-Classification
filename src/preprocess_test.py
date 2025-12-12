# src/preprocess_test.py
import pandas as pd
import os
from glob import glob

OUT = "data/processed/test_final.csv"
FEATURE_GLOB = "data/processed/test_features_*.csv"

def run():
    parts = sorted(glob(FEATURE_GLOB))
    if not parts:
        combined = "data/processed/test_features_all.csv"
        if os.path.exists(combined):
            parts = [combined]
        else:
            raise FileNotFoundError("No test feature files found. Run data_preprocessing first.")
    feats = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    # normalize object_id
    feats['object_id'] = feats['object_id'].astype(str).str.strip()
    os.makedirs("data/processed", exist_ok=True)
    feats.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(feats)} rows)")
    return OUT

if __name__ == "__main__":
    run()
