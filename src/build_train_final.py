# src/build_train_final.py
import pandas as pd
import os

RAW_LOG = "data/raw/train_log.csv"
OUT = "data/processed/train_final.csv"
FEATURE_GLOB = "data/processed/train_features_*.csv"  # matches per-split files or train_features_all.csv

def run(use_combined_if_exists=True):
    # load all feature parts if exist
    parts = sorted([p for p in __import__("glob").glob(FEATURE_GLOB)])
    if not parts:
        # try the combined file name
        combined = "data/processed/train_features_all.csv"
        if os.path.exists(combined):
            parts = [combined]
        else:
            raise FileNotFoundError("No train feature files found. Run data_preprocessing first.")
    feats = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    meta = pd.read_csv(RAW_LOG)
    # ensure types and strip whitespace
    feats['object_id'] = feats['object_id'].astype(str).str.strip()
    meta['object_id'] = meta['object_id'].astype(str).str.strip()

    # merge features with target only (do not import noisy text columns as model inputs)
    meta_small = meta[['object_id','target']]
    df = meta_small.merge(feats, on='object_id', how='left')

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(df)} rows, {df['target'].value_counts().to_dict()})")
    return OUT

if __name__ == "__main__":
    run()
