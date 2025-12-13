# src/data_preprocessing_v2.py
import pandas as pd
import os

TEST_FEATS = "data/processed/test_features_all.csv"
TEST_META  = "data/raw/test_log.csv"
OUT        = "data/processed/test_final.csv"


def run():
    os.makedirs("data/processed", exist_ok=True)

    print("🔹 Loading test features (ALL splits)...")
    feats = pd.read_csv(TEST_FEATS)
    feats["object_id"] = feats["object_id"].astype(str).str.strip()

    print(f"👉 Feature rows = {len(feats)}")

    print("🔹 Loading test metadata...")
    meta = pd.read_csv(TEST_META)
    meta["object_id"] = meta["object_id"].astype(str).str.strip()
    meta = meta[["object_id", "Z", "Z_err", "EBV"]]

    print("🔹 Merging features + metadata...")
    final = feats.merge(meta, on="object_id", how="left")

    n_rows = len(final)
    n_unique = final["object_id"].nunique()

    print(f"🔍 Rows = {n_rows} | Unique object_id = {n_unique}")

    if n_rows != n_unique:
        raise ValueError("❌ Duplicate object_id detected after merge!")

    final.to_csv(OUT, index=False)
    print(f"✅ Saved → {OUT}")
    print("🎉 data_preprocessing_v2 COMPLETED — READY FOR SUBMISSION")


if __name__ == "__main__":
    run()
