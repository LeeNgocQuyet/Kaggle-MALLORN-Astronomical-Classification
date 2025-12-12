import pandas as pd
import os

# ---- 1. LOAD RAW ----
def load_lightcurves():
    path = "data/raw/split_01/test_full_lightcurves.csv"
    return pd.read_csv(path)

def load_meta():
    path = "data/raw/test_log.csv"
    return pd.read_csv(path)


# ---- 2. FEATURE EXTRACTION (same as train) ----
def extract_features(df):
    grouped = df.groupby("object_id")

    features = grouped.agg(
        mean_flux=("Flux", "mean"),
        std_flux=("Flux", "std"),
        min_flux=("Flux", "min"),
        max_flux=("Flux", "max"),
        mean_flux_err=("Flux_err", "mean"),
        num_points=("Flux", "count"),
        duration=("Time (MJD)", lambda x: x.max() - x.min())
    ).reset_index()

    return features


# ---- 3. MERGE META + FEATURES ----
def merge(features, meta):
    # chỉ lấy các cột mà SVM dùng
    selected_meta = meta[["object_id", "Z", "Z_err", "EBV"]]

    df = features.merge(selected_meta, on="object_id", how="left")
    return df


# ---- 4. RUN ----
def run():
    os.makedirs("data/processed", exist_ok=True)

    print("🔹 Loading TEST lightcurves...")
    df_lc = load_lightcurves()

    print("🔹 Extracting lightcurve features...")
    features = extract_features(df_lc)

    print("🔹 Loading TEST metadata...")
    meta = load_meta()

    print("🔹 Merging features + metadata...")
    final = merge(features, meta)

    print("🔹 Saving → data/processed/test_final.csv")
    final.to_csv("data/processed/test_final.csv", index=False)

    print("✅ preprocess_test completed!")


if __name__ == "__main__":
    run()
