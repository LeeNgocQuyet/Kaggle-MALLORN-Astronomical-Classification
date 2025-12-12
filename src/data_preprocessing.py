import pandas as pd
import numpy as np
import os

def extract_features(df):
    grouped = df.groupby("object_id")

    features = grouped["Flux"].agg(
        mean_flux="mean",
        std_flux="std",
        min_flux="min",
        max_flux="max"
    )

    features["mean_flux_err"] = grouped["Flux_err"].mean()
    features["num_points"] = grouped.size()
    features["duration"] = grouped["Time (MJD)"].max() - grouped["Time (MJD)"].min()

    return features.reset_index()

def run():

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/raw/split_01/train_full_lightcurves.csv")
    print("🔹 Extracting features...")

    features = extract_features(df)

    print("🔹 Saving to data/processed/train_features.csv")
    features.to_csv("data/processed/train_features.csv", index=False)

    print("✅ Done!")

if __name__ == "__main__":
    run()
