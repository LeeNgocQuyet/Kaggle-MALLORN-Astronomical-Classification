import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os


def load_data(path="data/processed/train_final.csv"):
    return pd.read_csv(path)


def preprocess(df):

    # Z_err bị trống 100%, không thể dùng
    numeric_cols = [
        'mean_flux', 'std_flux', 'min_flux', 'max_flux',
        'mean_flux_err', 'num_points', 'duration',
        'Z', 'EBV'
    ]

    # Chỉ drop những cột thật sự cần thiết
    df = df.dropna(subset=numeric_cols + ['target'])

    print(f"👉 Rows after preprocess: {len(df)}")

    if len(df) == 0:
        raise ValueError("❌ ERROR: No rows left after preprocess. Check missing values!")

    X = df[numeric_cols]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, numeric_cols


def train_svm(X, y):
    model = SVC(kernel="rbf", C=3, gamma="scale", probability=True)
    model.fit(X, y)
    return model


def save_model(model, scaler, feature_cols, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, f"{out_dir}/svm_model.pkl")
    joblib.dump(scaler, f"{out_dir}/svm_scaler.pkl")
    joblib.dump(feature_cols, f"{out_dir}/svm_features.pkl")
    print("✓ Model saved to models/")


def run():

    print("🔹 Loading data...")
    df = load_data()

    print("🔹 Preprocessing...")
    X, y, scaler, feature_cols = preprocess(df)

    print("🔹 Training SVM...")
    model = train_svm(X, y)

    print("🔹 Saving...")
    save_model(model, scaler, feature_cols)

    print("\n🎉 DONE — SVM train complete!")


if __name__ == "__main__":
    run()
