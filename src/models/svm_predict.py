import pandas as pd
import numpy as np
import joblib
import os

def load_test_data(path="data/processed/test_final.csv"):
    return pd.read_csv(path)

def load_artifacts(model_dir="models"):
    model = joblib.load(f"{model_dir}/svm_model.pkl")
    scaler = joblib.load(f"{model_dir}/svm_scaler.pkl")
    features = joblib.load(f"{model_dir}/svm_features.pkl")
    return model, scaler, features

def preprocess_test(df, feature_cols):
    # Giữ đúng thứ tự cột đã dùng khi train
    X = df[feature_cols]

    # Fill NaN nếu có (test thường có missing)
    X = X.fillna(X.mean())

    return X

def run():
    print("🔹 Loading artifacts...")
    model, scaler, features = load_artifacts()

    print("🔹 Loading TEST data...")
    df_test = load_test_data()

    print("🔹 Preprocessing test...")
    X_test = preprocess_test(df_test, features)
    X_scaled = scaler.transform(X_test)

    print("🔹 Predicting...")
    preds = model.predict(X_scaled)

    print("🔹 Saving submission.csv...")
    sub = pd.DataFrame({
        "object_id": df_test["object_id"],
        "target": preds
    })

    os.makedirs("submissions", exist_ok=True)
    sub.to_csv("submissions/svm_submission.csv", index=False)

    print("🎉 DONE! File saved at submissions/svm_submission.csv")

if __name__ == "__main__":
    run()
