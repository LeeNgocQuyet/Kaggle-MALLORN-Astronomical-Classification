import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_submission(df, path):
    df.to_csv(path, index=False)
