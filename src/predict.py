"""Inference script for MALLORN experiments.

Usage example:
  python src/predict.py --base-path data/raw --model models/exp07_model.joblib --out submissions/submission_exp07.csv
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd

from src.data_processing import load_all_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', required=True, help='Base path to dataset')
    parser.add_argument('--model', required=True, help='Path to saved model joblib')
    parser.add_argument('--out', required=True, help='Output CSV path for submission')
    args = parser.parse_args(argv)

    base = args.base_path
    model_path = args.model
    out_path = args.out

    print('Loading test features...')
    test_feats = load_all_splits(base, mode='test')
    if test_feats.empty:
        raise SystemExit(f'No test features found under {base}')

    test_log_path = os.path.join(base, 'test_log.csv')
    test_log = pd.read_csv(test_log_path)

    full_test = test_log.merge(test_feats, on='object_id', how='left')
    full_test.fillna(0, inplace=True)

    drop_cols = ['object_id', 'SpecType', 'English Translation', 'split', 'target', 'Z_err']
    feature_cols = [c for c in full_test.columns if c not in drop_cols]

    X_test = full_test[feature_cols]

    print('Loading model:', model_path)
    model = joblib.load(model_path)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X_test)

    submission = pd.DataFrame({'object_id': full_test['object_id'], 'prediction': preds})
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    submission.to_csv(out_path, index=False)
    print('Wrote submission to', out_path)


if __name__ == '__main__':
    main()
