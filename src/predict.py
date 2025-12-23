"""Inference script for MALLORN experiments.

Usage example:
  python src/predict.py --base-path data/raw --model models/exp07_model.joblib --out submissions/submission_exp07.csv --threshold 0.45
  # Or use manifest to auto-load threshold:
  python src/predict.py --base-path data/raw --manifest experiments/exp07/manifest.json --out submissions/submission_exp07.csv
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd

from src.data_processing import load_all_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', required=True, help='Base path to dataset')
    parser.add_argument('--model', help='Path to saved model joblib')
    parser.add_argument('--manifest', help='Path to experiment manifest.json (to auto-load model and threshold)')
    parser.add_argument('--out', required=True, help='Output CSV path for submission')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold (overrides manifest)')
    args = parser.parse_args(argv)

    base = args.base_path
    out_path = args.out
    
    threshold = args.threshold
    model_path = args.model

    # Load from manifest if provided
    if args.manifest:
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
        
        # If model path not explicitly provided, try to find it via manifest
        if not model_path:
            # manifest['model_path'] might be relative or absolute. 
            # If it's relative, it's likely relative to project root or experiments dir.
            # We assume project root for simplicty or checking existence.
            cand = manifest.get('model_path')
            if cand and os.path.exists(cand):
                model_path = cand
            else:
                 # fallback/warning
                 print(f"Warning: Could not resolve model_path from manifest: {cand}")

        # If threshold not explicitly provided, use best_threshold
        if threshold is None:
            threshold = manifest.get('best_threshold')
            if threshold is not None:
                print(f"Using threshold from manifest: {threshold}")

    if not model_path:
        raise ValueError("Model path must be provided via --model or --manifest")

    if threshold is None:
        threshold = 0.5
        print("Using default threshold: 0.5")

    print('Loading test features...')
    test_feats = load_all_splits(base, mode='test')
    if test_feats.empty:
        raise SystemExit(f'No test features found under {base}')

    test_log_path = os.path.join(base, 'test_log.csv')
    test_log = pd.read_csv(test_log_path)

    full_test = test_log.merge(test_feats, on='object_id', how='left')
    full_test.fillna(0, inplace=True)


    # Ensure feature alignment
    # If manifest provides feature_cols, we MUST use them in that exact order.
    manifest_feats = manifest.get('feature_cols') if args.manifest and 'manifest' in locals() else None
    
    if manifest_feats:
        print(f"Aligning to {len(manifest_feats)} features from manifest...")
        # Add missing columns
        missing_cols = set(manifest_feats) - set(full_test.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features missing in test data, filling with 0.")
            for c in missing_cols:
                full_test[c] = 0.0
        
        # Select and reorder
        X_test = full_test[manifest_feats]
    else:
        # Fallback to local logic (risky if order differs)
        drop_cols = ['object_id', 'SpecType', 'English Translation', 'split', 'target', 'Z_err']
        feature_cols = [c for c in full_test.columns if c not in drop_cols]
        X_test = full_test[feature_cols]

    print(f'Loading model: {model_path}')
    model = joblib.load(model_path)

    print(f'Predicting with threshold {threshold}...')
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        print("Model does not support predict_proba, using predict (ignoring threshold)...")
        preds = model.predict(X_test)

    submission = pd.DataFrame({'object_id': full_test['object_id'], 'prediction': preds})
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    submission.to_csv(out_path, index=False)
    print('Wrote submission to', out_path)
    print(submission['prediction'].value_counts())


if __name__ == '__main__':
    main()
