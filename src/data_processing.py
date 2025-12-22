"""
Feature extraction helpers for MALLORN project.
"""
import os
import pandas as pd
import gc


def extract_features_from_split(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    aggs = df.groupby(['object_id', 'Filter'])['Flux'].agg(['max', 'min', 'mean', 'std']).unstack()
    aggs.columns = [f'{stat}_{filt}' for stat, filt in aggs.columns]

    filters = df['Filter'].unique()
    for f in filters:
        if f'max_{f}' in aggs.columns and f'min_{f}' in aggs.columns:
            aggs[f'amp_{f}'] = aggs[f'max_{f}'] - aggs[f'min_{f}']

    counts = df.groupby('object_id').size().to_frame('n_obs')
    features = aggs.merge(counts, left_index=True, right_index=True)
    return features


def load_all_splits(base_path, mode='train'):
    all_features = []
    for i in range(1, 21):
        split_name = f'split_{i:02d}'
        file_name = f'{mode}_full_lightcurves.csv'
        full_path = os.path.join(base_path, split_name, file_name)
        feats = extract_features_from_split(full_path)
        if feats is not None:
            all_features.append(feats)
        del feats
        gc.collect()

    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features)
