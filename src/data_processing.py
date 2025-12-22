"""Data processing helpers for MALLORN project.

Provides:
- `extract_features_from_split(csv_path)` -> DataFrame of per-object features
- `load_all_splits(base_path, mode='train')` -> concat features from split_01..split_20

This file is a direct, minimal refactor of the feature extraction cells in the canonical
notebook. It is intentionally dependency-light to make unit testing straightforward.
"""
from __future__ import annotations

import os
from typing import Optional

import gc
import numpy as np
import pandas as pd
from scipy.stats import skew

def extract_features_from_split(csv_path: str) -> Optional[pd.DataFrame]:
    """Read a lightcurve CSV and return per-object aggregated features.

    Returns None if the file does not exist or is empty.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return None

    # Signal-to-noise ratio
    df['snr'] = df['Flux'] / (df.get('Flux_err', 0) + 1e-6)

    # Group by object and filter, compute statistics
    aggs = df.groupby(['object_id', 'Filter']).agg(
        Flux_max=pd.NamedAgg(column='Flux', aggfunc='max'),
        Flux_min=pd.NamedAgg(column='Flux', aggfunc='min'),
        Flux_mean=pd.NamedAgg(column='Flux', aggfunc='mean'),
        Flux_std=pd.NamedAgg(column='Flux', aggfunc='std'),
        Flux_skew=pd.NamedAgg(column='Flux', aggfunc=skew),
        snr_max=pd.NamedAgg(column='snr', aggfunc='max'),
        snr_mean=pd.NamedAgg(column='snr', aggfunc='mean'),
    )

    # Flatten multiindex columns: currently simple names from NamedAgg
    aggs = aggs.unstack(level=1)
    # After unstack the columns become multiindex like ('Flux_max','g'), rebuild names
    aggs.columns = [f"{stat}_{flt}" for stat, flt in aggs.columns]

    # Color features (example: g - r)
    if 'Flux_max_g' in aggs.columns and 'Flux_max_r' in aggs.columns:
        aggs['color_g_r'] = aggs['Flux_max_g'] - aggs['Flux_max_r']

    # Amplitude per filter
    # detect filters by parsing column suffixes
    filters = set()
    for col in aggs.columns:
        if '_' in col:
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                filters.add(parts[1])

    for f in filters:
        max_col = f'Flux_max_{f}'
        min_col = f'Flux_min_{f}'
        amp_col = f'amp_{f}'
        if max_col in aggs.columns and min_col in aggs.columns:
            aggs[amp_col] = aggs[max_col] - aggs[min_col]

    # Number of observations per object
    counts = df.groupby('object_id').size().to_frame('n_obs')

    features = aggs.merge(counts, left_index=True, right_index=True)
    return features


def load_all_splits(base_path: str, mode: str = 'train') -> pd.DataFrame:
    """Load features from split_01..split_20 under `base_path`.

    Example file layout:
      <base_path>/split_01/train_full_lightcurves.csv
      <base_path>/split_02/train_full_lightcurves.csv

    Returns a concatenated DataFrame indexed by `object_id`.
    """
    all_features = []
    for i in range(1, 21):
        split_name = f'split_{i:02d}'
        file_name = f'{mode}_full_lightcurves.csv'
        full_path = os.path.join(base_path, split_name, file_name)
        feats = extract_features_from_split(full_path)
        if feats is not None:
            all_features.append(feats)
        # free memory
        del feats
        gc.collect()

    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features)


__all__ = ['extract_features_from_split', 'load_all_splits']
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
