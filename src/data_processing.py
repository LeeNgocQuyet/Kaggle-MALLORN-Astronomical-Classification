"""Data processing helpers for MALLORN project."""
from __future__ import annotations

import os
from typing import Optional

import gc
import numpy as np
import pandas as pd
from scipy.stats import skew

def extract_features_from_split(csv_path: str) -> Optional[pd.DataFrame]:
    """Read a lightcurve CSV and return per-object aggregated features.
    
    Logic aligns with User's 'Core Engine':
    - Aggregates: max, min, mean, std, skew per Filter.
    - Colors: g-r, u-z.
    - Amplitude: max - min per filter.
    - n_obs: count.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return None

    # 1. Basic Stats + Skew
    aggs = df.groupby(['object_id', 'Filter'])['Flux'].agg(['max', 'min', 'mean', 'std', skew]).unstack()
    aggs.columns = [f'{stat}_{filt}' for stat, filt in aggs.columns]

    # 2. Colors
    # g - r
    if 'max_g' in aggs.columns and 'max_r' in aggs.columns:
        aggs['color_g_r'] = aggs['max_g'] - aggs['max_r']
    # u - z
    if 'max_u' in aggs.columns and 'max_z' in aggs.columns:
        aggs['color_u_z'] = aggs['max_u'] - aggs['max_z']

    # 3. Amplitude
    # Filters in this dataset: u, g, r, i, z, y
    filters = df['Filter'].unique()
    for f in filters:
        if f'max_{f}' in aggs.columns and f'min_{f}' in aggs.columns:
            aggs[f'amp_{f}'] = aggs[f'max_{f}'] - aggs[f'min_{f}']

    # 4. n_obs
    counts = df.groupby('object_id').size().to_frame('n_obs')
    features = aggs.merge(counts, left_index=True, right_index=True)
    
    return features

def load_all_splits(base_path: str, mode: str = 'train') -> pd.DataFrame:
    """Load features from split_01..split_20."""
    all_features = []
    print(f"Propcessing {mode} data from 20 splits...")
    
    for i in range(1, 21):
        split_name = f'split_{i:02d}'
        file_name = f'{mode}_full_lightcurves.csv'
        full_path = os.path.join(base_path, split_name, file_name)
        
        # print(f"Processing {split_name}...", end='\r')
        feats = extract_features_from_split(full_path)
        if feats is not None:
            all_features.append(feats)
        
        del feats
        gc.collect()
        
    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features)
