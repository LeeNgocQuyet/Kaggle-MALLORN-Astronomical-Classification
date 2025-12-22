import os
import tempfile
import pandas as pd

from src.data_processing import extract_features_from_split


def test_extract_features_from_split_creates_expected_columns():
    # create a small temp CSV mimicking two objects and two filters
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, 'sample.csv')
    df = pd.DataFrame([
        {'object_id': 1, 'Filter': 'g', 'Flux': 10.0, 'Flux_err': 1.0},
        {'object_id': 1, 'Filter': 'g', 'Flux': 12.0, 'Flux_err': 1.2},
        {'object_id': 1, 'Filter': 'r', 'Flux': 8.0,  'Flux_err': 0.9},
        {'object_id': 2, 'Filter': 'g', 'Flux': 3.0,  'Flux_err': 0.5},
    ])
    df.to_csv(csv_path, index=False)

    feats = extract_features_from_split(csv_path)
    assert feats is not None
    # index should contain object_id values
    assert 1 in feats.index
    assert 2 in feats.index
    # expected columns: n_obs and at least one Flux_max_{filter}
    assert 'n_obs' in feats.columns
    # At least one per-filter stat should exist for filter 'g'
    assert any(col.endswith('_g') for col in feats.columns)
