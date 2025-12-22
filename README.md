# MALLORN Astronomical Classification

Project scaffold for the MALLORN Astronomical Classification challenge. This repository contains a cleaned notebook, reproducible scripts, and minimal documentation to run the baseline SVM pipeline described in the provided notebook.

## Overview
- Extract per-object per-filter summary statistics (max, min, mean, std, amplitude) from raw lightcurve CSVs (split_01..split_20).
- Merge these features with the event `*_log.csv` metadata.
- Preprocess: scaling (StandardScaler) and class balancing with SMOTE.
- Train an SVM (RBF) with GridSearchCV and produce submission CSV.

## Files added
- `notebooks/MALLORN_classification.ipynb` — cleaned notebook with outputs reset.
- `src/data_processing.py` — functions to extract features and load splits.
- `src/model_training.py` — end-to-end training and prediction script.
- `src/utils.py` — small helpers for saving/loading models and submissions.
- `requirements.txt` — Python dependencies.
- `.gitignore`

## Quickstart
1. Create environment and install dependencies:

   pip install -r requirements.txt

2. Edit `--base-path` when running scripts if your data is not in `data/raw`.

3. Run training and prediction (examples):

   - Run training (example CLI):

     python src/train.py --base-path data/raw --exp exp07 --out-dir experiments/exp07

   - Run inference / produce submission (example):

     python src/predict.py --base-path data/raw --model models/exp07_model.joblib --out submissions/submission_exp07.csv

Notes:
- Default dataset path used in notebooks: `/kaggle/input/mallorn-dataset`. If you work locally, place raw CSVs under `data/raw/` with the same split_X folders or set `--base-path` accordingly.
- If you run feature extraction for the first time, it's recommended to cache processed features into `data/processed/` to avoid reprocessing the 20 splits.

Reproducing the notebook results:
- Use `notebooks/20251223_exp07_svm-final-v6.ipynb` for the canonical experiment narrative. For reproducible runs, prefer the `src/` scripts above.


## Notes
- The notebook is intended for reporting and exploration; prefer running `src/*` scripts for reproducibility and CI.
- After experiments, update this README with final metrics and a link to the model and submission.

## Repository structure

- `notebooks/` : exploratory and report notebooks. Naming convention: `YYYYMMDD_exp-short-desc.ipynb` or `expNN_short-desc.ipynb`. Start new notebooks from `notebooks/template.ipynb` and include required metadata in the first markdown cell (Experiment ID, author, date, dataset base_path, brief description, related scripts).
- `src/` : reusable scripts and CLI tools (`data_processing.py`, `model_training.py`, `utils.py`). Keep heavy processing in scripts and call them from notebooks with `!python`.
- `data/raw/` : raw CSVs (split_01..split_20) and `train_log.csv` / `test_log.csv`.
- `data/processed/` : cached features (parquet or csv) to speed runs; avoid committing large processed files.
- `models/` : serialized models (name them `expID_model.joblib`) and companion metadata files describing params/metrics.
- `experiments/` : one folder per experiment with `manifest.json`, `metrics.json`, plots and final `submission.csv`.
- `reports/` : exported notebooks (HTML), figures and writeups.

Guidelines:

- Use `notebooks/template.ipynb` for new notebooks and include a short metadata block in the first markdown cell.
- For long-running steps (feature extraction, hyperparameter search) implement and run via `src/` scripts; keep notebooks for explanation and plots.
- Record experiment metadata in `experiments/expID/manifest.json` with keys `id`, `date`, `author`, `notebook`, `params`, `metrics`, `notes`.
- When adding notebooks or changing conventions, update `docs/STRUCTURE.md` accordingly.
