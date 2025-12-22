Repository structure and conventions

Purpose: keep experiments, notebooks, and scripts organized so new notebooks can be added consistently.

Top-level layout

- `notebooks/` — exploratory and report notebooks.
  - Naming convention: `YYYYMMDD_exp-short-desc.ipynb` or `expNN_short-desc.ipynb`.
  - First markdown cell MUST include metadata: `Experiment ID`, `Author`, `Date`, `Dataset base_path`, `Short description`, `Related scripts`.
  - Prefer using `notebooks/template.ipynb` as a starting point.

- `src/` — reusable scripts and CLI utilities.
  - `data_processing.py` — extract features from `data/raw/split_*/*.csv` and write processed files to `data/processed/`.
  - `model_training.py` — training, tuning, and producing submission CSVs.
  - `utils.py` — helpers (save/load model, metrics, reproducibility helpers).

- `data/raw/` — raw dataset files (split_01..split_20 + train_log.csv, test_log.csv).
- `data/processed/` — cached features (parquet/csv) to avoid reprocessing. Commit only small examples, not full datasets.

- `models/` — saved models (use `joblib` or `pickle`). Use names like `expID_model.joblib` and include a `metadata.json` alongside with params and metrics.
- `experiments/` — one folder per experiment: `experiments/expID/manifest.json`, `metrics.json`, figures, and final submission CSV.
- `reports/` — exported notebooks (HTML), plots, and short writeups.

Experiment manifest (suggested keys) — `experiments/expID/manifest.json`
{
  "id": "exp_20251221_01",
  "date": "2025-12-21",
  "author": "Name",
  "notebook": "notebooks/20251221_exp-desc.ipynb",
  "script": "src/model_training.py",
  "params": {"model":"SVM","C":10,"gamma":"scale"},
  "metrics": {"val_f1":0.42},
  "notes": "Short notes about dataset used and changes"
}

Guidelines

- Put heavy, reproducible code in `src/` and call from notebooks. Notebooks should document reasoning, show key plots, and link to `experiments/expID`.
- Keep `data/raw/` out of git for large files — use `.gitignore` and add small example splits if needed for CI and demos.
- When adding a new notebook, create an `experiments/expID/` folder and add `manifest.json` describing the experiment and results.
- Update `README.md` with short summary and link to notable experiments.
