# Software Defect Prediction — Modular Structure

```
.
├── config.py        # All constants & hyperparameters (single source of truth)
├── data_loader.py   # Kaggle setup, CSV loading, dataset exploration
├── preprocessor.py  # Encoding, train/test split, StandardScaler
├── train_model.py   # LogisticRegression training, predict(), save/load model
├── evaluate.py      # Metrics, confusion matrix, ROC curve, feature importance, comparison table
└── pipeline.py      # End-to-end orchestrator — imports and calls all modules
```

## Quick start

```bash
# Full run (downloads data from Kaggle first)
python pipeline.py

# Skip download if the CSV already exists locally
python pipeline.py --skip-download

# Change the single-instance prediction row
python pipeline.py --skip-download --row 42
```

## Module responsibilities

| File | Responsibility |
|------|---------------|
| `config.py` | Dataset path, model hyperparameters, feature lists, output paths |
| `data_loader.py` | `setup_kaggle_credentials()`, `download_dataset()`, `load_data()`, `explore_data()` |
| `preprocessor.py` | `preprocess(df, feature_subset?)` → scaled arrays + scaler |
| `train_model.py` | `train_logistic_regression()`, `predict()`, `save_model()`, `load_model()` |
| `evaluate.py` | `compute_metrics()`, `print_metrics()`, `plot_confusion_and_roc()`, `plot_feature_importance()`, `predict_single_instance()`, `build_comparison_table()` |
| `pipeline.py` | Calls modules in order; CLI entry point |

## Extending the project

- **New model** (e.g. Random Forest): add a `train_random_forest()` function to `train_model.py` and call it from `pipeline.py`.
- **New features**: update `TOP5_FEATURES` or add a new list in `config.py`.
- **Different threshold**: change `DECISION_THRESHOLD` in `config.py` — all modules read from there.
