# ============================================================
# pipeline.py — End-to-end orchestrator
#
# Usage:
#   python pipeline.py                  # full pipeline
#   python pipeline.py --skip-download  # skip Kaggle download step
# ============================================================

import argparse
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATASET_FILE, TOP5_FEATURES,
    PLOT_LR_RESULTS, PLOT_FEATURE_IMP, PLOT_TOP5_RESULTS,
    MODEL_FULL_PATH, MODEL_TOP5_PATH,
    DECISION_THRESHOLD,
)
from data_loader   import setup_kaggle_credentials, download_dataset, load_data, explore_data
from preprocessor  import preprocess
from train_model   import train_logistic_regression, predict, save_model
from evaluate      import (
    compute_metrics, print_metrics,
    plot_confusion_and_roc, plot_feature_importance,
    predict_single_instance, build_comparison_table,
)


def run(skip_download: bool = False, single_instance_row: int = 76) -> None:
    # ── Step 1: Data acquisition ───────────────────────────
    if not skip_download:
        setup_kaggle_credentials()
        download_dataset()

    # ── Step 2: Load & explore ─────────────────────────────
    df = load_data(DATASET_FILE)
    explore_data(df)

    # ── Step 3: Preprocess (all features) ──────────────────
    print("\n[3] Preprocessing — full feature set")
    X_train_sc, X_test_sc, X_train, X_test, y_train, y_test, X, _ = preprocess(df)

    # ── Step 4: Train ──────────────────────────────────────
    print("\n[4] Training — full feature model")
    model_full = train_logistic_regression(X_train_sc, y_train)
    save_model(model_full, MODEL_FULL_PATH)

    # ── Step 5: Evaluate ───────────────────────────────────
    print("\n[5] Evaluating — full feature model")
    y_pred_full, y_proba_full = predict(model_full, X_test_sc, DECISION_THRESHOLD)

    metrics_full = compute_metrics(y_test, y_pred_full, y_proba_full)
    print_metrics(metrics_full, DECISION_THRESHOLD)

    plot_confusion_and_roc(
        y_test, y_pred_full, y_proba_full,
        title_suffix="Logistic Regression (All Features)",
        save_path=PLOT_LR_RESULTS,
    )
    coef_df = plot_feature_importance(model_full, X.columns.tolist(), save_path=PLOT_FEATURE_IMP)

    # Print top features summary
    top_risk     = coef_df[coef_df["Coefficient"] > 0].head(3)["Feature"].tolist()
    top_reducing = coef_df[coef_df["Coefficient"] < 0].head(3)["Feature"].tolist()
    print(f"\nTop 3 defect-risk features    : {top_risk}")
    print(f"Top 3 defect-reducing features : {top_reducing}")

    # ── Step 6: Single-instance prediction ─────────────────
    print(f"\n[6] Single-instance prediction (test row {single_instance_row})")
    predict_single_instance(model_full, X_test_sc, X_test, y_test, row_index=single_instance_row)

    # ── Step 7: Retrain with top 5 features ────────────────
    print("\n[7] Preprocessing & training — Top 5 features")
    X_train5_sc, X_test5_sc, _, _, y_train5, y_test5, _, _ = preprocess(df, TOP5_FEATURES)

    model_top5 = train_logistic_regression(X_train5_sc, y_train5)
    save_model(model_top5, MODEL_TOP5_PATH)

    y_pred5, y_proba5 = predict(model_top5, X_test5_sc, DECISION_THRESHOLD)
    metrics_top5 = compute_metrics(y_test5, y_pred5, y_proba5)
    print_metrics(metrics_top5, DECISION_THRESHOLD)

    plot_confusion_and_roc(
        y_test5, y_pred5, y_proba5,
        title_suffix="Logistic Regression (Top 5 Features)",
        save_path=PLOT_TOP5_RESULTS,
    )

    # ── Step 8: Comparison table ───────────────────────────
    print("\n[8] Model comparison")
    build_comparison_table(metrics_full, metrics_top5)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Software Defect Prediction Pipeline")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Kaggle dataset download (use existing CSV)"
    )
    parser.add_argument(
        "--row", type=int, default=76,
        help="Test-set row index to use for single-instance prediction"
    )
    args = parser.parse_args()
    run(skip_download=args.skip_download, single_instance_row=args.row)
