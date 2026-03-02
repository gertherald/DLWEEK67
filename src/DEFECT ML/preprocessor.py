# ============================================================
# preprocessor.py — Feature engineering, splitting, and scaling
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def preprocess(
    df: pd.DataFrame,
    feature_subset: list[str] | None = None,
) -> tuple:
    """
    Separate features/target, encode categoricals, split, and scale.

    Parameters
    ----------
    df             : Raw DataFrame (must contain TARGET_COLUMN).
    feature_subset : Optional list of feature names to use. When None, all
                     columns except TARGET_COLUMN are used.

    Returns
    -------
    X_train_scaled, X_test_scaled, X_train, X_test,
    y_train, y_test, X (unscaled full feature frame), scaler
    """
    # ── Feature / target separation ────────────────────────
    if feature_subset:
        X = df[feature_subset].copy()
    else:
        X = df.drop(TARGET_COLUMN, axis=1).copy()

    y = df[TARGET_COLUMN]

    # ── Encode any categorical columns ─────────────────────
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"Encoding categorical columns: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    else:
        print("No categorical columns — all features are numeric.")

    # ── Train / test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train : {X_train.shape[0]:,} rows | Test : {X_test.shape[0]:,} rows")
    print(f"Features used : {X_train.shape[1]}")

    # ── Standard scaling ───────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, X, scaler
