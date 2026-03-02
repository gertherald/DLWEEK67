# ============================================================
# train_model.py — Model training and serialisation
# ============================================================

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from config import SOLVER, MAX_ITER, CLASS_WEIGHT, RANDOM_STATE, DECISION_THRESHOLD


def train_logistic_regression(
    X_train_scaled: np.ndarray,
    y_train,
) -> LogisticRegression:
    """
    Fit a Logistic Regression model on the provided (already-scaled) training data.

    Returns the fitted model.
    """
    print("Training Logistic Regression …")
    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        solver=SOLVER,
        class_weight=CLASS_WEIGHT,
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Training complete.")
    return model


def predict(
    model: LogisticRegression,
    X_scaled: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (binary predictions, defect probabilities) using a custom threshold.
    """
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred  = (y_proba > threshold).astype(int)
    return y_pred, y_proba


def save_model(model: LogisticRegression, path: str = "lr_model.pkl") -> None:
    """Serialise the trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to '{path}'")


def load_model(path: str = "lr_model.pkl") -> LogisticRegression:
    """Load a previously serialised model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from '{path}'")
    return model
