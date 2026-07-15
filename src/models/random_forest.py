import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from configs.constants import RF_N_ESTIMATORS, RF_MAX_FEATURES


def train_rf(X_train, y_train):
    """Train Random Forest classifier.

    Paper params: 100 trees, 5 variables per split.
    Works on pixel-level data (flattened patches).

    Returns:
        fitted RandomForestClassifier
    """
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_rf(model, X_test, y_test, feature_names=None):
    """Evaluate RF model and return metrics dict.

    Returns:
        dict with accuracy, iou, classification_report, feature_importances
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # IoU
    intersection = (y_test.astype(bool) & y_pred.astype(bool)).sum()
    union = (y_test.astype(bool) | y_pred.astype(bool)).sum()
    iou = intersection / (union + 1e-7)

    # Feature importance
    importances = None
    if feature_names is not None:
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

    report = classification_report(
        y_test, y_pred,
        target_names=["Non-Mangrove", "Mangrove"],
        output_dict=True
    )

    return {
        "accuracy": float(acc),
        "iou": float(iou),
        "report": report,
        "feature_importances": importances,
        "y_pred": y_pred
    }


def run_rf_pipeline(X, y, feature_names=None):
    """Full RF pipeline: split, train, evaluate.

    Args:
        X: (N_pixels, C) feature array (flattened from patches)
        y: (N_pixels,) label array
        feature_names: list of band names for importance

    Returns:
        metrics dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = train_rf(X_train, y_train)
    return evaluate_rf(model, X_test, y_test, feature_names)
