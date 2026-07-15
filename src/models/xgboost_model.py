import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from configs.constants import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE


def train_xgb(X_train, y_train):
    """Train XGBoost classifier.

    Params: 200 trees, depth 6, lr 0.1.
    Works on pixel-level data (flattened patches).

    Returns:
        fitted XGBClassifier
    """
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    return xgb


def evaluate_xgb(model, X_test, y_test, feature_names=None):
    """Evaluate XGBoost model and return metrics dict.

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


def run_xgb_pipeline(X, y, feature_names=None):
    """Full XGBoost pipeline: split, train, evaluate.

    Args:
        X: (N_pixels, C) feature array
        y: (N_pixels,) label array
        feature_names: list of band names

    Returns:
        metrics dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = train_xgb(X_train, y_train)
    return evaluate_xgb(model, X_test, y_test, feature_names)
