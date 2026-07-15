import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def compute_iou(y_true, y_pred):
    """Compute Intersection over Union for binary classification."""
    intersection = (y_true.astype(bool) & y_pred.astype(bool)).sum()
    union = (y_true.astype(bool) | y_pred.astype(bool)).sum()
    return intersection / (union + 1e-7)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Mangrove", "Mangrove"],
                yticklabels=["Non-Mangrove", "Mangrove"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def plot_feature_importance(importances_df, title="Feature Importance", save_path=None):
    """Plot feature importance bar chart."""
    plt.figure(figsize=(8, 5))
    plt.barh(importances_df["feature"], importances_df["importance"])
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results, save_path=None):
    """Bar chart comparing Accuracy and IoU across models."""
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    ious = [results[m]["iou"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy")
    bars2 = ax.bar(x + width/2, ious, width, label="IoU")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Coringa Mangrove Classification")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_comparison(results):
    """Print formatted comparison table."""
    print("=" * 60)
    print("CORINGA MANGROVE CLASSIFICATION — RESULTS")
    print("=" * 60)
    for model_name, m in results.items():
        print(f"{model_name:<15} | Accuracy: {m['accuracy']:.4f} | IoU: {m['iou']:.4f}")
    print("=" * 60)
