import numpy as np
import geemap
from configs.constants import (
    S2_PATCH_SIZE, S2_SCALE, L8_PATCH_SIZE, L8_SCALE,
    N_PATCHES, S2_FEATURES, L8_FEATURES
)
from src.data.loader import init_ee, get_features_image, load_worldcover
from src.data.patches import extract_patches, patches_to_pixels
from src.models.random_forest import run_rf_pipeline
from src.models.xgboost_model import run_xgb_pipeline
from src.models.deeplabv3 import train_dl
from src.evaluate import print_comparison, plot_comparison


def download_data(sensor="s2"):
    """Download features + labels from GEE to numpy.

    Args:
        sensor: 's2' or 'l8'

    Returns:
        features_array: (H, W, C) numpy array
        labels_array: (H, W, 1) numpy array
    """
    init_ee()

    features_img = get_features_image(sensor)
    labels_img = load_worldcover()

    # Combine into single image for download
    combined = features_img.addBands(labels_img)
    region = get_region()

    scale = S2_SCALE if sensor == "s2" else L8_SCALE
    print(f"Downloading {sensor.upper()} image at {scale}m scale...")
    combined_array = geemap.ee_to_numpy(combined, region=region, scale=scale)
    print(f"Combined shape: {combined_array.shape}")

    n_channels = len(S2_FEATURES) if sensor == "s2" else len(L8_FEATURES)
    features_array = combined_array[:, :, :n_channels]
    labels_array = combined_array[:, :, n_channels:]

    print(f"Features: {features_array.shape}, Labels: {labels_array.shape}")
    print(f"Mangrove ratio: {(labels_array == 1).mean():.4f}")

    return features_array, labels_array


def run_ml_comparison(sensor="s2"):
    """Run RF + XGBoost comparison on pixel-level data.

    Flattens patches to pixels, subsamples for speed.
    """
    features_array, labels_array = download_data(sensor)

    patch_size = S2_PATCH_SIZE if sensor == "s2" else L8_PATCH_SIZE
    feature_names = S2_FEATURES if sensor == "s2" else L8_FEATURES

    # Extract patches
    X, y = extract_patches(features_array, labels_array, patch_size, N_PATCHES)
    print(f"Patches: X={X.shape}, y={y.shape}")

    # Flatten to pixels
    X_pixels = patches_to_pixels(X)
    y_pixels = y.reshape(-1)

    # Subsample for speed
    sub_idx = np.random.choice(len(X_pixels), min(100_000, len(X_pixels)), replace=False)
    X_sub = X_pixels[sub_idx]
    y_sub = y_pixels[sub_idx]

    # Run RF
    print("\n--- Random Forest ---")
    rf_metrics = run_rf_pipeline(X_sub, y_sub, feature_names)
    print(f"RF Accuracy: {rf_metrics['accuracy']:.4f} | IoU: {rf_metrics['iou']:.4f}")

    # Run XGBoost
    print("\n--- XGBoost ---")
    xgb_metrics = run_xgb_pipeline(X_sub, y_sub, feature_names)
    print(f"XGB Accuracy: {xgb_metrics['accuracy']:.4f} | IoU: {xgb_metrics['iou']:.4f}")

    return {"RF+CT": rf_metrics, "XGBoost": xgb_metrics}


def run_dl_training(sensor="s2"):
    """Run DeepLabV3+ training on patch-level data."""
    features_array, labels_array = download_data(sensor)

    patch_size = S2_PATCH_SIZE if sensor == "s2" else L8_PATCH_SIZE
    in_channels = len(S2_FEATURES) if sensor == "s2" else len(L8_FEATURES)

    # Extract patches
    X, y = extract_patches(features_array, labels_array, patch_size, N_PATCHES)
    print(f"Patches: X={X.shape}, y={y.shape}")

    save_path = f"best_model_{sensor}.pt"
    model, dl_metrics = train_dl(X, y, in_channels, save_path)
    print(f"DL Accuracy: {dl_metrics['accuracy']:.4f} | IoU: {dl_metrics['iou']:.4f}")

    return {"DeepLabV3+": dl_metrics}


def run_full_comparison(sensor="s2"):
    """Run all 3 models and print comparison table."""
    print(f"\n{'='*60}")
    print(f"  CORINGA MANGROVE — {sensor.upper()} — FULL COMPARISON")
    print(f"{'='*60}\n")

    # ML models (pixel-level)
    ml_results = run_ml_comparison(sensor)

    # DL model (patch-level)
    dl_results = run_dl_training(sensor)

    # Combine and compare
    all_results = {**ml_results, **dl_results}
    print_comparison(all_results)
    plot_comparison(all_results, f"results/figures/model_comparison_{sensor}.png")

    return all_results
