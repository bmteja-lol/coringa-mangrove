#!/usr/bin/env python3
"""CLI entry point for mangrove classification pipelines.

Usage:
    python scripts/run_all.py --sensor s2 --method all
    python scripts/run_all.py --sensor l8 --method rf
    python scripts/run_all.py --sensor s2 --method dl
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import run_ml_comparison, run_dl_training, run_full_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Coringa Mangrove Classification — ML Comparison"
    )
    parser.add_argument(
        "--sensor", choices=["s2", "l8"], default="s2",
        help="Satellite sensor: s2 (Sentinel-2) or l8 (Landsat-8)"
    )
    parser.add_argument(
        "--method", choices=["rf", "xgboost", "dl", "all"], default="all",
        help="Model to run: rf, xgboost, dl, or all"
    )
    args = parser.parse_args()

    if args.method == "rf":
        from src.data.loader import init_ee
        from src.train import download_data
        from src.data.patches import extract_patches, patches_to_pixels
        from src.data.loader import get_region
        from configs.constants import S2_FEATURES, L8_FEATURES, S2_PATCH_SIZE, L8_PATCH_SIZE, N_PATCHES
        from src.models.random_forest import run_rf_pipeline
        import numpy as np

        init_ee()
        features_array, labels_array = download_data(args.sensor)
        patch_size = S2_PATCH_SIZE if args.sensor == "s2" else L8_PATCH_SIZE
        feature_names = S2_FEATURES if args.sensor == "s2" else L8_FEATURES
        X, y = extract_patches(features_array, labels_array, patch_size, N_PATCHES)
        X_pixels = patches_to_pixels(X)
        y_pixels = y.reshape(-1)
        sub_idx = np.random.choice(len(X_pixels), min(100_000, len(X_pixels)), replace=False)
        metrics = run_rf_pipeline(X_pixels[sub_idx], y_pixels[sub_idx], feature_names)
        print(f"\nRF — Accuracy: {metrics['accuracy']:.4f} | IoU: {metrics['iou']:.4f}")

    elif args.method == "xgboost":
        from src.data.loader import init_ee, download_data
        from src.train import download_data
        from src.data.patches import extract_patches, patches_to_pixels
        from configs.constants import S2_FEATURES, L8_FEATURES, S2_PATCH_SIZE, L8_PATCH_SIZE, N_PATCHES
        from src.models.xgboost_model import run_xgb_pipeline
        import numpy as np

        init_ee()
        features_array, labels_array = download_data(args.sensor)
        patch_size = S2_PATCH_SIZE if args.sensor == "s2" else L8_PATCH_SIZE
        feature_names = S2_FEATURES if args.sensor == "s2" else L8_FEATURES
        X, y = extract_patches(features_array, labels_array, patch_size, N_PATCHES)
        X_pixels = patches_to_pixels(X)
        y_pixels = y.reshape(-1)
        sub_idx = np.random.choice(len(X_pixels), min(100_000, len(X_pixels)), replace=False)
        metrics = run_xgb_pipeline(X_pixels[sub_idx], y_pixels[sub_idx], feature_names)
        print(f"\nXGBoost — Accuracy: {metrics['accuracy']:.4f} | IoU: {metrics['iou']:.4f}")

    elif args.method == "dl":
        run_dl_training(args.sensor)

    elif args.method == "all":
        run_full_comparison(args.sensor)


if __name__ == "__main__":
    main()
