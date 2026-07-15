import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from configs.constants import (
    DL_ENCODER, DL_ENCODER_WEIGHTS, DL_CLASSES,
    DL_FOCAL_ALPHA, DL_FOCAL_GAMMA, DL_LR,
    BATCH_SIZE, EPOCHS
)
from src.data.patches import make_torch_dataset, split_data


def build_model(in_channels=11):
    """Build DeepLabV3+ with EfficientNet-B0 encoder.

    Uses segmentation_models_pytorch library.
    Binary segmentation (mangrove vs non-mangrove).
    """
    model = smp.DeepLabV3Plus(
        encoder_name=DL_ENCODER,
        encoder_weights=DL_ENCODER_WEIGHTS,
        in_channels=in_channels,
        classes=DL_CLASSES,
    )
    return model


def compute_iou(preds, targets, threshold=0.5):
    """Compute Intersection over Union for binary segmentation."""
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + 1e-7) / (union + 1e-7)).item()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model. Returns average loss and average IoU."""
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
            total_iou += compute_iou(preds, y_batch)
    return total_loss / len(loader), total_iou / len(loader)


def train_dl(X, y, in_channels=11, save_path="best_model.pt"):
    """Full DeepLabV3+ training pipeline.

    Args:
        X: (N, C, H, W) patch array
        y: (N, 1, H, W) label array
        in_channels: number of input channels
        save_path: path to save best model weights

    Returns:
        model, metrics dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create data loaders
    train_loader = DataLoader(
        make_torch_dataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        make_torch_dataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        make_torch_dataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Build model, loss, optimizer
    model = build_model(in_channels).to(device)
    criterion = smp.losses.FocalLoss(
        mode="binary", alpha=DL_FOCAL_ALPHA, gamma=DL_FOCAL_GAMMA
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LR)

    # Training loop
    best_val_iou = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | IoU: {val_iou:.4f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(save_path))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_binary = (preds > 0.5).float().cpu().numpy()
            all_preds.append(preds_binary)
            all_labels.append(y_batch.numpy())

    dl_preds = np.concatenate(all_preds).flatten()
    dl_labels = np.concatenate(all_labels).flatten()
    dl_acc = (dl_preds == dl_labels).mean()
    dl_iou = compute_iou(
        torch.tensor(dl_preds), torch.tensor(dl_labels)
    )

    metrics = {
        "accuracy": float(dl_acc),
        "iou": float(dl_iou),
        "best_val_iou": float(best_val_iou),
        "y_pred": dl_preds
    }

    return model, metrics
