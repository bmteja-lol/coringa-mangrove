import numpy as np
import torch


def extract_patches(features, labels, patch_size=64, n_patches=1000, seed=42):
    """Extract random patches from a downloaded numpy array.

    Used in final.ipynb and sheesh.ipynb after geemap.ee_to_numpy().

    Args:
        features: (H, W, C) numpy array of spectral bands + indices
        labels: (H, W, 1) numpy array of binary mangrove mask
        patch_size: spatial size of each patch
        n_patches: number of patches to extract
        seed: random seed for reproducibility

    Returns:
        X: (N, C, patch_size, patch_size) normalized feature patches
        y: (N, 1, patch_size, patch_size) label patches
    """
    h, w, c = features.shape
    X_list = []
    y_list = []

    np.random.seed(seed)
    for _ in range(n_patches):
        r = np.random.randint(0, h - patch_size)
        c_idx = np.random.randint(0, w - patch_size)

        patch_x = features[r:r+patch_size, c_idx:c_idx+patch_size]
        patch_y = labels[r:r+patch_size, c_idx:c_idx+patch_size]

        X_list.append(patch_x.transpose(2, 0, 1))  # (C, H, W)
        y_list.append(patch_y.transpose(2, 0, 1))  # (1, H, W)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Per-channel normalization (z-score)
    for ch in range(X.shape[1]):
        mean = X[:, ch].mean()
        std = X[:, ch].std() + 1e-8
        X[:, ch] = (X[:, ch] - mean) / std

    return X, y


def patches_to_pixels(X):
    """Flatten patch array to pixel-level for RF/XGBoost.

    Args:
        X: (N, C, H, W) patch array

    Returns:
        (N*H*W, C) pixel array
    """
    n, c, h, w = X.shape
    return X.reshape(n, c, -1).transpose(0, 2, 1).reshape(-1, c)


def make_torch_dataset(X, y):
    """Convert numpy arrays to PyTorch Dataset."""
    class MangroveDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return MangroveDataset(X, y)


def split_data(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split data into train/val/test with random permutation.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train = X[idx[:n_train]]
    X_val = X[idx[n_train:n_train+n_val]]
    X_test = X[idx[n_train+n_val:]]
    y_train = y[idx[:n_train]]
    y_val = y[idx[n_train:n_train+n_val]]
    y_test = y[idx[n_train+n_val:]]

    return X_train, X_val, X_test, y_train, y_val, y_test
