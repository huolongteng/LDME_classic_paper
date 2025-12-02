import numpy as np
from sklearn.model_selection import train_test_split


class BODMASLoader:
    """
    A unified class for loading BODMAS-like .npz files,
    sampling subsets, and splitting train/test sets.
    """

    def __init__(self, npz_path, random_state=42):
        self.npz_path = npz_path
        self.random_state = random_state
        self.X_full = None
        self.y_full = None

    # -----------------------------
    # 1. Load features + labels
    # -----------------------------
    def load(self):
        data = np.load(self.npz_path)
        self.X_full = data["X"]
        self.y_full = data["y"]

        return self.X_full, self.y_full

    # -----------------------------
    # 2. Random sampling of subset
    # -----------------------------
    def sample_subset(self, zero_count, one_count):
        if self.X_full is None:
            raise ValueError("Call .load() before sampling")

        rng = np.random.default_rng(self.random_state)

        zero_idx = np.where(self.y_full == 0)[0]
        one_idx = np.where(self.y_full == 1)[0]

        selected_zeros = rng.choice(zero_idx, size=zero_count, replace=False)
        selected_ones = rng.choice(one_idx, size=one_count, replace=False)

        selected = np.concatenate([selected_zeros, selected_ones])
        selected = rng.permutation(selected)

        X_sub = self.X_full[selected]
        y_sub = self.y_full[selected]

        return X_sub, y_sub

    # -----------------------------
    # 3. Train-test split
    # -----------------------------
    def split(self, X_sub, y_sub, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub,
            y_sub,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_sub,
        )

        print(f"[INFO] Split dataset:")
        print(f"  - Train: {X_train.shape}, {y_train.shape}")
        print(f"  - Test: {X_test.shape}, {y_test.shape}")

        return X_train, X_test, y_train, y_test
