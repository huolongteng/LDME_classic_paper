import numpy as np
from sklearn.model_selection import train_test_split


def load_bodmas_dataset(npz_path: str = "matadata/bodmas.npz"):
    """加载 BODMAS 数据集，兼容 'X'/'y' 或 arr_0/arr_1 键名。"""
    bodmas_data = np.load(npz_path)
    if "X" in bodmas_data.files and "y" in bodmas_data.files:
        X_data = bodmas_data["X"]
        y_data = bodmas_data["y"]
    else:
        X_data = bodmas_data["arr_0"]
        y_data = bodmas_data["arr_1"]
    return X_data, y_data


def sample_bodmas_subset(
    X_full,
    y_full,
    benign_count: int = 1971,
    malware_count: int = 1651,
    random_state: int = 42,
):
    """从 BODMAS 全量数据中抽取指定数量的良性/恶意样本并打乱。"""
    rng = np.random.default_rng(seed=random_state)
    benign_indices = np.where(y_full == 0)[0]
    malware_indices = np.where(y_full == 1)[0]

    selected_benign = rng.choice(benign_indices, benign_count, replace=False)
    selected_malware = rng.choice(malware_indices, malware_count, replace=False)

    selected_indices = np.concatenate([selected_benign, selected_malware])
    shuffled_indices = rng.permutation(selected_indices)

    X_sub = X_full[shuffled_indices]
    y_sub = y_full[shuffled_indices]
    return X_sub, y_sub


def split_train_test(
    X_sub,
    y_sub,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """按给定比例分层划分训练/测试集。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=test_size,
        stratify=y_sub,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def prepare_bodmas_train_test(
    npz_path: str = "matadata/bodmas.npz",
    benign_count: int = 1971,
    malware_count: int = 1651,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """整体封装：加载、抽样并划分 BODMAS 数据集。"""
    X_full, y_full = load_bodmas_dataset(npz_path=npz_path)
    X_sub, y_sub = sample_bodmas_subset(
        X_full,
        y_full,
        benign_count=benign_count,
        malware_count=malware_count,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = split_train_test(
        X_sub,
        y_sub,
        test_size=test_size,
        random_state=random_state,
    )
    return {
        "X_full": X_full,
        "y_full": y_full,
        "X_sub": X_sub,
        "y_sub": y_sub,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    dataset = prepare_bodmas_train_test()
    print("Full dataset shapes:", dataset["X_full"].shape, dataset["y_full"].shape)
    print("Subset shapes:", dataset["X_sub"].shape, dataset["y_sub"].shape)
    print("Train shapes:", dataset["X_train"].shape, dataset["y_train"].shape)
    print("Test shapes:", dataset["X_test"].shape, dataset["y_test"].shape)
