import joblib
import os

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from load_data import BODMASLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_and_save(model, X_test, y_test, model_dir, model_filename, model_label):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(
        f"{model_label} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC:{roc:.4f}"
    )
    print(f"Confusion Matrix:\n{cm}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")


def train_lightgbm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        force_col_wise=True,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models", "lightgbm")
    evaluate_and_save(
        model,
        X_test,
        y_test,
        model_dir,
        "lightgbm_model.pkl",
        "LightGBM Model",
    )


def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=170,
    max_depth=4,
    learning_rate=0.13,
):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models", "xgboost")
    evaluate_and_save(
        model,
        X_test,
        y_test,
        model_dir,
        "xgboost_model.pkl",
        "XGBoost Model",
    )


def run_manual_training():
    """
    Manual entry point for IDE debugging.

    Modify the variables below to experiment with different tasks or data splits.
    """

    file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
    zero_count = 1971
    one_count = 1651
    test_size = 0.2
    task = "XGBoost"  # Options: "LightGBM", "XGBoost"

    loader = BODMASLoader(file_path)
    loader.load()
    X_sub, y_sub = loader.sample_subset(zero_count=zero_count, one_count=one_count)
    X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=test_size)

    if task == "XGBoost":
        train_xgboost(X_train, y_train, X_test, y_test)
    else:
        train_lightgbm(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_manual_training()
