import os
from typing import Any, Callable, Dict, Optional

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from load_data import BODMASLoader
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    """
    Unified training interface to support multiple models with a consistent workflow.

    Parameters
    ----------
    model_name : str
        The model identifier. Must be one of the keys in ``MODEL_BUILDERS``.
    custom_params : dict, optional
        Overrides or extends the default parameters for the chosen model.
    """

    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        "LightGBM": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "force_col_wise": True,
            "objective": "binary",
            "random_state": 42,
            "n_jobs": -1,
        },
        "XGBoost": {
            "n_estimators": 170,
            "max_depth": 4,
            "learning_rate": 0.13,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        },
        "GaussianNB": {},
        "LogisticRegression": {"max_iter": 1000, "n_jobs": -1, "solver": "lbfgs"},
        "LinearSVM": {"C": 1.0, "max_iter": 10000, "random_state": 42},
        "DecisionTree": {"random_state": 42},
        "RandomForest": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
        "GradientBoosting": {"random_state": 42},
        "AdaBoost_DT": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "estimator": DecisionTreeClassifier(max_depth=1, random_state=42),
            "random_state": 42,
        },
    }

    MODEL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "LightGBM": lambda params: lgb.LGBMClassifier(**params),
        "XGBoost": lambda params: xgb.XGBClassifier(**params),
        "GaussianNB": lambda params: GaussianNB(**params),
        "LogisticRegression": lambda params: LogisticRegression(**params),
        "LinearSVM": lambda params: LinearSVC(**params),
        "DecisionTree": lambda params: DecisionTreeClassifier(**params),
        "RandomForest": lambda params: RandomForestClassifier(**params),
        "GradientBoosting": lambda params: GradientBoostingClassifier(**params),
        "AdaBoost_DT": lambda params: AdaBoostClassifier(**params),
    }

    def __init__(self, model_name: str, custom_params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.custom_params = custom_params or {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_name not in self.MODEL_BUILDERS:
            available = ", ".join(sorted(self.MODEL_BUILDERS))
            raise ValueError(f"Unsupported model '{self.model_name}'. Available: {available}")

        default_params = self.DEFAULT_PARAMS.get(self.model_name, {}).copy()
        default_params.update(self.custom_params)
        builder = self.MODEL_BUILDERS[self.model_name]
        return builder(default_params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        roc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(
            f"{self.model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC:{roc:.4f}"
        )
        print(f"Confusion Matrix:\n{cm}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc,
            "confusion_matrix": cm,
        }

    def save(self, model_dir: str, model_filename: Optional[str] = None):
        os.makedirs(model_dir, exist_ok=True)
        filename = model_filename or f"{self.model_name.lower()}_model.pkl"
        model_path = os.path.join(model_dir, filename)
        joblib.dump(self.model, model_path)
        print(f"Model saved at: {model_path}")
        return model_path

    def train_evaluate_save(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_dir: Optional[str] = None,
        model_filename: Optional[str] = None,
    ):
        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)

        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "..", "models", self.model_name.lower())

        self.save(model_dir, model_filename)
        return metrics


def run_manual_training():
    """
    Manual entry point for IDE debugging.

    Modify the variables below to experiment with different tasks or data splits.
    """

    file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
    zero_count = 1971
    one_count = 1651
    test_size = 0.2
    task = "RandomForest"  # Options: ModelTrainer.MODEL_BUILDERS keys
    custom_params: Dict[str, Any] = {}

    loader = BODMASLoader(file_path)
    loader.load()
    X_sub, y_sub = loader.sample_subset(zero_count=zero_count, one_count=one_count)
    X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=test_size)


    trainer = ModelTrainer(task, {"random_state": 42, "max_depth": 2})
    trainer.train_evaluate_save(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_manual_training()
