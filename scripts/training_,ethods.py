import joblib
from load_data import BODMASLoader
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb

def train_lightgbm(X_train,
                   y_train,
                   X_test,
                   y_test,
                   n_estimators=500,
                   learning_rate=0.05,
                   num_leaves=64):
    model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"XGBoost Model - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models", "xgboost")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgboost_model.pkl")

    joblib.dump(model, model_path)
    print(f"Model saved at：{model_path}")


def train_xgboost(X_train,
                  y_train,
                  X_test,
                  y_test,
                  n_estimators=170,
                  max_depth=4,
                  learning_rate=0.13):
    xgb_model = xgb.XGBClassifier(
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

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"XGBoost Model - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "models", "xgboost")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgboost_model.pkl")

    joblib.dump(xgb_model, model_path)
    print(f"Model saved at：{model_path}")

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
    loader = BODMASLoader(file_path)
    X_full, y_full = loader.load()
    X_sub, y_sub = loader.sample_subset(zero_count=1971, one_count=1651)
    X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=0.2)
    task = "LightGBM"
    match task:
        case "XGBoost":
            train_xgboost(X_train, y_train, X_test, y_test)
        case "LightGBM":
            train_lightgbm(X_train, y_train, X_test, y_test)




    # n_estimators_list = range(1, 301, 5)
    # learning_rate_list = np.arange(0.01, 0.8, 0.02)
    # max_depth_list = np.arange(1, 11, 1)
    # f1_scores = []
    #
    # for depth in max_depth_list:
    #     xgb_model = xgb.XGBClassifier(
    #         n_estimators=500,
    #         max_depth=depth,
    #         learning_rate=0.05,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         objective="binary:logistic",
    #         eval_metric="auc",
    #         tree_method="hist",
    #         random_state=42,
    #         n_jobs=-1,
    #     )
    #     xgb_model.fit(X_train, y_train)
    #     y_pred = xgb_model.predict(X_test)
    #     f1 = f1_score(y_test, y_pred, average="binary")
    #     f1_scores.append(f1)
    #
    # # 绘图
    # plt.figure(figsize=(10, 6))
    # plt.plot(max_depth_list, f1_scores, marker='o', linewidth=2)
    # plt.xlabel("max_depth_estimators")
    # plt.ylabel("F1 Score")
    # plt.title("Impact of max_depth_estimators on F1 Score")
    # plt.grid(True)
    # plt.savefig("f1_vs_max_depth_estimators.png")
    # plt.show()