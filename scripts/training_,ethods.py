from load_data import BODMASLoader
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "metadata", "bodmas.npz")
    loader = BODMASLoader(file_path)
    X_full, y_full = loader.load()
    X_sub, y_sub = loader.sample_subset(zero_count=1971, one_count=1651)
    X_train, X_test, y_train, y_test = loader.split(X_sub, y_sub, test_size=0.2)
    # Testing xgb method.
    # xgb_model = xgb.XGBClassifier(
    #             n_estimators=100,
    #             max_depth=6,
    #             learning_rate=0.1,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             objective="binary:logistic",
    #             eval_metric="auc",
    #             tree_method="hist",
    #             random_state=42,
    #             n_jobs=-1,
    #         )
    #
    # xgb_model.fit(X_train, y_train)
    # y_pred = xgb_model.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average="binary")
    # recall = recall_score(y_test, y_pred, average="binary")
    # f1 = f1_score(y_test, y_pred, average="binary")
    # roc = roc_auc_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print(f"XGBoost Model - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC-AUC: {roc}")
    n_estimators_list = range(1, 301, 5)
    learning_rate_list = np.arange(0.01, 0.8, 0.02)
    f1_scores = []

    for lr in learning_rate_list:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
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
        f1 = f1_score(y_test, y_pred, average="binary")
        f1_scores.append(f1)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate_list, f1_scores, marker='o', linewidth=2)
    plt.xlabel("lr_estimators")
    plt.ylabel("F1 Score")
    plt.title("Impact of lr_estimators on F1 Score")
    plt.grid(True)
    plt.savefig("f1_vs_lr_estimators.png")
    plt.show()