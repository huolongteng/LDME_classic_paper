import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import importlib.util

# 路径配置：bodmas.npz 存放在 matadata 目录下
# 数据规模 (134435, 2381) 的特征矩阵和 (134435,) 的标签向量
npz_path = "matadata/bodmas.npz"

# 步骤 1：加载数据（features: X_full，labels: y_full）
# 需要本地存在 bodmas.npz 文件。
data = np.load(npz_path)
X_full = data["X"]
y_full = data["y"]

# 步骤 2：分别抽取 benign (y==0) 1971 条、malicious (y==1) 1651 条
rng = np.random.default_rng(seed=42)
benign_idx = np.where(y_full == 0)[0]
malicious_idx = np.where(y_full == 1)[0]
selected_benign = rng.choice(benign_idx, size=1971, replace=False)
selected_malicious = rng.choice(malicious_idx, size=1651, replace=False)

# 步骤 3：合并并随机打乱，得到 X_sub, y_sub
all_indices = np.concatenate([selected_benign, selected_malicious])
shuffled_indices = rng.permutation(all_indices)
X_sub = X_full[shuffled_indices]
y_sub = y_full[shuffled_indices]

# 步骤 4：划分训练测试集（80/20，分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X_sub,
    y_sub,
    test_size=0.2,
    random_state=42,
    stratify=y_sub,
)

# 步骤 5：定义模型列表，采用统一的 (name, estimator) 结构
models = []
models.append(("GaussianNB", GaussianNB()))
models.append(("LogisticRegression", make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200, penalty="l2"))))
models.append(("LinearSVM", make_pipeline(StandardScaler(with_mean=False), SVC(kernel="linear", probability=True, random_state=42))))
models.append(("DecisionTree", DecisionTreeClassifier(random_state=42, max_depth=None)))
models.append(("RandomForest", RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1)))
models.append(("GradientBoosting", GradientBoostingClassifier(random_state=42)))
models.append(("AdaBoost_DT", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=42), n_estimators=200, random_state=42)))

# 可选模型：XGBoost 和 LightGBM（若安装则添加）
xgb_spec = importlib.util.find_spec("xgboost")
if xgb_spec is not None:
    import xgboost as xgb  # noqa: E402

    models.append(
        (
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        )
    )

lgb_spec = importlib.util.find_spec("lightgbm")
if lgb_spec is not None:
    import lightgbm as lgb  # noqa: E402

    models.append(
        (
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
        )
    )

# 步骤 6：训练与评估，保存指标（准确率、ROC-AUC、混淆矩阵）
results = []
for name, estimator in models:
    fitted = estimator.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    if hasattr(fitted, "predict_proba"):
        y_scores = fitted.predict_proba(X_test)[:, 1]
    else:
        decision_scores = fitted.decision_function(X_test)
        y_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-12)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)
    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": roc,
            "Confusion_Matrix": cm,
        }
    )

# 步骤 7：输出结果表格
results_table = pd.DataFrame(results)
print("===== Model Comparison =====")
print(results_table[["Model", "Accuracy", "ROC_AUC"]])
print("\n===== Confusion Matrices =====")
for row in results:
    print(f"\n{row['Model']}\n{row['Confusion_Matrix']}")
