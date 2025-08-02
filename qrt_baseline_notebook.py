# QRT Challenge Notebook: Boosted Version with Threshold + Supplementary Features

# 1. Imports and Configuration
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Optional fix if LightGBM import stalls due to Dask
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"
import lightgbm as lgb

# 2. Load Data (assume all CSVs are in the same folder as notebook)
x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("X_test.csv")
supplementary = pd.read_csv("supplementary_data.csv")

# 3. Merge Data
# First join y_train to x_train
df = pd.merge(x_train, y_train, on="ID")
# Then join supplementary
df = pd.merge(df, supplementary, left_on="ID_TARGET", right_on="ID_asset", how="left")

# 4. Create Target Sign
df["SIGN"] = np.sign(df["RET_TARGET"]).astype(int)

# 5. Feature Engineering
ret_cols = [col for col in df.columns if col.startswith("RET_") and col != "RET_TARGET"]
df["RET_MEAN"] = df[ret_cols].mean(axis=1)
df["RET_STD"] = df[ret_cols].std(axis=1)
df["RET_MAX"] = df[ret_cols].max(axis=1)
df["RET_MIN"] = df[ret_cols].min(axis=1)
df["RET_POS"] = (df[ret_cols] > 0).sum(axis=1)
df["RET_NEG"] = (df[ret_cols] < 0).sum(axis=1)

# Encode class levels
df = pd.get_dummies(df, columns=["CLASS_LEVEL_1", "CLASS_LEVEL_2", "CLASS_LEVEL_3", "CLASS_LEVEL_4"], drop_first=True)

# Select Features
feature_cols = [col for col in df.columns if col.startswith("RET_") and col != "RET_TARGET"]
feature_cols += ["RET_MEAN", "RET_STD", "RET_MAX", "RET_MIN", "RET_POS", "RET_NEG"]
feature_cols += [col for col in df.columns if col.startswith("CLASS_LEVEL_")]

X = df[feature_cols]
y = df["SIGN"]
sample_weights = np.abs(df["RET_TARGET"])

# 6. Train/Test Split
X_train, X_val, y_train_split, y_val_split, w_train, w_val = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# 7. Train LightGBM Classifier
clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42)
clf.fit(X_train, y_train_split, sample_weight=w_train)

# 8. Evaluate with Threshold-based Predictions
probs = clf.predict_proba(X_val)
preds = np.where(probs[:, 1] > 0.6, 1, np.where(probs[:, 1] < 0.4, -1, 0))

def weighted_accuracy_score(y_true, y_pred, weights):
    return np.sum(weights * (y_pred == y_true)) / np.sum(weights)

score = weighted_accuracy_score(y_val_split, preds, w_val)
print("Custom Threshold Weighted Accuracy:", round(score, 4))

# 9. Feature Importance Plot
lgb.plot_importance(clf, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# 10. Prepare Test Data for Submission
x_test = pd.merge(x_test, supplementary, left_on="ID_TARGET", right_on="ID_asset", how="left")
x_test = pd.get_dummies(x_test, columns=["CLASS_LEVEL_1", "CLASS_LEVEL_2", "CLASS_LEVEL_3", "CLASS_LEVEL_4"], drop_first=True)

# Recreate engineered features
x_test["RET_MEAN"] = x_test[ret_cols].mean(axis=1)
x_test["RET_STD"] = x_test[ret_cols].std(axis=1)
x_test["RET_MAX"] = x_test[ret_cols].max(axis=1)
x_test["RET_MIN"] = x_test[ret_cols].min(axis=1)
x_test["RET_POS"] = (x_test[ret_cols] > 0).sum(axis=1)
x_test["RET_NEG"] = (x_test[ret_cols] < 0).sum(axis=1)

# Align columns
x_test = x_test.reindex(columns=X.columns, fill_value=0)

test_ids = x_test[["ID"]].copy()
probs_test = clf.predict_proba(x_test)
y_test_pred = np.where(probs_test[:, 1] > 0.6, 1, np.where(probs_test[:, 1] < 0.4, -1, 0))

# 11. Save Submission
submission = pd.DataFrame({
    "ID": test_ids["ID"],
    "RET_TARGET": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
