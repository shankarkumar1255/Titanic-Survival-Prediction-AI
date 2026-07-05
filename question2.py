import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("LOADING CLEANED DATASET")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("=" * 60)
print("CREATING FEATURES AND TARGETS")
print("=" * 60)

# Regression Target
y_reg = df["Fare"]

# Classification Target
y_clf = (df["Fare"] > df["Fare"].median()).astype(int)

# Feature Matrix
X = df.drop(columns=["Fare"])

print("Feature Shape:", X.shape)
print("Regression Target Shape:", y_reg.shape)
print("Classification Target Shape:", y_clf.shape)
print("=" * 60)
print("ENCODING CATEGORICAL COLUMNS")
print("=" * 60)

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

print("Encoded Feature Shape:", X.shape)
print(X.head())
print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regression Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Classification Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

print("Regression Train:", X_train_reg_scaled.shape)
print("Regression Test :", X_test_reg_scaled.shape)

print("Classification Train:", X_train_clf_scaled.shape)
print("Classification Test :", X_test_clf_scaled.shape)
print("=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_train_reg)

y_pred_reg = lr.predict(X_test_reg_scaled)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
})

coef_df["Abs"] = coef_df["Coefficient"].abs()

print("\nTop 3 Important Features:")
print(coef_df.sort_values("Abs", ascending=False).head(3))

print("=" * 60)
print("RIDGE REGRESSION")
print("=" * 60)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg_scaled, y_train_reg)

ridge_pred = ridge.predict(X_test_reg_scaled)

ridge_mse = mean_squared_error(y_test_reg, ridge_pred)
ridge_r2 = r2_score(y_test_reg, ridge_pred)

print("Ridge MSE :", ridge_mse)
print("Ridge R2  :", ridge_r2)

print("\nComparison")
print("Linear Regression R2 :", r2)
print("Ridge Regression  R2 :", ridge_r2)

print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_clf_scaled, y_train_clf)

y_pred = model.predict(X_test_clf_scaled)

print("Accuracy :", accuracy_score(y_test_clf, y_pred))
print("Precision:", precision_score(y_test_clf, y_pred))
print("Recall   :", recall_score(y_test_clf, y_pred))
print("F1 Score :", f1_score(y_test_clf, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test_clf, y_pred))

print("\nClassification Report")
print(classification_report(y_test_clf, y_pred))
\
print("=" * 60)
print("ROC CURVE AND AUC")
print("=" * 60)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test_clf_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_clf, y_prob)
auc = roc_auc_score(y_test_clf, y_prob)

print("AUC Score:", auc)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

from sklearn.metrics import precision_score, recall_score, f1_score

y_prob = model.predict_proba(X_test_clf_scaled)[:, 1]

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    precision = precision_score(y_test_clf, y_pred)
    recall = recall_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred)

    print(f"{t:<10.2f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
    print("=" * 60)
print("REGULARIZATION (C = 0.01)")
print("=" * 60)

log_reg_strong = LogisticRegression(C=0.01, max_iter=1000)

log_reg_strong.fit(X_train_clf_scaled, y_train_clf)

y_pred_strong = log_reg_strong.predict(X_test_clf_scaled)
y_prob_strong = log_reg_strong.predict_proba(X_test_clf_scaled)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score

print("Accuracy :", accuracy_score(y_test_clf, y_pred_strong))
print("AUC Score:", roc_auc_score(y_test_clf, y_prob_strong))

print("=" * 60)
print("BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 60)

import numpy as np
from sklearn.metrics import roc_auc_score

np.random.seed(42)

baseline_auc = roc_auc_score(y_test_clf, y_prob)
strong_auc = roc_auc_score(y_test_clf, y_prob_strong)

auc_differences = []

for i in range(500):
    indices = np.random.choice(
        len(y_test_clf),
        size=len(y_test_clf),
        replace=True
    )

    y_boot = y_test_clf.iloc[indices]
    prob_base = y_prob[indices]
    prob_strong = y_prob_strong[indices]

    auc_base = roc_auc_score(y_boot, prob_base)
    auc_strong = roc_auc_score(y_boot, prob_strong)

    auc_differences.append(auc_base - auc_strong)

mean_diff = np.mean(auc_differences)
lower = np.percentile(auc_differences, 2.5)
upper = np.percentile(auc_differences, 97.5)

print(f"Mean AUC Difference : {mean_diff:.6f}")
print(f"95% CI Lower Bound : {lower:.6f}")
print(f"95% CI Upper Bound : {upper:.6f}")

if lower > 0 or upper < 0:
    print("Confidence interval excludes zero.")
else:
    print("Confidence interval includes zero.")