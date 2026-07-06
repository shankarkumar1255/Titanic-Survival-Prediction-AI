import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

import joblib
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

print("Dataset Shape:", df.shape)

# Regression and Classification Target
y_reg = df["Fare"]
y_clf = (df["Fare"] > df["Fare"].median()).astype(int)

# Feature Matrix
X = df.drop(columns=["Fare"])

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_clf,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Shape :", X_train_scaled.shape)
print("Testing Shape  :", X_test_scaled.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("\n" + "=" * 60)
print("TASK 1 - DECISION TREE BASELINE")
print("=" * 60)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train_scaled, y_train)

train_pred = dt.predict(X_train_scaled)
test_pred = dt.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")
print("\n" + "=" * 60)
print("TASK 2 - CONTROLLED DECISION TREE")
print("=" * 60)

dt_control = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

dt_control.fit(X_train_scaled, y_train)

train_pred2 = dt_control.predict(X_train_scaled)
test_pred2 = dt_control.predict(X_test_scaled)

train_acc2 = accuracy_score(y_train, train_pred2)
test_acc2 = accuracy_score(y_test, test_pred2)

print(f"Training Accuracy : {train_acc2:.4f}")
print(f"Testing Accuracy  : {test_acc2:.4f}")

print("\n" + "=" * 60)
print("TASK 3 - GINI vs ENTROPY")
print("=" * 60)

# Gini Model
gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

gini_model.fit(X_train_scaled, y_train)
gini_pred = gini_model.predict(X_test_scaled)
gini_acc = accuracy_score(y_test, gini_pred)

# Entropy Model
entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

entropy_model.fit(X_train_scaled, y_train)
entropy_pred = entropy_model.predict(X_test_scaled)
entropy_acc = accuracy_score(y_test, entropy_pred)

print(f"Gini Test Accuracy    : {gini_acc:.4f}")
print(f"Entropy Test Accuracy : {entropy_acc:.4f}")

from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("TASK 4 - RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

train_pred_rf = rf_model.predict(X_train_scaled)
test_pred_rf = rf_model.predict(X_test_scaled)

train_acc_rf = accuracy_score(y_train, train_pred_rf)
test_acc_rf = accuracy_score(y_test, test_pred_rf)

train_prob_rf = rf_model.predict_proba(X_train_scaled)[:, 1]
test_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

train_auc_rf = roc_auc_score(y_train, train_prob_rf)
test_auc_rf = roc_auc_score(y_test, test_prob_rf)

print(f"Training Accuracy : {train_acc_rf:.4f}")
print(f"Testing Accuracy  : {test_acc_rf:.4f}")
print(f"Training ROC-AUC  : {train_auc_rf:.4f}")
print(f"Testing ROC-AUC   : {test_auc_rf:.4f}")

print("\nTop 5 Important Features")

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance.head(5))

print(feature_importance.head(5))

# ----------------------------
# YAHAN SE NAYA CODE ADD KARO
# ----------------------------

print("\n" + "=" * 60)
print("TASK 4A - GRADIENT BOOSTING")
print("=" * 60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

train_pred_gb = gb_model.predict(X_train_scaled)
test_pred_gb = gb_model.predict(X_test_scaled)

train_acc_gb = accuracy_score(y_train, train_pred_gb)
test_acc_gb = accuracy_score(y_test, test_pred_gb)

train_prob_gb = gb_model.predict_proba(X_train_scaled)[:, 1]
test_prob_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

train_auc_gb = roc_auc_score(y_train, train_prob_gb)
test_auc_gb = roc_auc_score(y_test, test_prob_gb)

print(f"Training Accuracy : {train_acc_gb:.4f}")
print(f"Testing Accuracy  : {test_acc_gb:.4f}")
print(f"Training ROC-AUC  : {train_auc_gb:.4f}")
print(f"Testing ROC-AUC   : {test_auc_gb:.4f}")
print("\n" + "=" * 60)
print("TASK 4A - GRADIENT BOOSTING")

print("\n" + "=" * 60)
print("TASK 5 - CROSS VALIDATION")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": None,
    "Decision Tree": dt_control,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}
for name, model in models.items():

    if model is None:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)

    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    print(f"\n{name}")
    print(f"Mean ROC-AUC : {scores.mean():.4f}")
    print(f"Std ROC-AUC  : {scores.std():.4f}")

    print("\n" + "=" * 60)
print("TASK 6 - GRID SEARCH CV")
print("=" * 60)
param_grid = {
    "randomforestclassifier__n_estimators": [50, 100],
    "randomforestclassifier__max_depth": [5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 2]
}

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest ROC-AUC Score:")
print(grid_search.best_score_)

print("\n" + "=" * 60)
print("TASK 7 - MANUAL LEARNING CURVE")
print("=" * 60)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

print("Fraction\tTrain AUC\tTest AUC")
for frac in fractions:

    n = int(len(X_train) * frac)

    X_small = X_train.iloc[:n]
    y_small = y_train.iloc[:n]

    best_pipeline = grid_search.best_estimator_

    best_pipeline.fit(X_small, y_small)

    train_prob = best_pipeline.predict_proba(X_small)[:, 1]
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_small, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"{frac:.1f}\t\t{train_auc:.4f}\t\t{test_auc:.4f}")

    print("\n" + "=" * 60)
print("TASK 8 - SAVE BEST MODEL")
print("=" * 60)

best_pipeline = grid_search.best_estimator_

joblib.dump(best_pipeline, "best_model.pkl")

print("Model saved successfully as best_model.pkl")

loaded_model = joblib.load("best_model.pkl")

sample_prediction = loaded_model.predict(X_test.iloc[:2])

print("Sample Prediction:", sample_prediction)

print("\n" + "=" * 60)
print("TASK 9 - SUMMARY COMPARISON")
print("=" * 60)

summary = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Mean ROC-AUC": [
        0.9597,
        0.9237,
        0.9319,
        0.9600
    ]
})

print(summary)