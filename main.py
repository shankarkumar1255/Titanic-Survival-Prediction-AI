import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.shape)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[0, 1, 2, 3, 4]
).astype(int)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

title_mapping = {
    "Mr": 0,
    "Miss": 1,
    "Mrs": 2,
    "Master": 3,
    "Dr": 4,
    "Rev": 5,
    "Major": 6,
    "Mlle": 1,
    "Col": 6,
    "Don": 0,
    "Mme": 2,
    "Ms": 1,
    "Lady": 2,
    "Sir": 0,
    "Capt": 6,
    "Countess": 2,
    "Jonkheer": 0
}

df["Title"] = df["Title"].map(title_mapping)

print(df["Title"].value_counts())

print(df[["Name", "Title"]].head())

print(df[["FamilySize", "IsAlone"]].head())

print(df[["SibSp", "Parch", "FamilySize"]].head())

print(df.isnull().sum())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print(df.head())
from sklearn.model_selection import train_test_split

X = df.drop("Survived", axis=1)
X = X.drop(["Name", "Ticket", "Age"], axis=1)

y = df["Survived"]

print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(rf_model, "models/titanic_model.pkl")

print("Model saved successfully!")

print("\n========== PART 3 ==========")

# Decision Tree (Baseline)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree Train Accuracy:", dt.score(X_train, y_train))
print("Decision Tree Test Accuracy:", dt.score(X_test, y_test))


# Controlled Decision Tree
dt2 = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

dt2.fit(X_train, y_train)

print("Controlled DT Train Accuracy:", dt2.score(X_train, y_train))
print("Controlled DT Test Accuracy:", dt2.score(X_test, y_test))


# Gini vs Entropy
gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

gini.fit(X_train, y_train)
entropy.fit(X_train, y_train)

print("Gini Accuracy:", gini.score(X_test, y_test))
print("Entropy Accuracy:", entropy.score(X_test, y_test))

# Random Forest ROC-AUC
rf_prob = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)

print("Random Forest ROC-AUC:", rf_auc)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 5 Important Features:")
print(importance.head())