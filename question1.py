import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/train.csv")

print("="*50)
print("FIRST 5 ROWS")
print(df.head())

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("="*50)
print("DATA TYPES")
print(df.dtypes)

print("="*50)
print("NULL VALUES")
print(df.isnull().sum())

print("="*50)
print("NULL VALUE PERCENTAGE")
print((df.isnull().sum()/len(df))*100)
print("="*50)
print("DATA CLEANING STARTED")

# Drop Cabin column (More than 20% missing values)
df = df.drop(columns=["Cabin"])

# Fill Age with Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("="*50)
print("NULL VALUES AFTER CLEANING")
print(df.isnull().sum())

print("="*50)
print("DATASET SHAPE AFTER CLEANING")
print(df.shape)
print("="*50)
print("DUPLICATE CHECK")

duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:")
print(df.shape)
print("="*50)
print("DATA TYPE CORRECTION")

# Sex column ko category me convert karna
df["Sex"] = df["Sex"].astype("category")

# Embarked column ko category me convert karna
df["Embarked"] = df["Embarked"].astype("category")

print(df.dtypes)

print("="*50)
print("MEMORY USAGE")
print(df.memory_usage(deep=True).sum())
print("="*50)
print("DATA TYPE CORRECTION")

df["Sex"] = df["Sex"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print(df.dtypes)

print("="*50)
print("MEMORY USAGE")
print(df.memory_usage(deep=True).sum())
print("="*50)
print("DESCRIPTIVE STATISTICS")

print(df.describe())

print("="*50)
print("SKEWNESS")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print(df[numeric_cols].skew())

highest_skew = df[numeric_cols].skew().abs().idxmax()

print("="*50)
print("COLUMN WITH HIGHEST SKEWNESS:")
print(highest_skew)
print("Skewness Value:", df[highest_skew].skew())
print("="*50)
print("OUTLIER DETECTION (IQR)")

for col in ["Age", "Fare"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print("\nColumn:", col)
    print("Q1 =", Q1)
    print("Q3 =", Q3)
    print("IQR =", IQR)
    print("Lower Bound =", lower)
    print("Upper Bound =", upper)
    print("Outliers =", outliers)
    print("="*50)
print("VISUALIZATIONS")

# 1. Line Plot
plt.figure(figsize=(8,4))
plt.plot(df["Age"])
plt.title("Age Line Plot")
plt.xlabel("Passenger Index")
plt.ylabel("Age")
plt.show()


# 2. Bar Plot
plt.figure(figsize=(6,4))
df.groupby("Sex")["Fare"].mean().plot(kind="bar")
plt.title("Average Fare by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Fare")
plt.show()


# 3. Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()


# 4. Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["Age"], df["Fare"])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()


# 5. Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x="Sex", y="Age", data=df)
plt.title("Age Distribution by Gender")
plt.show()
print("=" * 50)
print("CORRELATION MATRIX")

numeric_df = df.select_dtypes(include=["number"])

correlation = numeric_df.corr()

print(correlation)
print("=" * 50)
print("SAVING CLEANED DATASET")

df.to_csv("cleaned_data.csv", index=False)

print("File saved successfully.")
print("=" * 50)
print("SPEARMAN CORRELATION")

spearman_corr = numeric_df.corr(method="spearman")

print(spearman_corr)
print("=" * 50)
print("CORRELATION HEATMAP")

plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap="Blues", fmt=".2f")

plt.title("Spearman Correlation Heatmap")
plt.tight_layout()
plt.show()
print("=" * 50)
print("GROUPED AGGREGATION")

group_stats = df.groupby("Sex")["Fare"].agg(["mean", "std", "count"])

print(group_stats)

highest_mean = group_stats["mean"].idxmax()
highest_std = group_stats["std"].idxmax()

print("\nGroup with Highest Mean Fare:", highest_mean)
print("Group with Highest Standard Deviation:", highest_std)

ratio = group_stats["mean"].max() / group_stats["mean"].min()
print("Highest Mean / Lowest Mean Ratio:", round(ratio, 2))