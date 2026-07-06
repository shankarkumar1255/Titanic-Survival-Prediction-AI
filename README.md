# Titanic Dataset - Data Acquisition, Cleaning and Exploratory Data Analysis

## Overview

This project performs data acquisition, data cleaning, exploratory data analysis (EDA), correlation analysis and visualization on the Titanic dataset. Missing values are handled, duplicate records are checked, data types are corrected, outliers are analyzed using the IQR method, and different visualizations are created to better understand the dataset.

## Dataset

- Dataset: Titanic Train Dataset
- Total Rows: 891
- Total Columns: 12 (before cleaning)
- Cleaned Dataset: cleaned_data.csv

## Data Cleaning

- Loaded the dataset using pandas.
- Checked dataset shape and data types.
- Calculated missing values and their percentages.
- Removed the Cabin column because it had a large number of missing values.
- Filled missing Age values using the median.
- Filled missing Embarked values using the mode.
- Checked duplicate records and found no duplicates.
- Converted suitable columns to category type to reduce memory usage.

## Descriptive Statistics

The numerical columns were analyzed using descriptive statistics. The Age column had an average value close to 29 years. The Fare column showed high variation because a few passengers paid much higher ticket prices than others.

## Skewness Analysis

The Fare column had the highest positive skewness (around 4.79). This indicates that most passengers paid lower fares while a small number paid very high fares. Since the Fare distribution is highly skewed, the median is more reliable than the mean for handling missing values because it is less affected by extreme values.

## Outlier Detection (IQR)

Outliers were detected using the IQR method.

- Age: 66 outliers were found.
- Fare: 116 outliers were found.

The outliers were not removed because they may represent valid passenger information rather than incorrect data.

## Visualizations

The following visualizations were created:

- Age Line Plot
- Average Fare by Gender (Bar Chart)
- Fare Distribution (Histogram)
- Age vs Fare (Scatter Plot)
- Age Distribution by Gender (Box Plot)
- Spearman Correlation Heatmap

## Correlation Analysis

Both Pearson and Spearman correlation matrices were calculated.

- Fare and Pclass showed a strong negative correlation, indicating that passengers in higher classes generally paid higher fares.
- SibSp and Parch showed a moderate positive correlation because family members often travelled together.
- Spearman correlation was also calculated to better understand monotonic relationships between numerical variables.

## Grouped Aggregation

The Fare column was grouped by Sex.

Results:

- Female passengers had the highest average fare.
- Female passengers also had the highest standard deviation in fare.
- The ratio between the highest and lowest average fare was approximately **1.74**, showing that female passengers paid higher fares on average.

## Conclusion

The dataset was successfully cleaned and explored. Missing values were handled, duplicate records were checked, data types were optimized, outliers were analyzed, and multiple visualizations were created. Correlation analysis and grouped aggregation provided additional insights into the dataset. Finally, the cleaned dataset was saved as **cleaned_data.csv** for use in the next stages of the project.

---

# Question 2 - Machine Learning Models

## Linear Regression
- Trained a Linear Regression model.
- Evaluated using Mean Squared Error (MSE) and R² Score.

## Ridge Regression
- Applied Ridge Regression to reduce overfitting.
- Compared its performance with Linear Regression.

## Logistic Regression
- Built a Logistic Regression classifier.
- Evaluated using Accuracy, Precision, Recall and F1 Score.

## ROC Curve and AUC
- Generated ROC Curve.
- Achieved an AUC score of approximately 0.978.

## Threshold Analysis
- Compared different probability thresholds (0.30, 0.40, 0.50, 0.60 and 0.70).
- Observed changes in Precision, Recall and F1 Score.

## Regularization
- Applied Logistic Regression with C = 0.01.
- Compared Accuracy and AUC with the default model.

## Bootstrap Confidence Interval
- Estimated the confidence interval for AUC using bootstrap sampling.
- The confidence interval includes zero, indicating no statistically significant difference.