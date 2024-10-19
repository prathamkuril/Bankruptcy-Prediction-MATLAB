# Bankruptcy Prediction with MATLAB using Linear Regression and Random Forest models

## Overview

This project is focused on predicting bankruptcy using financial data. We explore data preprocessing, oversampling techniques (SMOTE), feature selection, and classification models like Logistic Regression and Random Forest. The project aims to address class imbalance and optimize model performance using k-fold cross-validation.

## Table of Contents

1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [SMOTE Oversampling](#smote-oversampling)
4. [Feature Selection](#feature-selection)
5. [Model Training & Evaluation](#model-training-evaluation)
6. [Results](#results)
7. [Files](#files)

## Dataset

The dataset used contains financial variables that are indicators of whether a company is bankrupt or not. The target variable is `Bankrupt`, with `0` representing non-bankrupt companies and `1` representing bankrupt companies.

**Dataset Path**: `data.csv`

## Preprocessing

We perform the following preprocessing steps:
1. **Size Check**: We check the size of the dataset using the `size()` function.
2. **Statistics Summary**: Summary statistics for each feature are displayed using the `summary()` function.
3. **Missing Values Check**: We check for missing values and report their summary.
4. **Duplicate Rows Check**: Duplicate rows are detected and their count is displayed.
5. **Feature Separation**: Features (X) and target (y) are separated, excluding the `Bankrupt` column for feature matrix.

## SMOTE Oversampling

Due to class imbalance (i.e., more non-bankrupt companies than bankrupt), we apply **Synthetic Minority Oversampling Technique (SMOTE)** to balance the dataset. Here's how it's done:
1. We calculate the class distribution before applying SMOTE.
2. Using K-nearest neighbors (k=5), synthetic samples are generated for the minority class (bankrupt companies).
3. After applying SMOTE, the class distribution is rechecked and visualized.

## Feature Selection

To select the most relevant features, we compute correlations between the features and the target variable:
1. **Top 10 Positively Correlated Features**
2. **Top 10 Negatively Correlated Features**

These features are used to create a balanced feature matrix for training the models.

## Model Training & Evaluation

We implement two classification models:
- **Logistic Regression** using Lasso for feature selection and model optimization.
- **Random Forest** with 100 trees for classification.

We use **5-fold cross-validation** to evaluate model performance on metrics such as accuracy, precision, recall, and F1-score.

Metrics are calculated for both models and the model with the best performance is saved.

## Results

### Logistic Regression
- **Average Accuracy**: 88.82%
- **Average Precision**: 88.10%
- **Average Recall**: 90.40%
- **Average F1 Score**: 89.42%

### Random Forest
- **Average Accuracy**: 96.55%
- **Average Precision**: 95.63%
- **Average Recall**: 97.58%
- **Average F1 Score**: 96.59%

The Random Forest model outperforms Logistic Regression, showing higher accuracy and F1-score.

## Files

- **best_LR_model.mat**: Best Logistic Regression model.
- **best_RF_model.mat**: Best Random Forest model.
- **X_train.csv, y_train.csv**: Training data after feature selection and SMOTE.
- **X_test.csv, y_test.csv**: Test data after feature selection and SMOTE.

## Visualization

The following visualizations are included:
- **Class Distributions**: Before and after SMOTE.
- **Correlation Heatmap**: Displaying the correlation between selected features and the target variable.
- **Model Performance Plots**: Accuracy, precision, recall, and F1-score for both Logistic Regression and Random Forest across all cross-validation folds.
- **Confusion Matrices**: For both models on the entire dataset.

---

**Published with MATLAB® R2023b**
