# Loan Default Prediction Using LendingClub Data

## Overview

This project uses historical LendingClub loan data to build a machine learning model that predicts whether a loan will be fully paid or charged off (defaulted). The goal is to help financial institutions identify high-risk loans before approval. The workflow includes data cleaning, exploratory data analysis (EDA), feature encoding, model training, evaluation, and feature importance visualization.

---

## Dataset

* **Source:** [LendingClub Loan Data (2007–2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **File Used:** `accepted_2007_to_2018Q4.csv`
* **Size:** 2M+ records with borrower info, loan details, and repayment outcomes

---

## Objective

Predict the `loan_status` of applicants using:

* **0 = Fully Paid**
* **1 = Charged Off (Defaulted)**
  This is treated as a binary classification task.

---

## Tools & Technologies

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **IDE:** PyCharm
* **Version Control:** Git & GitHub

---

## Process

### 1. Data Cleaning

* Selected relevant columns such as `loan_amnt`, `term`, `emp_length`, `home_ownership`, `annual_inc`, `purpose`, and `loan_status`
* Filtered data to include only “Fully Paid” and “Charged Off” outcomes
* Dropped or filled missing values
* Encoded categorical variables using `LabelEncoder`

### 2. Exploratory Data Analysis (EDA)

* Analyzed distributions of loan amounts, income, and employment length
* Explored class imbalance in loan status
* Identified the most common loan purposes and patterns

### 3. Model Building

* Used `RandomForestClassifier` from scikit-learn
* Split dataset into training and test sets (80/20 split)
* Trained model using key features

### 4. Model Evaluation

* Evaluated with confusion matrix and classification report
* Focused on accuracy, precision, recall, and F1-score
* Visualized feature importance to interpret model behavior

## Results

### Confusion Matrix

```
[[187979  15831]
 [ 43105  64451]]
```

### Classification Report

| Metric    | Fully Paid (0) | Charged Off (1) |
| --------- | -------------- | --------------- |
| Precision | 0.81           | 0.29            |
| Recall    | 0.92           | 0.60            |
| F1-Score  | 0.86           | 0.40            |
| Accuracy  | 0.77           |                 |

### Feature Importance Plot

![feature_importance](https://github.com/user-attachments/assets/e8e90102-98a4-44d6-ae85-3a5705c49f76)

Top predictors were:

* `annual_inc`
* `loan_amnt`
* `emp_length`

---

## Key Skills Demonstrated

**Data Cleaning:** Encoded categories, removed nulls, filtered classes
**EDA:** Identified patterns and class imbalance
**Modeling:** Trained and evaluated a Random Forest classifier
**Visualization:** Plotted feature importance and evaluation metrics
**Tools:** Used Python, pandas, scikit-learn, matplotlib, seaborn, Git

---

## Next Steps

* Try class balancing (e.g., SMOTE or undersampling)
* Experiment with other models like Logistic Regression or XGBoost
* Deploy as a Streamlit app for interactive predictions
* Use SHAP values for deeper model interpretability

---

## Author

**Synthia Pial**
[LinkedIn](https://www.linkedin.com/in/your-link) | [Portfolio](https://datascienceportfol.io/synthiapial3152) 
