# Loan Default Prediction Using LendingClub Data

This project uses machine learning to predict whether a loan applicant is likely to default based on historical LendingClub data. It includes data cleaning, exploratory data analysis, feature engineering, model training with a Random Forest classifier, and performance evaluation.

## Dataset

- Source:** [LendingClub Loan Data (2007–2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Size:** Over 2 million loan records
- File:** `accepted_2007_to_2018Q4.csv`

## Objective

Build a binary classification model to predict `loan_status`:
- `0` → Fully Paid
- `1` → Charged Off (defaulted)

## Technologies Used

- Python 3.9
- `pandas`, `numpy` for data wrangling
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for machine learning (Random Forest)
- PyCharm as the development environment


## Project Workflow

1. Data Cleaning**
   - Removed unnecessary columns
   - Filtered rows for binary classification (`Fully Paid` vs `Charged Off`)
   - Encoded categorical features with `LabelEncoder`

2. Exploratory Data Analysis
   - Checked distributions of features
   - Handled missing values
   - Limited rows

