# loan_default_risk_prediction.py
# Author: Synthia Pial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# STEP 1: Load Dataset
# ---------------------------
df = pd.read_csv("accepted_2007_to_2018Q4.csv", low_memory=False)

# ---------------------------
# STEP 2: Select Useful Columns
# ---------------------------
cols = ['loan_amnt', 'term', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'loan_status']
df = df[cols]

# ---------------------------
# STEP 3: Drop Missing & Simplify Target
# ---------------------------
df.dropna(inplace=True)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]  # Binary classification
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# ---------------------------
# STEP 4: Encode Categorical Features
# ---------------------------
label_cols = ['term', 'emp_length', 'home_ownership', 'purpose']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------
# STEP 5: Train/Test Split
# ---------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# STEP 6: Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# STEP 7: Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------
# STEP 8: Feature Importance Plot
# ---------------------------
importance = model.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
