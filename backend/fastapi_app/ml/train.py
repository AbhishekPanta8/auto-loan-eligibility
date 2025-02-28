''' OLD SYNTHETIC DATA DO NOT USE !! '''

'''
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import os

# Step 1: Generate Synthetic Dataset
np.random.seed(42)

n_samples = 5000  # Number of synthetic customers

# Generating random financial attributes
data = {
    "Age": np.random.randint(21, 65, n_samples),
    "Employment_Status": np.random.choice(["Salaried", "Self-employed", "Unemployed"], n_samples, p=[0.7, 0.2, 0.1]),
    "Annual_Income": np.random.randint(20000, 200000, n_samples),
    "Existing_Debt": np.random.randint(0, 100000, n_samples),
    "Debt_to_Income_Ratio": np.round(np.random.uniform(0.05, 0.6, n_samples), 2),
    "Credit_Score": np.random.randint(500, 850, n_samples),
    "Credit_History_Length": np.random.randint(1, 25, n_samples),
    "Missed_Payments": np.random.randint(0, 10, n_samples),
    "Loan_Amount_Requested": np.random.randint(5000, 50000, n_samples),
    "Preferred_Term_Months": np.random.choice([12, 24, 36, 48, 60], n_samples),
    "Collateral_Available": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),  # 1 = Yes, 0 = No
}

df = pd.DataFrame(data)

# Encode categorical variables
df["Employment_Status"] = df["Employment_Status"].map({"Salaried": 2, "Self-employed": 1, "Unemployed": 0})

# Generate target variables
df["Loan_Approved"] = ((df["Credit_Score"] > 650) & (df["Debt_to_Income_Ratio"] < 0.4)).astype(int)
df["Estimated_Loan_Amount"] = df["Loan_Amount_Requested"] * (np.random.uniform(0.7, 1.2, n_samples))
df["Estimated_Interest_Rate"] = np.round(np.random.uniform(3, 15, n_samples), 2)

# Step 2: Preprocessing
X = df.drop(columns=["Loan_Approved", "Estimated_Loan_Amount", "Estimated_Interest_Rate"])
y_class = df["Loan_Approved"]
y_reg_loan = df["Estimated_Loan_Amount"]
y_reg_interest = df["Estimated_Interest_Rate"]

X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_loan_train, y_reg_loan_test = train_test_split(X, y_reg_loan, test_size=0.2, random_state=42)
_, _, y_reg_interest_train, y_reg_interest_test = train_test_split(X, y_reg_interest, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_class_train)

reg_loan = RandomForestRegressor(n_estimators=100, random_state=42)
reg_loan.fit(X_train_scaled, y_reg_loan_train)

reg_interest = RandomForestRegressor(n_estimators=100, random_state=42)
reg_interest.fit(X_train_scaled, y_reg_interest_train)

# Step 4: Save Trained Models
model_dir = "../models/"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, "loan_approval_model.pkl"))
joblib.dump(reg_loan, os.path.join(model_dir, "loan_amount_model.pkl"))
joblib.dump(reg_interest, os.path.join(model_dir, "interest_rate_model.pkl"))

print("Models saved successfully in:", model_dir)
'''