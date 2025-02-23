import pandas as pd
import numpy as np
import os
import joblib
import datetime
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from time import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 📌 Step 1: Load the Dataset
DATA_PATH = "../datasets/german.data-numeric"  # Adjust path if needed

# Load the dataset (no headers in original, so add column names)
columns = [
    "Status_checking", "Duration", "Credit_history", "Purpose", "Credit_amount",
    "Savings_account", "Employment", "Installment_rate", "Personal_status_sex",
    "Other_debtors", "Present_residence", "Property", "Age", "Other_installments",
    "Housing", "Existing_credits", "Job", "People_liable", "Telephone",
    "Foreign_worker", "Loan_Approved"
]

df = pd.read_csv(DATA_PATH, sep='\s+', header=None, names=columns)  # Fix for FutureWarning

print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
print(df.head())

# 📌 Step 2: Define Features (X) and Target (y)
y = df["Loan_Approved"]  # Target variable (1 = good credit, 2 = bad credit)
X = df.drop(columns=["Loan_Approved"])

# Convert target to binary classification (1 = Approved, 0 = Denied)
y = (y == 1).astype(int)

# 📌 Step 3: Normalize Numerical Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Step 3.1: Drop Low-Importance Features **Before Scaling**
drop_features = ["People_liable", "Housing", "Present_residence"]
X.drop(columns=drop_features, inplace=True)  # Modify X directly

# 📌 Step 3.2: Normalize Numerical Features After Feature Selection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Now this matches the correct feature set

# 📌 Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Step 4.1: Save Train/Test Splits for Future Use
DATA_SPLIT_DIR = "../datasets/splits/"
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)  # Ensure directory exists

# Save as NumPy arrays
np.save(os.path.join(DATA_SPLIT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_SPLIT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_SPLIT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_SPLIT_DIR, "y_test.npy"), y_test)

print(f"\n✅ Train/Test splits saved successfully in: {DATA_SPLIT_DIR}")

# 📌 Step 5: Perform Grid Search for XGBoost Model Optimization
param_grid_xgb = {
    'n_estimators': [100, 300, 500, 1000],  # More estimators for better learning
    'max_depth': [3, 5, 7, 9],  # Controls model complexity
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Adjusts how quickly the model learns
    'min_child_weight': [1, 3, 5, 7],  # Helps prevent overfitting
    'gamma': [0, 0.1, 0.3, 0.5],  # Controls how aggressively to split nodes
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Controls randomness in training
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Similar to subsample but for features
    'scale_pos_weight': [0.6, 0.8, 1.0],  # Handles class imbalance
}

xgb_model = XGBClassifier(random_state=42)  # Initialize model

total_combinations = np.prod([len(v) for v in param_grid_xgb.values()])

# 📌 Run Grid Search with Progress Tracking
print("\n🔎 Running GridSearch for XGBoost hyperparameter tuning...")

start_time = time()  # Track start time

grid_search = GridSearchCV(
    xgb_model, param_grid_xgb, scoring='accuracy', cv=5, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Retrieve best model from GridSearch
clf_xgb = grid_search.best_estimator_

# 📌 Display Best Parameters
print(f"✅ Best XGBoost parameters: {grid_search.best_params_}")

# 📌 Show total training time
elapsed_time = time() - start_time
print(f"⏳ Training completed in {elapsed_time:.2f} seconds")

# 📌 Step 6: Evaluate Model Performance
y_pred_xgb = clf_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("\n📊 Optimized XGBoost Model Evaluation:")
print(f"✅ Best XGBoost Accuracy: {accuracy_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb))

# 📌 Step 7: Save Model with Versioning
MODEL_DIR = "../models/"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure directory exists

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
xgb_model_filename = f"german_credit_xgb_{timestamp}.pkl"
scaler_filename = f"scaler_{timestamp}.pkl"

joblib.dump(clf_xgb, os.path.join(MODEL_DIR, xgb_model_filename))
joblib.dump(scaler, os.path.join(MODEL_DIR, scaler_filename))

# 📌 Step 7.1: Maintain a "latest" model file for easy reference
latest_xgb_model_path = os.path.join(MODEL_DIR, "german_credit_xgb_latest.pkl")
latest_scaler_path = os.path.join(MODEL_DIR, "scaler_latest.pkl")

# Remove previous "latest" files if they exist
for path in [latest_xgb_model_path, latest_scaler_path]:
    if os.path.exists(path):
        os.remove(path)

shutil.copy(os.path.join(MODEL_DIR, xgb_model_filename), latest_xgb_model_path)
shutil.copy(os.path.join(MODEL_DIR, scaler_filename), latest_scaler_path)

print(f"\n✅ XGBoost Model saved as: {xgb_model_filename}")
print(f"✅ Latest model references updated")

# 📌 Step 8: Feature Importance Analysis
feature_names = list(X.columns)

xgb_importances = clf_xgb.feature_importances_
sorted_indices_xgb = np.argsort(xgb_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.barh([feature_names[i] for i in sorted_indices_xgb], xgb_importances[sorted_indices_xgb])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance in Loan Approval Model")
plt.gca().invert_yaxis()
plt.show()
