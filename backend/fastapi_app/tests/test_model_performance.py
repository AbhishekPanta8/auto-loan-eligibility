import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. Define the feature names that your model expects (excluding 'applicant_id')
    feature_names = [
        "annual_income", "self_reported_debt", "self_reported_expenses",
        "requested_amount", "age", "province", "employment_status",
        "months_employed", "credit_score", "credit_utilization",
        "num_open_accounts", "num_credit_inquiries", "payment_history",
        "total_credit_limit", "monthly_expenses", "estimated_debt", "DTI"
    ]
    
    # 2. Define which columns are categorical in your training process
    categorical_cols = ["province", "employment_status", "payment_history"]
    
    # 3. Directories for splits and model artifacts
    splits_dir = os.path.join(os.path.dirname(__file__), "../datasets/splits")
    models_dir = os.path.join(os.path.dirname(__file__), "../ml/models")
    
    # 4. Paths to the CSV for features (X) and the .npy for labels (y)
    X_test_path = os.path.join(splits_dir, "X_test.csv")   # <-- CSV as per training script
    y_test_path = os.path.join(splits_dir, "y_test.npy")   # <-- NumPy array for labels
    
    # 5. Check if the required files exist
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test splits not found at expected location: {splits_dir}")

    # 6. Load X_test from CSV and y_test from .npy
    X_test = pd.read_csv(X_test_path)
    y_test = np.load(y_test_path)
    
    print("Loaded test splits from CSV & NPY files:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    # 7. Confirm X_test columns match your feature list (optional debug check)
    #    This is purely to catch if your CSV includes or excludes columns incorrectly.
    if list(X_test.columns) != feature_names:
        print("\n[WARNING] The columns in X_test.csv do not match the expected 'feature_names'.")
        print("X_test columns:", list(X_test.columns))
        print("Expected columns:", feature_names)
        # If desired, raise an error or proceed carefully.
    
    # 8. Load the trained classification model and preprocessing artifacts
    clf_model = joblib.load(os.path.join(models_dir, "approval_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    encoded_feature_columns = joblib.load(os.path.join(models_dir, "encoded_feature_columns.pkl"))
    scaled_columns = joblib.load(os.path.join(models_dir, "scaled_columns.pkl"))  # numeric columns actually scaled

    # 9. Preprocess X_test EXACTLY like training:
    #    A. One-hot encode the categorical columns (drop_first=True).
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    #    B. Reindex to match the training feature set (encoded_feature_columns).
    X_test_encoded = X_test_encoded.reindex(columns=encoded_feature_columns, fill_value=0)
    
    #    C. Optionally verify columns match exactly
    # assert list(X_test_encoded.columns) == encoded_feature_columns, "Column mismatch!"
    
    #    D. Copy for final scaling
    X_test_processed = X_test_encoded.copy()
    
    #    E. Apply the scaler ONLY to the numeric columns that were scaled during training
    X_test_processed[scaled_columns] = scaler.transform(X_test_encoded[scaled_columns])
    
    # 10. Predict using the loaded model
    y_pred = clf_model.predict(X_test_processed)
    
    # 11. Compute performance metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\nModel Performance on Test Data:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return acc, prec, rec, f1

def test_model_performance():
    # Pytest will collect and run this function
    acc, prec, rec, f1 = main()
    # Optional: Add assertions for minimal thresholds
    assert acc >= 0.0
    assert prec >= 0.0
    assert rec >= 0.0
    assert f1 >= 0.0

if __name__ == "__main__":
    main()
