import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

@pytest.fixture(scope="module")
def model_and_test_data():
    # Load your trained XGBoost model
    clf_model = joblib.load("models/german_credit_xgb_latest.pkl")
    
    # Load the scaler for transforming test data
    scaler = joblib.load("models/scaler_latest.pkl")
    
    # Load test data
    X_test = np.load("datasets/splits/X_test.npy")
    y_test = np.load("datasets/splits/y_test.npy")
    
    # Ensure X_test is a DataFrame with correct column names after feature selection
    feature_names = [
        "Status_checking", "Duration", "Credit_history", "Purpose", "Credit_amount",
        "Savings_account", "Employment", "Installment_rate", "Personal_status_sex",
        "Other_debtors", "Property", "Age", "Other_installments",
        "Existing_credits", "Job", "Telephone", "Foreign_worker"
    ]  # Ensure this matches features used in training
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)  # Convert to DataFrame with correct column names
    print(f"✅ Model expects {clf_model.n_features_in_} features")
    print(f"✅ X_test shape before scaling: {X_test_df.shape}")

    return clf_model, X_test_df, y_test, scaler

def test_classification_performance(model_and_test_data):
    clf_model, X_test_df, y_test, scaler = model_and_test_data

    # Apply scaling with correct feature names
    X_test_scaled = scaler.transform(X_test_df)
    print(f"✅ X_test shape after scaling: {X_test_scaled.shape}")

    # Run prediction
    y_pred = clf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n📊 Test Results:")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall: {rec:.4f}")

    assert acc > 0.75, f"Expected accuracy > 0.75 but got {acc:.4f}"
    assert prec > 0.70, f"Expected precision > 0.70 but got {prec:.4f}"
    assert rec > 0.65, f"Expected recall > 0.65 but got {rec:.4f}"
