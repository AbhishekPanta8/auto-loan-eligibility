import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
APPROVAL_MODEL_PATH = os.path.join(MODEL_DIR, "approval_model.pkl")
LOAN_AMOUNT_MODEL_PATH = os.path.join(MODEL_DIR, "credit_limit_model.pkl")
INTEREST_RATE_MODEL_PATH = os.path.join(MODEL_DIR, "interest_rate_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "encoded_feature_columns.pkl")
SCALED_COLUMNS_PATH = os.path.join(MODEL_DIR, "scaled_columns.pkl")

# Create figures directory for plots
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Skip all tests if models are not available
if not os.path.exists(APPROVAL_MODEL_PATH):
    pytest.skip("Model files not found, skipping advanced model tests", allow_module_level=True)

# Helper function to create a feature vector for predictions
def create_feature_vector(feature_values, feature_columns):
    """
    Creates a feature DataFrame with the required structure for model prediction.
    
    Args:
        feature_values: Dict with feature values
        feature_columns: List of all required feature columns
        
    Returns:
        pandas.DataFrame: Properly formatted feature vector
    """
    # Create a single-row DataFrame with all features initialized to 0
    X = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Fill in the values from feature_values
    for feature, value in feature_values.items():
        if feature in X.columns:
            X[feature] = value
        # Handle categorical features (one-hot encoded)
        elif '_' in feature:
            prefix, val = feature.split('_', 1)
            cat_cols = [col for col in X.columns if col.startswith(f"{prefix}_")]
            # Set all to 0, then set the specified one to 1
            for col in cat_cols:
                X[col] = 0
            if feature in X.columns:
                X[feature] = 1
    
    return X

# Helper function to make predictions
def predict_with_models(feature_values):
    """
    Make predictions using all three models (approval, loan amount, interest rate).
    
    Args:
        feature_values: Dict with feature values
        
    Returns:
        dict: Containing prediction results
    """
    try:
        # Load models
        approval_model = joblib.load(APPROVAL_MODEL_PATH)
        loan_model = joblib.load(LOAN_AMOUNT_MODEL_PATH)
        interest_model = joblib.load(INTEREST_RATE_MODEL_PATH)
        
        # Load feature columns and scaler
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        scaled_columns = joblib.load(SCALED_COLUMNS_PATH) if os.path.exists(SCALED_COLUMNS_PATH) else []
        
        # Create feature vector
        X = create_feature_vector(feature_values, feature_columns)
        
        # Scale features if needed
        if scaler is not None and len(scaled_columns) > 0:
            scale_cols = [col for col in scaled_columns if col in X.columns]
            if scale_cols:
                X[scale_cols] = scaler.transform(X[scale_cols])
        
        # Make predictions
        approval_prob = approval_model.predict_proba(X)[0, 1]
        approval = approval_prob >= 0.5
        
        # For approved loans, predict amount and rate
        amount = loan_model.predict(X)[0] if approval else 0.0
        rate = interest_model.predict(X)[0] if approval else 0.0
        
        return {
            "approval_probability": approval_prob,
            "approved": approval,
            "loan_amount": amount,
            "interest_rate": rate
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

def test_model_robustness_missing_values():
    """Test model robustness against missing values"""
    # Define a base good application
    base_features = {
        "annual_income": 100000.0,
        "self_reported_debt": 1000.0,
        "self_reported_expenses": 2500.0,
        "requested_amount": 25000.0,
        "age": 35,
        "credit_score": 780,
        "credit_utilization": 20.0,
        "months_employed": 60,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "total_credit_limit": 50000.0,
        "monthly_expenses": 3000.0,
        "estimated_debt": 500.0,
        "DTI": 0.2,
        "province_ON": 1,
        "employment_status_Full-time": 1,
        "payment_history_On Time": 1
    }
    
    # Fields to test with missing values
    fields_to_test = [
        "annual_income", 
        "credit_score", 
        "months_employed",
        "credit_utilization"
    ]
    
    results = []
    # Get baseline prediction
    baseline = predict_with_models(base_features)
    
    # Test removing each field and replacing with 0 (missing)
    for field in fields_to_test:
        test_features = base_features.copy()
        test_features[field] = 0  # Simulate missing value
        
        # Get prediction
        prediction = predict_with_models(test_features)
        
        # Calculate impact
        approval_impact = prediction["approval_probability"] - baseline["approval_probability"]
        
        results.append({
            "field": field,
            "baseline_approval": baseline["approval_probability"],
            "missing_approval": prediction["approval_probability"],
            "approval_impact": approval_impact,
            "still_approved": prediction["approved"]
        })
    
    # Log results
    logger.info("\nRobustness Test - Missing Values:")
    for r in results:
        logger.info(f"  Missing {r['field']}: Impact on approval probability: {r['approval_impact']:.4f}")
        logger.info(f"    Baseline: {r['baseline_approval']:.4f}, With missing: {r['missing_approval']:.4f}")
        logger.info(f"    Still approved: {r['still_approved']}")
    
    # Assert the model is reasonably robust - at least one field with missing value should
    # still result in approval (this is a simple test, may need adjusting)
    assert any(r["still_approved"] for r in results), "Model fails with any missing value"
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar([r["field"] for r in results], [abs(r["approval_impact"]) for r in results])
    plt.title("Impact of Missing Values on Approval Probability")
    plt.xlabel("Missing Field")
    plt.ylabel("Absolute Change in Approval Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "missing_values_impact.png"))

def test_model_feature_sensitivity():
    """Test how sensitive the model is to changes in key features"""
    # Load base good applicant profile
    base_features = {
        "annual_income": 80000.0,
        "self_reported_debt": 1000.0,
        "self_reported_expenses": 2500.0,
        "requested_amount": 20000.0,
        "age": 35,
        "credit_score": 700,
        "credit_utilization": 30.0,
        "months_employed": 60,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "total_credit_limit": 30000.0,
        "monthly_expenses": 2500.0,
        "estimated_debt": 500.0,
        "DTI": 0.2,
        "province_ON": 1,
        "employment_status_Full-time": 1,
        "payment_history_On Time": 1
    }
    
    # Test features to vary
    sensitivity_tests = [
        {"feature": "credit_score", "values": [450, 550, 600, 650, 700, 750, 800]},
        {"feature": "annual_income", "values": [30000, 50000, 70000, 90000, 110000, 130000]},
        {"feature": "credit_utilization", "values": [10, 30, 50, 70, 90]},
        {"feature": "DTI", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        {"feature": "months_employed", "values": [1, 6, 12, 24, 36, 60, 120]}
    ]
    
    # Results for plotting
    feature_results = {}
    
    for test in sensitivity_tests:
        feature = test["feature"]
        values = test["values"]
        results = []
        
        for value in values:
            # Create test features with just this value changed
            test_features = base_features.copy()
            test_features[feature] = value
            
            # Get model prediction
            prediction = predict_with_models(test_features)
            
            # Store result
            results.append({
                "value": value, 
                "approval_prob": prediction["approval_probability"],
                "approved": prediction["approved"],
                "loan_amount": prediction["loan_amount"],
                "interest_rate": prediction["interest_rate"]
            })
        
        # Store for plotting
        feature_results[feature] = results
        
        # Log results
        logger.info(f"\nSensitivity for {feature}:")
        for r in results:
            logger.info(f"  {feature}={r['value']}: approval_prob={r['approval_prob']:.4f}, approved={r['approved']}")
            if r['approved']:
                logger.info(f"    loan_amount=${r['loan_amount']:.2f}, interest_rate={r['interest_rate']:.2f}%")
        
        # Create plot for this feature
        plt.figure(figsize=(10, 6))
        plt.plot([r["value"] for r in results], [r["approval_prob"] for r in results], 'o-')
        plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)  # Decision threshold
        plt.title(f"Sensitivity of Approval Probability to {feature}")
        plt.xlabel(feature)
        plt.ylabel("Approval Probability")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, f"sensitivity_{feature}.png"))
        
        # If we have loan amounts, plot those too for approved cases
        if any(r["approved"] for r in results):
            plt.figure(figsize=(10, 6))
            approved_values = [r["value"] for r in results if r["approved"]]
            approved_amounts = [r["loan_amount"] for r in results if r["approved"]]
            plt.plot(approved_values, approved_amounts, 'o-')
            plt.title(f"Sensitivity of Loan Amount to {feature}")
            plt.xlabel(feature)
            plt.ylabel("Loan Amount ($)")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(FIGURES_DIR, f"sensitivity_{feature}_amount.png"))

def test_model_fairness():
    """
    Test model fairness across different demographic groups.
    This tests if the model gives similar results for different provinces, age groups, etc.
    """
    # Base good applicant profile
    base_features = {
        "annual_income": 80000.0,
        "self_reported_debt": 1000.0,
        "self_reported_expenses": 2500.0,
        "requested_amount": 20000.0,
        "age": 35,
        "credit_score": 700,
        "credit_utilization": 30.0,
        "months_employed": 60,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "total_credit_limit": 30000.0,
        "monthly_expenses": 2500.0,
        "estimated_debt": 500.0,
        "DTI": 0.2,
        "province_ON": 1,  # Default province is ON
        "province_BC": 0,
        "employment_status_Full-time": 1,
        "employment_status_Part-time": 0,
        "employment_status_Unemployed": 0,
        "payment_history_On Time": 1,
        "payment_history_Late<30": 0,
        "payment_history_Late 30-60": 0,
        "payment_history_Late>60": 0
    }
    
    # Test fairness for different provinces
    provinces = ["ON", "BC"]
    province_results = []
    
    for province in provinces:
        test_features = base_features.copy()
        # Reset all province flags
        for p in provinces:
            test_features[f"province_{p}"] = 0
        # Set current province
        test_features[f"province_{province}"] = 1
        
        # Get prediction
        prediction = predict_with_models(test_features)
        
        province_results.append({
            "province": province,
            "approval_prob": prediction["approval_probability"],
            "approved": prediction["approved"],
            "loan_amount": prediction["loan_amount"],
            "interest_rate": prediction["interest_rate"]
        })
    
    # Test fairness for different employment statuses
    employment_statuses = ["Full-time", "Part-time"]
    employment_results = []
    
    for status in employment_statuses:
        test_features = base_features.copy()
        # Reset all employment flags
        for s in ["Full-time", "Part-time", "Unemployed"]:
            test_features[f"employment_status_{s}"] = 0
        # Set current status
        test_features[f"employment_status_{status}"] = 1
        
        # Get prediction
        prediction = predict_with_models(test_features)
        
        employment_results.append({
            "status": status,
            "approval_prob": prediction["approval_probability"],
            "approved": prediction["approved"],
            "loan_amount": prediction["loan_amount"],
            "interest_rate": prediction["interest_rate"]
        })
    
    # Test fairness for different age groups
    age_groups = [22, 35, 50, 65]
    age_results = []
    
    for age in age_groups:
        test_features = base_features.copy()
        test_features["age"] = age
        
        # Get prediction
        prediction = predict_with_models(test_features)
        
        age_results.append({
            "age": age,
            "approval_prob": prediction["approval_probability"],
            "approved": prediction["approved"],
            "loan_amount": prediction["loan_amount"],
            "interest_rate": prediction["interest_rate"]
        })
    
    # Log results
    logger.info("\nFairness Test Results:")
    
    logger.info("\nProvince Fairness:")
    for r in province_results:
        logger.info(f"  Province {r['province']}: approval_prob={r['approval_prob']:.4f}, approved={r['approved']}")
        if r['approved']:
            logger.info(f"    loan_amount=${r['loan_amount']:.2f}, interest_rate={r['interest_rate']:.2f}%")
    
    logger.info("\nEmployment Status Fairness:")
    for r in employment_results:
        logger.info(f"  Status {r['status']}: approval_prob={r['approval_prob']:.4f}, approved={r['approved']}")
        if r['approved']:
            logger.info(f"    loan_amount=${r['loan_amount']:.2f}, interest_rate={r['interest_rate']:.2f}%")
    
    logger.info("\nAge Group Fairness:")
    for r in age_results:
        logger.info(f"  Age {r['age']}: approval_prob={r['approval_prob']:.4f}, approved={r['approved']}")
        if r['approved']:
            logger.info(f"    loan_amount=${r['loan_amount']:.2f}, interest_rate={r['interest_rate']:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Province fairness
    plt.subplot(1, 3, 1)
    plt.bar([r["province"] for r in province_results], [r["approval_prob"] for r in province_results])
    plt.title("Approval by Province")
    plt.ylabel("Approval Probability")
    plt.ylim(0, 1)
    
    # Employment fairness
    plt.subplot(1, 3, 2)
    plt.bar([r["status"] for r in employment_results], [r["approval_prob"] for r in employment_results])
    plt.title("Approval by Employment Status")
    plt.ylim(0, 1)
    
    # Age fairness
    plt.subplot(1, 3, 3)
    plt.bar([str(r["age"]) for r in age_results], [r["approval_prob"] for r in age_results])
    plt.title("Approval by Age")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fairness_analysis.png"))
    
    # Check for fairness issues (simple check, can be enhanced)
    # For provinces, there should be similar approval rates
    province_approvals = [r["approval_prob"] for r in province_results]
    province_difference = max(province_approvals) - min(province_approvals)
    assert province_difference < 0.2, f"Large approval difference between provinces: {province_difference:.4f}"
    
    # For employment, full-time should not be dramatically better than part-time
    if len(employment_results) >= 2:
        ft_approval = next(r["approval_prob"] for r in employment_results if r["status"] == "Full-time")
        pt_approval = next(r["approval_prob"] for r in employment_results if r["status"] == "Part-time")
        employment_diff = ft_approval - pt_approval
        # Don't fail the test, but log a warning if large difference
        if employment_diff > 0.3:
            logger.warning(f"Large approval difference between employment statuses: {employment_diff:.4f}")

def test_model_stability_outliers():
    """Test model stability with outlier values"""
    # Base good applicant profile
    base_features = {
        "annual_income": 80000.0,
        "self_reported_debt": 1000.0,
        "self_reported_expenses": 2500.0,
        "requested_amount": 20000.0,
        "age": 35,
        "credit_score": 700,
        "credit_utilization": 30.0,
        "months_employed": 60,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "total_credit_limit": 30000.0,
        "monthly_expenses": 2500.0,
        "estimated_debt": 500.0,
        "DTI": 0.2,
        "province_ON": 1,
        "employment_status_Full-time": 1,
        "payment_history_On Time": 1
    }
    
    # Test with outlier values
    outlier_tests = [
        {"feature": "annual_income", "value": 1000000.0},  # Very high income
        {"feature": "age", "value": 100},                 # Very old
        {"feature": "months_employed", "value": 600},     # 50 years employed
        {"feature": "credit_score", "value": 900},        # Perfect credit
        {"feature": "num_open_accounts", "value": 50},    # Many accounts
        {"feature": "num_credit_inquiries", "value": 30}  # Many inquiries
    ]
    
    results = []
    # Get baseline prediction
    baseline = predict_with_models(base_features)
    
    for test in outlier_tests:
        feature = test["feature"]
        value = test["value"]
        
        # Create test features with outlier value
        test_features = base_features.copy()
        test_features[feature] = value
        
        # Get prediction
        prediction = predict_with_models(test_features)
        
        results.append({
            "feature": feature,
            "outlier_value": value,
            "baseline_approval": baseline["approval_probability"],
            "outlier_approval": prediction["approval_probability"],
            "approval_diff": prediction["approval_probability"] - baseline["approval_probability"],
            "approved": prediction["approved"]
        })
    
    # Log results
    logger.info("\nModel Stability with Outliers:")
    for r in results:
        logger.info(f"  {r['feature']}={r['outlier_value']}: approval_diff={r['approval_diff']:.4f}")
        logger.info(f"    Baseline: {r['baseline_approval']:.4f}, With outlier: {r['outlier_approval']:.4f}")
        logger.info(f"    Approved: {r['approved']}")
    
    # Check model doesn't fail with outliers
    assert all(not pd.isna(r["outlier_approval"]) for r in results), "Model failed with outlier values"
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar([r["feature"] for r in results], [r["approval_diff"] for r in results])
    plt.title("Impact of Outlier Values on Approval Probability")
    plt.xlabel("Feature with Outlier Value")
    plt.ylabel("Change in Approval Probability")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "outlier_impact.png"))

if __name__ == "__main__":
    # For manual testing
    test_model_robustness_missing_values()
    test_model_feature_sensitivity()
    test_model_fairness()
    test_model_stability_outliers() 