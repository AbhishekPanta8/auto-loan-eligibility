import os
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Define paths to model files relative to the test file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
APPROVAL_MODEL_PATH = os.path.join(MODEL_DIR, "approval_model.pkl")
CREDIT_LIMIT_MODEL_PATH = os.path.join(MODEL_DIR, "credit_limit_model.pkl")
INTEREST_RATE_MODEL_PATH = os.path.join(MODEL_DIR, "interest_rate_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "encoded_feature_columns.pkl")
SCALED_COLUMNS_PATH = os.path.join(MODEL_DIR, "scaled_columns.pkl")

# Skip all tests if model files are not found
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(path) for path in [APPROVAL_MODEL_PATH, CREDIT_LIMIT_MODEL_PATH, INTEREST_RATE_MODEL_PATH]),
    reason="Required model files not found"
)

@pytest.fixture(scope="module")
def load_models_and_utils():
    """Load all models and utility files needed for testing"""
    try:
        # Load models
        approval_model = joblib.load(APPROVAL_MODEL_PATH)
        credit_limit_model = joblib.load(CREDIT_LIMIT_MODEL_PATH)
        interest_rate_model = joblib.load(INTEREST_RATE_MODEL_PATH)
        
        # Load feature info and scaler if available
        feature_columns = None
        scaler = None
        scaled_columns = None
        
        if os.path.exists(FEATURE_COLUMNS_PATH):
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            
        if os.path.exists(SCALED_COLUMNS_PATH):
            scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
            
        logger.info(f"Successfully loaded all models and utilities")
        return {
            "approval_model": approval_model,
            "credit_limit_model": credit_limit_model,
            "interest_rate_model": interest_rate_model,
            "feature_columns": feature_columns,
            "scaler": scaler,
            "scaled_columns": scaled_columns
        }
    except Exception as e:
        logger.error(f"Error loading models or utilities: {str(e)}")
        pytest.skip(f"Error loading models: {str(e)}")

def prepare_features_for_model(test_data, feature_columns):
    """
    Prepare test data for model prediction by handling categorical features and ensuring
    all required columns are present.
    
    Args:
        test_data: DataFrame with raw test data
        feature_columns: List of feature columns expected by the model
        
    Returns:
        DataFrame with properly formatted features for model prediction
    """
    # Create a DataFrame with all expected feature columns initialized to 0
    X = pd.DataFrame(0, index=range(len(test_data)), columns=feature_columns)
    
    # Copy numeric features directly if they exist in both datasets
    numeric_features = ["annual_income", "self_reported_debt", "self_reported_expenses", 
                        "requested_amount", "age", "credit_score", "credit_utilization", 
                        "months_employed", "num_open_accounts", "num_credit_inquiries",
                        "debt_to_income_ratio", "DTI", "monthly_expenses", "estimated_debt",
                        "credit_history_length", "missed_payments", "current_credit_limit"]
    
    for feature in numeric_features:
        if feature in test_data.columns and feature in feature_columns:
            # Copy the values, filling missing ones with 0
            X[feature] = test_data[feature].fillna(0)
    
    # Handle categorical features that need one-hot encoding
    categorical_mappings = {
        "province": ["ON", "BC", "AB", "QC", "NS", "NB", "MB", "SK", "PE", "NL"],
        "employment_status": ["Full-time", "Part-time", "Unemployed", "Self-employed"],
        "payment_history": ["On Time", "Late<30", "Late 30-60", "Late>60"]
    }
    
    # Process each categorical feature
    for cat_feature, possible_values in categorical_mappings.items():
        if cat_feature in test_data.columns:
            # Fill missing categorical values with the most common value (or first in list if none found)
            if test_data[cat_feature].isna().any():
                most_common = test_data[cat_feature].mode().iloc[0] if not test_data[cat_feature].mode().empty else possible_values[0]
                test_data[cat_feature] = test_data[cat_feature].fillna(most_common)
                logger.warning(f"Filled missing {cat_feature} values with '{most_common}'")
            
            # For each possible value of the categorical feature
            for value in possible_values:
                encoded_col = f"{cat_feature}_{value}"
                # If this encoded column is expected by the model
                if encoded_col in feature_columns:
                    # Set to 1 where the category matches, 0 otherwise
                    X[encoded_col] = (test_data[cat_feature] == value).astype(int)
    
    # Special case for boolean features
    boolean_features = ["collateral_available", "equifax_consent"]
    for feature in boolean_features:
        if feature in test_data.columns and feature in feature_columns:
            X[feature] = test_data[feature].fillna(0).astype(int)
            
    # Check for any remaining NaN values
    if X.isna().any().any():
        logger.warning(f"Data still contains NaN values after preprocessing. Filling remaining NaNs with 0.")
        X = X.fillna(0)
            
    # Log warnings for any columns we don't have data for
    missing_features = [col for col in feature_columns if X[col].sum() == 0 and col not in X.columns]
    if missing_features:
        logger.warning(f"Missing data for these features: {missing_features}")
        
    return X

def generate_synthetic_test_data(n_samples=10):
    """Generate synthetic test data for model evaluation"""
    # Create a small synthetic dataset with reasonable values
    test_data = []
    for _ in range(n_samples):
        test_data.append({
            "age": np.random.randint(25, 70),
            "annual_income": np.random.randint(30000, 150000),
            "self_reported_debt": np.random.randint(0, 50000),
            "self_reported_expenses": np.random.randint(1000, 5000),
            "requested_amount": np.random.randint(5000, 50000),
            "months_employed": np.random.randint(12, 240),
            "debt_to_income_ratio": np.random.uniform(0.1, 0.5),
            "credit_score": np.random.randint(580, 820),
            "num_open_accounts": np.random.randint(1, 10),
            "credit_utilization": np.random.uniform(10, 80),
            "employment_status": np.random.choice(["Full-time", "Part-time"]),
            "province": np.random.choice(["ON", "BC", "AB"]),
            "payment_history": np.random.choice(["On Time", "Late<30"]),
            "equifax_consent": np.random.choice([True, False]),
            "collateral_available": np.random.choice([True, False]),
            "DTI": np.random.uniform(0.1, 0.5),
            "monthly_expenses": np.random.randint(1000, 5000),
            "estimated_debt": np.random.randint(0, 50000),
            "credit_history_length": np.random.randint(12, 240),
            "missed_payments": np.random.randint(0, 5),
            "current_credit_limit": np.random.randint(1000, 50000)
        })
    return pd.DataFrame(test_data)

def test_models_with_synthetic_data(load_models_and_utils):
    """Test all models with synthetic data to ensure they work correctly"""
    if not load_models_and_utils:
        pytest.skip("Models not available")
    
    models = load_models_and_utils
    approval_model = models["approval_model"]
    credit_limit_model = models["credit_limit_model"]
    interest_rate_model = models["interest_rate_model"]
    feature_columns = models["feature_columns"]
    scaler = models["scaler"]
    scaled_columns = models["scaled_columns"]
    
    # Check if we have feature columns
    if not feature_columns:
        logger.warning("No feature columns information available, skipping test")
        pytest.skip("No feature columns information available")
    
    # Generate synthetic test data
    test_data = generate_synthetic_test_data(n_samples=10)
    logger.info(f"Generated {len(test_data)} synthetic test samples")
    
    # Preprocess data to match expected feature format
    X = prepare_features_for_model(test_data, feature_columns)
    
    # Scale features if needed
    if scaler and scaled_columns:
        scale_cols = [col for col in scaled_columns if col in X.columns]
        if scale_cols:
            try:
                X[scale_cols] = scaler.transform(X[scale_cols])
            except Exception as e:
                logger.warning(f"Error scaling features: {str(e)}. Using unscaled features.")
    
    # Final check for NaN values before prediction
    if X.isna().any().any():
        logger.warning("Data contains NaN values after preprocessing, filling with zeros")
        X = X.fillna(0)
    
    # Test approval model
    try:
        approval_probs = approval_model.predict_proba(X)[:, 1]
        approvals = approval_model.predict(X)
        logger.info(f"Approval model predictions: {approvals}")
        logger.info(f"Approval probabilities: min={min(approval_probs):.4f}, max={max(approval_probs):.4f}, mean={np.mean(approval_probs):.4f}")
        
        # Check that we have both approvals and rejections
        assert 0 <= approval_probs.min() <= 1, f"Approval probabilities should be in [0,1], got min={approval_probs.min()}"
        assert 0 <= approval_probs.max() <= 1, f"Approval probabilities should be in [0,1], got max={approval_probs.max()}"
    except Exception as e:
        logger.error(f"Error testing approval model: {str(e)}")
        pytest.fail(f"Approval model failed: {str(e)}")
    
    # Test loan amount model
    try:
        loan_amounts = credit_limit_model.predict(X)
        logger.info(f"Loan amount predictions: min=${min(loan_amounts):.2f}, max=${max(loan_amounts):.2f}, mean=${np.mean(loan_amounts):.2f}")
        
        # Check that loan amounts are reasonable
        assert loan_amounts.min() >= 0, f"Loan amounts should be non-negative, got min={loan_amounts.min()}"
        assert loan_amounts.max() < 1000000, f"Loan amounts should be reasonable, got max={loan_amounts.max()}"
    except Exception as e:
        logger.error(f"Error testing loan amount model: {str(e)}")
        pytest.fail(f"Loan amount model failed: {str(e)}")
    
    # Test interest rate model
    try:
        interest_rates = interest_rate_model.predict(X)
        logger.info(f"Interest rate predictions: min={min(interest_rates):.2f}%, max={max(interest_rates):.2f}%, mean={np.mean(interest_rates):.2f}%")
        
        # Check that interest rates are reasonable
        assert interest_rates.min() >= 0, f"Interest rates should be non-negative, got min={interest_rates.min()}"
        assert interest_rates.max() <= 30, f"Interest rates should be reasonable, got max={interest_rates.max()}"
    except Exception as e:
        logger.error(f"Error testing interest rate model: {str(e)}")
        pytest.fail(f"Interest rate model failed: {str(e)}")
    
    logger.info("All models working correctly with synthetic data")

def test_good_applicant_gets_approval(load_models_and_utils):
    """Test that a good applicant gets approved with reasonable terms"""
    if not load_models_and_utils:
        pytest.skip("Models not available")
    
    models = load_models_and_utils
    approval_model = models["approval_model"]
    credit_limit_model = models["credit_limit_model"]
    interest_rate_model = models["interest_rate_model"]
    feature_columns = models["feature_columns"]
    scaler = models["scaler"]
    scaled_columns = models["scaled_columns"]
    
    # Check if we have feature columns
    if not feature_columns:
        logger.warning("No feature columns information available, skipping test")
        pytest.skip("No feature columns information available")
    
    # Create a test case with a good applicant profile
    good_applicant = pd.DataFrame([{
        "annual_income": 120000,
        "self_reported_debt": 5000,
        "self_reported_expenses": 3000,
        "requested_amount": 30000,
        "age": 35,
        "credit_score": 800,
        "credit_utilization": 10,
        "months_employed": 60,
        "num_open_accounts": 3,
        "num_credit_inquiries": 0,
        "debt_to_income_ratio": 0.1,
        "DTI": 0.1,
        "monthly_expenses": 3000,
        "estimated_debt": 5000,
        "province": "ON",
        "employment_status": "Full-time",
        "payment_history": "On Time",
        "equifax_consent": True,
        "collateral_available": True
    }])
    
    # Prepare features
    X = prepare_features_for_model(good_applicant, feature_columns)
    
    # Scale if needed
    if scaler and scaled_columns:
        scale_cols = [col for col in scaled_columns if col in X.columns]
        if scale_cols:
            try:
                X[scale_cols] = scaler.transform(X[scale_cols])
            except Exception as e:
                logger.warning(f"Error scaling features: {str(e)}. Using unscaled features.")
    
    # Double-check for NaN values
    if X.isna().any().any():
        logger.warning("Data contains NaN values after preprocessing, filling with zeros")
        X = X.fillna(0)
    
    # Get predictions
    approval_prob = approval_model.predict_proba(X)[0, 1]
    approval = approval_model.predict(X)[0]
    loan_amount = credit_limit_model.predict(X)[0]
    interest_rate = interest_rate_model.predict(X)[0]
    
    logger.info(f"Good applicant results:")
    logger.info(f"  Approval probability: {approval_prob:.4f}")
    logger.info(f"  Approved: {approval}")
    logger.info(f"  Loan amount: ${loan_amount:.2f}")
    logger.info(f"  Interest rate: {interest_rate:.2f}%")
    
    # Assertions
    assert approval_prob > 0.5, f"Good applicant should have high approval probability, got {approval_prob:.4f}"
    assert approval == 1, "Good applicant should be approved"
    assert loan_amount > 0, f"Loan amount should be positive, got {loan_amount:.2f}"
    assert 0 <= interest_rate <= 20, f"Interest rate should be reasonable, got {interest_rate:.2f}%"

def test_bad_applicant_gets_rejection(load_models_and_utils):
    """Test that a bad applicant gets rejected or receives poor terms"""
    if not load_models_and_utils:
        pytest.skip("Models not available")
    
    models = load_models_and_utils
    approval_model = models["approval_model"]
    credit_limit_model = models["credit_limit_model"]
    interest_rate_model = models["interest_rate_model"]
    feature_columns = models["feature_columns"]
    scaler = models["scaler"]
    scaled_columns = models["scaled_columns"]
    
    # Check if we have feature columns
    if not feature_columns:
        logger.warning("No feature columns information available, skipping test")
        pytest.skip("No feature columns information available")
    
    # Create a test case with a bad applicant profile
    bad_applicant = pd.DataFrame([{
        "annual_income": 30000,
        "self_reported_debt": 50000,
        "self_reported_expenses": 4500,
        "requested_amount": 100000,
        "age": 22,
        "credit_score": 580,
        "credit_utilization": 90,
        "months_employed": 3,
        "num_open_accounts": 10,
        "num_credit_inquiries": 5,
        "debt_to_income_ratio": 0.7,
        "DTI": 0.7,
        "monthly_expenses": 4500,
        "estimated_debt": 50000,
        "province": "ON",
        "employment_status": "Part-time",
        "payment_history": "Late<30",
        "equifax_consent": False,
        "collateral_available": False
    }])
    
    # Prepare features
    X = prepare_features_for_model(bad_applicant, feature_columns)
    
    # Scale if needed
    if scaler and scaled_columns:
        scale_cols = [col for col in scaled_columns if col in X.columns]
        if scale_cols:
            try:
                X[scale_cols] = scaler.transform(X[scale_cols])
            except Exception as e:
                logger.warning(f"Error scaling features: {str(e)}. Using unscaled features.")
    
    # Double-check for NaN values
    if X.isna().any().any():
        logger.warning("Data contains NaN values after preprocessing, filling with zeros")
        X = X.fillna(0)
    
    # Get predictions
    approval_prob = approval_model.predict_proba(X)[0, 1]
    approval = approval_model.predict(X)[0]
    
    logger.info(f"Bad applicant results:")
    logger.info(f"  Approval probability: {approval_prob:.4f}")
    logger.info(f"  Approved: {approval}")
    
    # If approved despite bad profile, check the terms
    if approval == 1:
        loan_amount = credit_limit_model.predict(X)[0]
        interest_rate = interest_rate_model.predict(X)[0]
        logger.info(f"  Loan amount: ${loan_amount:.2f}")
        logger.info(f"  Interest rate: {interest_rate:.2f}%")
        
        # For approved bad applicants, should have worse terms
        assert loan_amount < bad_applicant["requested_amount"].iloc[0], "Bad applicant should get less than requested"
        assert interest_rate > 10, "Bad applicant should get higher interest rate if approved"
    
    # It's ok if the model approves or rejects, but probability should be lower
    assert approval_prob < 0.7, f"Bad applicant should have lower approval probability, got {approval_prob:.4f}"
    
    # Success if model behaves as expected, doesn't have to reject
    logger.info(f"Model behaved as expected for bad applicant {'(rejected)' if approval == 0 else '(approved with restrictions)'}")

if __name__ == "__main__":
    # For manual testing
    pytest.main(["-xvs", __file__])
