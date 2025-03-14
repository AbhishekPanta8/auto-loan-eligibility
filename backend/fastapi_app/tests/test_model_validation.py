import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
import joblib
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model performance thresholds
MIN_APPROVAL_ACCURACY = 0.80
MIN_APPROVAL_PRECISION = 0.75
MIN_APPROVAL_RECALL = 0.70
MAX_LOAN_AMOUNT_MAE = 5000  # Max mean absolute error for loan amount
MIN_LOAN_AMOUNT_R2 = 0.50   # Min R² for loan amount model
MAX_INTEREST_RATE_MAE = 2.0  # Max mean absolute error for interest rate
MIN_INTEREST_RATE_R2 = 0.50  # Min R² for interest rate model

# Paths to model files - adjust as needed for your project
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
APPROVAL_MODEL_PATH = os.path.join(MODEL_DIR, "approval_model.pkl")
LOAN_AMOUNT_MODEL_PATH = os.path.join(MODEL_DIR, "credit_limit_model.pkl")
INTEREST_RATE_MODEL_PATH = os.path.join(MODEL_DIR, "interest_rate_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "encoded_feature_columns.pkl")
SCALED_COLUMNS_PATH = os.path.join(MODEL_DIR, "scaled_columns.pkl")

# Path to validation data
VALIDATION_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "data", "validation_data.csv")

def load_validation_data():
    """
    Load validation dataset for model testing.
    Returns X (features) and y (targets) for each model.
    """
    try:
        # Try to load the validation dataset
        df = pd.read_csv(VALIDATION_DATA_PATH)
        logger.info(f"Loaded validation data with {len(df)} records")
        
        # If validation data isn't available, create a synthetic dataset from the synthetic loan applications
        if len(df) == 0:
            synthetic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "..", "datasets", "data", "synthetic_loan_applications.csv")
            if os.path.exists(synthetic_path):
                df = pd.read_csv(synthetic_path)
                # Take 20% of the data for validation
                df = df.sample(frac=0.2, random_state=42)
                logger.info(f"Using {len(df)} synthetic records for validation")
            else:
                logger.error("No validation data found and no synthetic data available")
                return None, None, None, None, None, None
        
        # Load the feature columns we need
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        
        # Prepare features (X) - depends on your feature encoding
        # Here we assume all required columns are in the validation data
        # This is a simplified version - adjust based on your actual preprocessing
        
        # Check which columns we actually have in the data
        available_cols = [col for col in feature_columns if col in df.columns]
        missing_cols = [col for col in feature_columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} columns in validation data: {missing_cols[:5]}...")
            
            # Handle categorical columns that might be missing (one-hot encoded)
            for col in missing_cols:
                if '_' in col:  # Likely a one-hot encoded column
                    df[col] = 0  # Set to 0 (not present)
                else:
                    # For continuous features, we'll impute with mean or set to 0
                    df[col] = 0
        
        # Get features
        X = df[feature_columns].copy()
        
        # Fill any missing values (simplified approach)
        X = X.fillna(0)
        
        # Target variables
        y_approval = df['approved'].astype(int) if 'approved' in df.columns else None
        y_amount = df['approved_amount'] if 'approved_amount' in df.columns else None
        y_interest = df['interest_rate'] if 'interest_rate' in df.columns else None
        
        if y_approval is None or y_amount is None or y_interest is None:
            logger.error("Validation data missing target columns")
            raise ValueError("Validation data is missing required target columns")
            
        return X, y_approval, y_amount, y_interest, df, feature_columns
    
    except Exception as e:
        logger.error(f"Error loading validation data: {str(e)}")
        raise

@pytest.mark.skipif(not os.path.exists(APPROVAL_MODEL_PATH), 
                   reason="Approval model file not found")
def test_approval_model_performance():
    """Test the approval classification model performance metrics"""
    try:
        # Load the model
        model = joblib.load(APPROVAL_MODEL_PATH)
        logger.info(f"Loaded approval model: {type(model).__name__}")
        
        # Load validation data
        X, y_approval, _, _, _, _ = load_validation_data()
        
        # Load scaler if needed
        scaler = None
        scaled_columns = []
        if os.path.exists(SCALER_PATH) and os.path.exists(SCALED_COLUMNS_PATH):
            scaler = joblib.load(SCALER_PATH)
            scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
        
        # Scale features if needed
        if scaler is not None and len(scaled_columns) > 0:
            # Only scale the columns that need scaling
            X_scaled = X.copy()
            scale_cols = [col for col in scaled_columns if col in X.columns]
            if scale_cols:
                X_scaled[scale_cols] = scaler.transform(X[scale_cols])
            X = X_scaled
        
        # Make predictions (get probabilities and convert to binary)
        y_prob = model.predict_proba(X)
        y_pred = (y_prob[:, 1] >= 0.5).astype(int)  # Assume threshold of 0.5
        
        # Calculate metrics
        accuracy = accuracy_score(y_approval, y_pred)
        precision = precision_score(y_approval, y_pred, zero_division=0)
        recall = recall_score(y_approval, y_pred, zero_division=0)
        
        logger.info(f"Approval Model Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f} (threshold: {MIN_APPROVAL_ACCURACY})")
        logger.info(f"  Precision: {precision:.4f} (threshold: {MIN_APPROVAL_PRECISION})")
        logger.info(f"  Recall:    {recall:.4f} (threshold: {MIN_APPROVAL_RECALL})")
        
        # Assert model meets minimum performance
        assert accuracy >= MIN_APPROVAL_ACCURACY, f"Accuracy {accuracy} below threshold {MIN_APPROVAL_ACCURACY}"
        assert precision >= MIN_APPROVAL_PRECISION, f"Precision {precision} below threshold {MIN_APPROVAL_PRECISION}"
        assert recall >= MIN_APPROVAL_RECALL, f"Recall {recall} below threshold {MIN_APPROVAL_RECALL}"
        
    except FileNotFoundError:
        pytest.skip("Model files not found, skipping test")
    except Exception as e:
        logger.error(f"Error testing approval model: {str(e)}")
        raise

@pytest.mark.skipif(not os.path.exists(LOAN_AMOUNT_MODEL_PATH), 
                   reason="Loan amount model file not found")
def test_loan_amount_model_performance():
    """Test the loan amount regression model performance metrics"""
    try:
        # Load the model
        model = joblib.load(LOAN_AMOUNT_MODEL_PATH)
        logger.info(f"Loaded loan amount model: {type(model).__name__}")
        
        # Load validation data
        X, _, y_amount, _, df, _ = load_validation_data()
        
        # Filter to only approved loans for amount prediction
        if 'approved' in df.columns:
            approved_idx = df['approved'] == 1
            X_approved = X[approved_idx].copy()
            y_amount_approved = y_amount[approved_idx]
        else:
            X_approved = X
            y_amount_approved = y_amount
        
        if len(X_approved) == 0:
            pytest.skip("No approved loans in validation data")
        
        # Load scaler if needed
        scaler = None
        scaled_columns = []
        if os.path.exists(SCALER_PATH) and os.path.exists(SCALED_COLUMNS_PATH):
            scaler = joblib.load(SCALER_PATH)
            scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
        
        # Scale features if needed
        if scaler is not None and len(scaled_columns) > 0:
            X_scaled = X_approved.copy()
            scale_cols = [col for col in scaled_columns if col in X_approved.columns]
            if scale_cols:
                X_scaled[scale_cols] = scaler.transform(X_approved[scale_cols])
            X_approved = X_scaled
        
        # Make predictions
        y_pred = model.predict(X_approved)
        
        # Calculate metrics
        mae = mean_absolute_error(y_amount_approved, y_pred)
        r2 = r2_score(y_amount_approved, y_pred)
        
        logger.info(f"Loan Amount Model Metrics:")
        logger.info(f"  MAE: ${mae:.2f} (threshold: ${MAX_LOAN_AMOUNT_MAE:.2f})")
        logger.info(f"  R²:  {r2:.4f} (threshold: {MIN_LOAN_AMOUNT_R2})")
        
        # Assert model meets minimum performance
        assert mae <= MAX_LOAN_AMOUNT_MAE, f"MAE ${mae:.2f} above threshold ${MAX_LOAN_AMOUNT_MAE:.2f}"
        assert r2 >= MIN_LOAN_AMOUNT_R2, f"R² {r2:.4f} below threshold {MIN_LOAN_AMOUNT_R2}"
        
    except FileNotFoundError:
        pytest.skip("Model files not found, skipping test")
    except Exception as e:
        logger.error(f"Error testing loan amount model: {str(e)}")
        raise

@pytest.mark.skipif(not os.path.exists(INTEREST_RATE_MODEL_PATH), 
                    reason="Interest rate model file not found")
def test_interest_rate_model_performance():
    """Test the interest rate regression model performance metrics"""
    try:
        # Load the model
        model = joblib.load(INTEREST_RATE_MODEL_PATH)
        logger.info(f"Loaded interest rate model: {type(model).__name__}")
        
        # Load validation data
        X, _, _, y_interest, df, _ = load_validation_data()
        
        # Filter to only approved loans for interest rate prediction
        if 'approved' in df.columns:
            approved_idx = df['approved'] == 1
            X_approved = X[approved_idx].copy()
            y_interest_approved = y_interest[approved_idx]
        else:
            X_approved = X
            y_interest_approved = y_interest
        
        if len(X_approved) == 0:
            pytest.skip("No approved loans in validation data")
        
        # Load scaler if needed
        scaler = None
        scaled_columns = []
        if os.path.exists(SCALER_PATH) and os.path.exists(SCALED_COLUMNS_PATH):
            scaler = joblib.load(SCALER_PATH)
            scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
        
        # Scale features if needed
        if scaler is not None and len(scaled_columns) > 0:
            X_scaled = X_approved.copy()
            scale_cols = [col for col in scaled_columns if col in X_approved.columns]
            if scale_cols:
                X_scaled[scale_cols] = scaler.transform(X_approved[scale_cols])
            X_approved = X_scaled
        
        # Make predictions
        y_pred = model.predict(X_approved)
        
        # Calculate metrics
        mae = mean_absolute_error(y_interest_approved, y_pred)
        r2 = r2_score(y_interest_approved, y_pred)
        
        logger.info(f"Interest Rate Model Metrics:")
        logger.info(f"  MAE: {mae:.4f}% (threshold: {MAX_INTEREST_RATE_MAE:.4f}%)")
        logger.info(f"  R²:  {r2:.4f} (threshold: {MIN_INTEREST_RATE_R2})")
        
        # Assert model meets minimum performance
        assert mae <= MAX_INTEREST_RATE_MAE, f"MAE {mae:.4f}% above threshold {MAX_INTEREST_RATE_MAE:.4f}%"
        assert r2 >= MIN_INTEREST_RATE_R2, f"R² {r2:.4f} below threshold {MIN_INTEREST_RATE_R2}"
        
    except FileNotFoundError:
        pytest.skip("Model files not found, skipping test")
    except Exception as e:
        logger.error(f"Error testing interest rate model: {str(e)}")
        raise

def test_specific_applicant_predictions():
    """
    Test model predictions for specific applicant profiles to ensure they're reasonable.
    This checks that models are properly loaded and give sensible outputs.
    """
    try:
        # Load all models
        if not (os.path.exists(APPROVAL_MODEL_PATH) and 
                os.path.exists(LOAN_AMOUNT_MODEL_PATH) and 
                os.path.exists(INTEREST_RATE_MODEL_PATH)):
            pytest.skip("One or more model files not found")
            
        approval_model = joblib.load(APPROVAL_MODEL_PATH)
        loan_model = joblib.load(LOAN_AMOUNT_MODEL_PATH)
        interest_model = joblib.load(INTEREST_RATE_MODEL_PATH)
        
        # Load feature columns and scaler
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        scaler = None
        scaled_columns = []
        if os.path.exists(SCALER_PATH) and os.path.exists(SCALED_COLUMNS_PATH):
            scaler = joblib.load(SCALER_PATH)
            scaled_columns = joblib.load(SCALED_COLUMNS_PATH)
        
        # Define test profiles
        test_profiles = [
            {
                "name": "Excellent applicant",
                "features": {
                    "annual_income": 120000.0,
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
                    "province_BC": 0,
                    "employment_status_Full-time": 1,
                    "employment_status_Part-time": 0,
                    "employment_status_Unemployed": 0,
                    "payment_history_On Time": 1,
                    "payment_history_Late<30": 0,
                    "payment_history_Late 30-60": 0,
                    "payment_history_Late>60": 0
                },
                "expected_approval": True,
                "min_amount": 10000.0,
                "max_amount": 50000.0,
                "min_rate": 3.0,
                "max_rate": 7.0
            },
            {
                "name": "Poor applicant",
                "features": {
                    "annual_income": 40000.0,
                    "self_reported_debt": 2000.0,
                    "self_reported_expenses": 3000.0,
                    "requested_amount": 20000.0,
                    "age": 45,
                    "credit_score": 450,
                    "credit_utilization": 70.0,
                    "months_employed": 12,
                    "num_open_accounts": 5,
                    "num_credit_inquiries": 6,
                    "total_credit_limit": 10000.0,
                    "monthly_expenses": 3500.0,
                    "estimated_debt": 1500.0,
                    "DTI": 0.5,
                    "province_ON": 1,
                    "province_BC": 0,
                    "employment_status_Full-time": 1,
                    "employment_status_Part-time": 0,
                    "employment_status_Unemployed": 0,
                    "payment_history_On Time": 0,
                    "payment_history_Late<30": 0,
                    "payment_history_Late 30-60": 0,
                    "payment_history_Late>60": 1
                },
                "expected_approval": False,
                "min_amount": 0.0,
                "max_amount": 5000.0,
                "min_rate": 10.0,
                "max_rate": 15.0
            }
        ]
        
        # Test each profile
        for profile in test_profiles:
            logger.info(f"Testing profile: {profile['name']}")
            
            # Create feature vector
            X = pd.DataFrame([profile["features"]])
            X = X.reindex(columns=feature_columns, fill_value=0)
            
            # Scale features if needed
            if scaler is not None and len(scaled_columns) > 0:
                scale_cols = [col for col in scaled_columns if col in X.columns]
                if scale_cols:
                    X[scale_cols] = scaler.transform(X[scale_cols])
            
            # Get predictions
            approval_prob = approval_model.predict_proba(X)[0, 1]
            approval = approval_prob >= 0.5
            
            # For approved loans, predict amount and rate
            if approval:
                amount = loan_model.predict(X)[0]
                rate = interest_model.predict(X)[0]
            else:
                amount = 0.0
                rate = 0.0
            
            logger.info(f"  Approval: {approval} (prob: {approval_prob:.4f}, expected: {profile['expected_approval']})")
            if approval:
                logger.info(f"  Amount: ${amount:.2f} (expected: ${profile['min_amount']}-${profile['max_amount']})")
                logger.info(f"  Rate: {rate:.2f}% (expected: {profile['min_rate']}%-{profile['max_rate']}%)")
            
            # Verify predictions are reasonable
            assert approval == profile["expected_approval"], \
                f"Expected approval {profile['expected_approval']}, got {approval}"
            
            if approval:
                assert profile["min_amount"] <= amount <= profile["max_amount"], \
                    f"Amount ${amount:.2f} outside expected range ${profile['min_amount']}-${profile['max_amount']}"
                assert profile["min_rate"] <= rate <= profile["max_rate"], \
                    f"Rate {rate:.2f}% outside expected range {profile['min_rate']}%-{profile['max_rate']}%"
        
    except FileNotFoundError:
        pytest.skip("Model files not found, skipping test")
    except Exception as e:
        logger.error(f"Error testing specific applicant predictions: {str(e)}")
        raise

if __name__ == "__main__":
    # Run tests manually
    test_approval_model_performance()
    test_loan_amount_model_performance()
    test_interest_rate_model_performance()
    test_specific_applicant_predictions() 