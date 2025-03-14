import os
import sys
import pytest
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TestClient without any mocks
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)

# Mark all tests in this file as end-to-end
pytestmark = pytest.mark.end_to_end

# Check if we can run end-to-end tests with real models
def has_real_models():
    """Check if real models are available to run end-to-end tests"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model_files = [
        "approval_model.pkl",
        "credit_limit_model.pkl",
        "interest_rate_model.pkl"
    ]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in model_files)

# Skip all tests if models are not available
if not has_real_models():
    pytest.skip("Real models not found, skipping end-to-end tests", allow_module_level=True)

# Helper function to generate test application data
def generate_test_application(profile_type="good"):
    """Generate test application data based on profile type"""
    
    # Base application template
    application = {
        "full_name": "E2E Test User",
        "age": 35,
        "province": "ON",
        "employment_status": "Full-time",
        "months_employed": 60,
        "annual_income": 80000.0,
        "self_reported_debt": 1000.0,
        "debt_to_income_ratio": 0.2,
        "credit_score": 700,
        "credit_history_length": 60,
        "missed_payments": 0,
        "credit_utilization": 30.0,
        "num_open_accounts": 3,
        "num_credit_inquiries": 1,
        "payment_history": "On Time",
        "current_credit_limit": 25000.0,
        "monthly_expenses": 2500.0,
        "self_reported_expenses": 2000.0,
        "estimated_debt": 500.0,
        "requested_amount": 20000.0,
        "preferred_term_months": 36,
        "collateral_available": 1,
        "equifax_consent": False
    }
    
    # Modify based on profile type
    if profile_type == "excellent":
        application.update({
            "full_name": "E2E Excellent",
            "annual_income": 120000.0,
            "credit_score": 780,
            "credit_utilization": 20.0,
            "debt_to_income_ratio": 0.15,
            "requested_amount": 25000.0
        })
    elif profile_type == "good":
        # Use defaults (already good)
        pass
    elif profile_type == "borderline":
        application.update({
            "full_name": "E2E Borderline",
            "annual_income": 60000.0,
            "credit_score": 670,
            "credit_utilization": 50.0,
            "debt_to_income_ratio": 0.35,
            "months_employed": 24,
            "employment_status": "Part-time",
            "requested_amount": 15000.0
        })
    elif profile_type == "poor":
        application.update({
            "full_name": "E2E Poor",
            "annual_income": 45000.0,
            "credit_score": 500,
            "credit_utilization": 70.0,
            "debt_to_income_ratio": 0.45,
            "missed_payments": 5,
            "payment_history": "Late>60",
            "self_reported_debt": 2000.0,
            "requested_amount": 30000.0
        })
    elif profile_type == "very_poor":
        application.update({
            "full_name": "E2E Very Poor",
            "annual_income": 35000.0,
            "credit_score": 450,
            "credit_utilization": 85.0,
            "debt_to_income_ratio": 0.55,
            "missed_payments": 8,
            "payment_history": "Late>60",
            "self_reported_debt": 3000.0,
            "requested_amount": 25000.0
        })
    
    return application

# Log the results of a test prediction to help with debugging
def log_prediction_result(application, response):
    """Log detailed information about prediction results"""
    logger.info("-" * 50)
    logger.info(f"Prediction for {application['full_name']}")
    logger.info(f"  Credit Score: {application['credit_score']}")
    logger.info(f"  Annual Income: ${application['annual_income']}")
    logger.info(f"  DTI: {application['debt_to_income_ratio']:.2f}")
    logger.info(f"  Requested: ${application['requested_amount']}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"  Decision: {'APPROVED' if data['loan_approved'] else 'REJECTED'}")
        logger.info(f"  Amount: ${data['approved_amount']}")
        logger.info(f"  Interest Rate: {data['interest_rate']}%")
    else:
        logger.warning(f"  Error: Status {response.status_code}")
        logger.warning(f"  Details: {response.text}")
    logger.info("-" * 50)

@pytest.mark.parametrize(
    "profile_type, expected_decision",
    [
        ("excellent", True),    # Should definitely be approved
        ("good", True),         # Should be approved
        ("borderline", None),   # Could go either way (don't assert)
        ("poor", None),         # Could go either way (don't assert)
        ("very_poor", False)    # Should definitely be rejected
    ]
)
def test_real_models_with_different_profiles(profile_type, expected_decision):
    """
    End-to-end test using real models with different applicant profiles.
    This test skips the mocking and uses the actual deployed models.
    """
    # Generate test application data
    application = generate_test_application(profile_type)
    
    # Send request to the endpoint
    response = client.post("/predict/", json=application)
    
    # Log detailed information about prediction results
    log_prediction_result(application, response)
    
    # Verify the API response is successful
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "loan_approved" in data
    assert "approved_amount" in data
    assert "interest_rate" in data
    
    # For definite cases, check expected decision
    if expected_decision is not None:
        assert data["loan_approved"] == expected_decision, \
            f"Expected {expected_decision} but got {data['loan_approved']} for {profile_type} profile"
    
    # For approved loans, check reasonable amounts and rates
    if data["loan_approved"]:
        assert data["approved_amount"] > 0
        assert data["approved_amount"] <= application["requested_amount"]
        assert 3.0 <= data["interest_rate"] <= 15.0
    else:
        assert data["approved_amount"] == 0

def test_real_models_with_real_data():
    """
    Test real models with some actual data from the dataset.
    Uses a small sample of the synthetic dataset to verify end-to-end behavior.
    """
    # Path to synthetic data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "..", "datasets", "data", "synthetic_loan_applications.csv")
    
    # Skip if no data available
    if not os.path.exists(data_path):
        pytest.skip("Synthetic data not found, skipping test")
    
    try:
        # Load a small sample of the data
        df = pd.read_csv(data_path)
        sample_size = min(5, len(df))  # Take at most 5 samples
        samples = df.sample(n=sample_size, random_state=42)
        
        # Map dataset columns to API input format
        results = []
        for _, row in samples.iterrows():
            # Create application data from dataset row
            application = {
                "full_name": f"Sample {row['applicant_id']}" if 'applicant_id' in row else "Sample User",
                "age": row["age"] if "age" in row else 35,
                "province": row["province"] if "province" in row else "ON",
                "employment_status": row["employment_status"] if "employment_status" in row else "Full-time",
                "months_employed": int(row["months_employed"]) if "months_employed" in row else 60,
                "annual_income": float(row["annual_income"]) if "annual_income" in row else 80000.0,
                "self_reported_debt": float(row["self_reported_debt"]) if "self_reported_debt" in row else 1000.0,
                "debt_to_income_ratio": float(row["DTI"]/100) if "DTI" in row else 0.2,
                "credit_score": int(row["credit_score"]) if "credit_score" in row else 700,
                "credit_history_length": 60,  # Default if not in data
                "missed_payments": 0,  # Default if not in data
                "credit_utilization": float(row["credit_utilization"]) if "credit_utilization" in row else 30.0,
                "num_open_accounts": int(row["num_open_accounts"]) if "num_open_accounts" in row else 3,
                "num_credit_inquiries": int(row["num_credit_inquiries"]) if "num_credit_inquiries" in row else 1,
                "payment_history": row["payment_history"] if "payment_history" in row else "On Time",
                "current_credit_limit": float(row["total_credit_limit"]) if "total_credit_limit" in row else 25000.0,
                "monthly_expenses": float(row["monthly_expenses"]) if "monthly_expenses" in row else 2500.0,
                "self_reported_expenses": float(row["self_reported_expenses"]) if "self_reported_expenses" in row else 2000.0,
                "estimated_debt": float(row["estimated_debt"]) if "estimated_debt" in row else 500.0,
                "requested_amount": float(row["requested_amount"]) if "requested_amount" in row else 20000.0,
                "preferred_term_months": 36,
                "collateral_available": 1,
                "equifax_consent": False
            }
            
            # Send request to the endpoint
            response = client.post("/predict/", json=application)
            
            # Log detailed information about prediction results
            log_prediction_result(application, response)
            
            # Verify the API response is successful
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "loan_approved" in data
            assert "approved_amount" in data
            assert "interest_rate" in data
            
            # For approved loans, check reasonable amounts and rates
            if data["loan_approved"]:
                assert data["approved_amount"] > 0
                assert data["approved_amount"] <= application["requested_amount"]
                assert 3.0 <= data["interest_rate"] <= 15.0
            else:
                assert data["approved_amount"] == 0
            
            # Compare decision with synthetic dataset (if available)
            if 'approved' in row:
                expected = bool(row['approved'])
                actual = data["loan_approved"]
                
                # Log comparison but don't fail test - models may have changed
                if expected != actual:
                    logger.warning(f"Model decision differs from dataset: expected {expected}, got {actual}")
                    logger.warning("This is not necessarily an error - models may have been updated")
            
            # Store results for logging
            results.append({
                "applicant_id": row.get('applicant_id', 'unknown'),
                "credit_score": row.get('credit_score', 0),
                "income": row.get('annual_income', 0),
                "dti": row.get('DTI', 0),
                "expected": bool(row['approved']) if 'approved' in row else None,
                "actual": data["loan_approved"],
                "amount": data["approved_amount"],
                "rate": data["interest_rate"]
            })
        
        # Log summary of results
        logger.info("\nE2E Test Results Summary:")
        for r in results:
            logger.info(f"ID: {r['applicant_id']}, Score: {r['credit_score']}, " +
                      f"DTI: {r['dti']}, Decision: {r['actual']}, " +
                      f"Amount: ${r['amount']}, Rate: {r['rate']}%")
        
    except Exception as e:
        logger.error(f"Error in real data test: {str(e)}")
        raise

def test_business_rules_with_real_models():
    """
    Test that business rules are correctly applied with real models.
    This verifies that even when models would approve, business rules can override.
    """
    # Test cases designed to trigger business rules
    test_cases = [
        {
            "name": "Very Low Credit Score",
            "application": generate_test_application("good"),  # Start with good profile
            "override": {"credit_score": 400},  # But override credit score to be very bad
            "expected_approved": False  # Should be rejected by business rules
        },
        {
            "name": "Very High Credit Utilization",
            "application": generate_test_application("good"),
            "override": {"credit_utilization": 95.0},
            "expected_approved": False
        }
    ]
    
    for tc in test_cases:
        # Apply overrides
        application = tc["application"].copy()
        for key, value in tc["override"].items():
            application[key] = value
        
        # Send request to the endpoint
        response = client.post("/predict/", json=application)
        
        # Log detailed information about prediction results
        logger.info(f"Testing business rule: {tc['name']}")
        log_prediction_result(application, response)
        
        # Verify the API response is successful
        assert response.status_code == 200
        data = response.json()
        
        # Verify business rule was applied correctly
        assert data["loan_approved"] == tc["expected_approved"], \
            f"Business rule not applied correctly for {tc['name']}"

if __name__ == "__main__":
    # Run tests manually
    if has_real_models():
        test_real_models_with_different_profiles("excellent", True)
        test_real_models_with_different_profiles("good", True)
        test_real_models_with_different_profiles("very_poor", False)
        test_real_models_with_real_data()
        test_business_rules_with_real_models()
    else:
        print("Real models not found, skipping end-to-end tests") 