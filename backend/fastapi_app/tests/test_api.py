import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI "app" instance

client = TestClient(app)

@pytest.mark.parametrize(
    "description, payload, expected_status",
    [
        (
            "Valid example: typical applicant (should pass)",
            {
                "full_name": "John Doe",
                "age": 30,
                "province": "ON",
                "employment_status": "Full-time",
                "months_employed": 12,
                "annual_income": 75000,
                "self_reported_debt": 10000,
                "debt_to_income_ratio": 0.2,
                "credit_score": 720,
                "credit_history_length": 10,
                "missed_payments": 0,
                "credit_utilization": 30,
                "num_open_accounts": 3,
                "num_credit_inquiries": 1,
                "payment_history": "On Time",
                "current_credit_limit": 5000,
                "monthly_expenses": 2000,
                "self_reported_expenses": 2000,
                "estimated_debt": 0,
                "requested_amount": 20000,
                "preferred_term_months": 36,
                "collateral_available": 1,
                "equifax_consent": False,
                "sin": "",
                "date_of_birth": "1990-01-01",
                "street_address": "123 Main St",
                "city": "Toronto",
                "postal_code": "M5H2N2",
                "house_number": "123",
                "street_name": "Main",
                "street_type": "St"
            },
            200,
        ),
        (
            "Invalid example: negative age (should fail)",
            {
                "full_name": "John Doe",
                "age": -5,  # Invalid: negative age
                "province": "ON",
                "employment_status": "Full-time",
                "months_employed": 12,
                "annual_income": 75000,
                "self_reported_debt": 10000,
                "debt_to_income_ratio": 0.2,
                "credit_score": 720,
                "credit_history_length": 10,
                "missed_payments": 0,
                "credit_utilization": 30,
                "num_open_accounts": 3,
                "num_credit_inquiries": 1,
                "payment_history": "On Time",
                "current_credit_limit": 5000,
                "monthly_expenses": 2000,
                "self_reported_expenses": 2000,
                "estimated_debt": 0,
                "requested_amount": 20000,
                "preferred_term_months": 36,
                "collateral_available": 1,
                "equifax_consent": False,
                "sin": "",
                "date_of_birth": "1990-01-01",
                "street_address": "123 Main St",
                "city": "Toronto",
                "postal_code": "M5H2N2",
                "house_number": "123",
                "street_name": "Main",
                "street_type": "St"
            },
            422,
        ),
    ],
)
def test_predict(description, payload, expected_status):
    response = client.post("/predict/", json=payload)
    outcome = "PASSED" if response.status_code == expected_status else "FAILED"
    print(f"Test: {description} | Expected: {expected_status}, Actual: {response.status_code} | {outcome}")
    assert response.status_code == expected_status
