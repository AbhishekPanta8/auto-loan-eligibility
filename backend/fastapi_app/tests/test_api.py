# import pytest
# from fastapi.testclient import TestClient
# from app.main import app  # Import your FastAPI "app" instance

# client = TestClient(app)

# @pytest.mark.parametrize(
#     "payload, expected_status",
#     [
#         # Valid example
#         (
#             {
#                 "age": 30,
#                 "employment_status": 2,  # Salaried
#                 "annual_income": 75000,
#                 "existing_debt": 10000,
#                 "debt_to_income_ratio": 0.2,
#                 "credit_score": 720,
#                 "credit_history_length": 10,
#                 "missed_payments": 0,
#                 "loan_amount_requested": 20000,
#                 "preferred_term_months": 36,
#                 "collateral_available": 1
#             },
#             200,
#         ),
#         # Edge case: borderline credit score
#         (
#             {
#                 "age": 18,
#                 "employment_status": 1,  # Self-employed
#                 "annual_income": 20000,
#                 "existing_debt": 0,
#                 "debt_to_income_ratio": 0.9,
#                 "credit_score": 600,
#                 "credit_history_length": 0,
#                 "missed_payments": 5,
#                 "loan_amount_requested": 50000,
#                 "preferred_term_months": 60,
#                 "collateral_available": 0
#             },
#             200,
#         ),
#         # Invalid example: missing required field or negative age
#         (
#             {
#                 "employment_status": 2,
#                 "annual_income": 75000,
#                 "existing_debt": 10000,
#                 # "age" is missing or negative
#                 "age": -5,
#                 "debt_to_income_ratio": 0.2,
#                 "credit_score": 720,
#                 "credit_history_length": 10,
#                 "missed_payments": 0,
#                 "loan_amount_requested": 20000,
#                 "preferred_term_months": 36,
#                 "collateral_available": 1
#             },
#             422,  # Expect Unprocessable Entity for invalid inputs
#         ),
#     ],
# )
# def test_predict(payload, expected_status):
#     response = client.post("/predict/", json=payload)
#     assert response.status_code == expected_status

#     if response.status_code == 200:
#         data = response.json()
#         # Basic checks
#         assert "loan_approved" in data
#         assert "estimated_loan_amount" in data
#         assert "estimated_interest_rate" in data
#         assert isinstance(data["loan_approved"], bool)
#         assert isinstance(data["estimated_loan_amount"], float)
#         assert isinstance(data["estimated_interest_rate"], float)
import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI "app" instance

client = TestClient(app)

@pytest.mark.parametrize(
    "payload, expected_status",
    [
        # Valid example
        (
            {
                "age": 30,
                "employment_status": 2,  # Salaried
                "annual_income": 75000,
                "existing_debt": 10000,
                "debt_to_income_ratio": 0.2,
                "credit_score": 720,
                "credit_history_length": 10,
                "missed_payments": 0,
                "loan_amount_requested": 20000,
                "preferred_term_months": 36,
                "collateral_available": 1
            },
            200,
        ),
        # Edge case: borderline credit score
        (
            {
                "age": 18,
                "employment_status": 1,  # Self-employed
                "annual_income": 20000,
                "existing_debt": 0,
                "debt_to_income_ratio": 0.9,
                "credit_score": 600,
                "credit_history_length": 0,
                "missed_payments": 5,
                "loan_amount_requested": 50000,
                "preferred_term_months": 60,
                "collateral_available": 0
            },
            200,
        ),
        # Invalid example: missing required field or negative age
        (
            {
                "employment_status": 2,
                "annual_income": 75000,
                "existing_debt": 10000,
                # "age" is missing or negative
                "age": -5,
                "debt_to_income_ratio": 0.2,
                "credit_score": 720,
                "credit_history_length": 10,
                "missed_payments": 0,
                "loan_amount_requested": 20000,
                "preferred_term_months": 36,
                "collateral_available": 1
            },
            422,  # Expect Unprocessable Entity for invalid inputs
        ),
    ],
)
def test_predict(payload, expected_status):
    # Print the payload being sent to give context
    print(f"\n[TEST] Sending payload to /predict/: {payload}")
    
    response = client.post("/predict/", json=payload)
    
    # Print the status code and response for debugging
    print(f"[TEST] Expected status: {expected_status}, Actual status: {response.status_code}")
    print(f"[TEST] Response body: {response.text}")

    # Assert the status code as before (no logic change)
    assert response.status_code == expected_status

    # If 200 (OK), print more info about the JSON response
    if response.status_code == 200:
        data = response.json()
        
        print("[TEST] Response JSON parsed successfully.")
        print(f"[TEST] loan_approved: {data.get('loan_approved')}, "
              f"estimated_loan_amount: {data.get('estimated_loan_amount')}, "
              f"estimated_interest_rate: {data.get('estimated_interest_rate')}")

        # The original checks remain the same
        assert "loan_approved" in data
        assert "estimated_loan_amount" in data
        assert "estimated_interest_rate" in data
        assert isinstance(data["loan_approved"], bool)
        assert isinstance(data["estimated_loan_amount"], float)
        assert isinstance(data["estimated_interest_rate"], float)
