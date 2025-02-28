from fastapi import APIRouter
from app.schemas.loan import LoanApplication
import numpy as np
import joblib
import os

router = APIRouter()

# Correctly set BASE_DIR to the `fastapi-app/` directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Set the models directory inside `fastapi-app/`
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Debugging: Print paths to verify correctness
print(f"\n--- Debugging Paths ---")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")

# Load models with the correct path
clf_model = joblib.load(os.path.join(MODEL_DIR, "loan_approval_model.pkl"))
reg_loan_model = joblib.load(os.path.join(MODEL_DIR, "loan_amount_model.pkl"))
reg_interest_model = joblib.load(os.path.join(MODEL_DIR, "interest_rate_model.pkl"))

@router.post("/predict/")
def predict_loan_eligibility(application: LoanApplication):
    input_data = np.array([[
        application.age, application.employment_status, application.annual_income,
        application.existing_debt, application.debt_to_income_ratio, application.credit_score,
        application.credit_history_length, application.missed_payments,
        application.loan_amount_requested, application.preferred_term_months,
        application.collateral_available
    ]])

    loan_approval = clf_model.predict(input_data)[0]
    estimated_loan_amount = None
    estimated_interest_rate = None

    if loan_approval == 1:
        estimated_loan_amount = reg_loan_model.predict(input_data)[0]
        estimated_interest_rate = reg_interest_model.predict(input_data)[0]
    else:
        # 0 for rejected loans, can be made to None if desired
        estimated_loan_amount = 0.0
        estimated_interest_rate = 0.0

    return {
        "loan_approved": bool(loan_approval),
        "estimated_loan_amount": estimated_loan_amount,
        "estimated_interest_rate": estimated_interest_rate
    }
