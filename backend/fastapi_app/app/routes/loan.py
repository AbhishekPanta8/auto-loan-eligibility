from fastapi import APIRouter
import numpy as np
import joblib
import os
from app.schemas.loan import LoanApplication

router = APIRouter()

# Correctly set the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models
clf_model = joblib.load(os.path.join(MODEL_DIR, "loan_approval_model.pkl"))
reg_loan_model = joblib.load(os.path.join(MODEL_DIR, "loan_amount_model.pkl"))
reg_interest_model = joblib.load(os.path.join(MODEL_DIR, "interest_rate_model.pkl"))

# Employment status mapping (if the model expects integers)
EMPLOYMENT_MAPPING = {
    "Full-time": 1,
    "Part-time": 2,
    "Unemployed": 3,
    "Self-employed": 4,
    "Retired": 5,
    "Student": 6,
}

@router.post("/predict/")
def predict_loan_eligibility(application: LoanApplication):
    # Compute missing `debt_to_income_ratio`
    if application.annual_income > 0:
        debt_to_income_ratio = application.self_reported_debt / application.annual_income
    else:
        debt_to_income_ratio = 0.0

    # Ensure `employment_status` is converted to an integer if needed
    employment_status = EMPLOYMENT_MAPPING.get(application.employment_status, 0)

    input_data = np.array([[
        application.age,
        employment_status,
        application.annual_income,
        application.self_reported_debt,
        debt_to_income_ratio,  # Computed dynamically
        application.credit_score,
        application.credit_history_length,
        application.missed_payments,
        application.requested_amount,
        application.preferred_term_months,
        application.collateral_available
    ]])

    # Predict loan approval
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
        "approved_amount": estimated_loan_amount,
        "interest_rate": estimated_interest_rate
    }
