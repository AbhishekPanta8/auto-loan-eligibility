from fastapi import APIRouter, HTTPException
import os
import joblib
import numpy as np
import pandas as pd
import logging

from app.schemas.loan import LoanApplication, CreditReport, LoanPredictionResponse
from app.services.equifax_api import EquifaxAPI
from app.services.loan import transform_loan_application  # if additional transformation is needed
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

# Set the base directory and model directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "ml","models")
print("MODEL_DIR", MODEL_DIR)
# Load models
clf_model = joblib.load(os.path.join(MODEL_DIR, "approval_model.pkl"))
reg_loan_model = joblib.load(os.path.join(MODEL_DIR, "credit_limit_model.pkl"))
reg_interest_model = joblib.load(os.path.join(MODEL_DIR, "interest_rate_model.pkl"))

# Load preprocessing artifacts saved during training:
# 1. The list of encoded feature columns
encoded_feature_columns = joblib.load(os.path.join(MODEL_DIR, "encoded_feature_columns.pkl"))
# 2. The fitted scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Load approval threshold from .env file with fallback to default value
APPROVAL_THRESHOLD = float(os.environ.get("APPROVAL_THRESHOLD", 0.5))
print(f"Using approval threshold: {APPROVAL_THRESHOLD}")

# Employment status mapping (if the model expects integers)
EMPLOYMENT_MAPPING = {
    "Full-time": 1,
    "Part-time": 2,
    "Unemployed": 3,
    "Self-employed": 4,
    "Retired": 5,
    "Student": 6,
}

# Initialize Equifax API client
equifax_api = EquifaxAPI()

@router.post("/predict/", response_model=LoanPredictionResponse)
def predict_loan_eligibility(application: LoanApplication):
    try:
        # ----- Equifax API Logic -----
        credit_report = None
        credit_score = application.credit_score  # Default to self-reported score

        if application.equifax_consent:
            name_parts = application.full_name.split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            address_parts = application.street_address.split(" ") if application.street_address else []
            house_number = address_parts[0] if address_parts else ""
            street_type = address_parts[-1] if len(address_parts) > 1 else ""
            street_name = " ".join(address_parts[1:-1]) if len(address_parts) > 2 else ""
            
            user_data = {
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": application.date_of_birth,
                "house_number": house_number,
                "street_name": street_name,
                "street_type": street_type,
                "city": application.city,
                "province": application.province,
                "postal_code": application.postal_code,
                "sin": application.sin,
                "credit_score": application.credit_score,
                "credit_history_length": application.credit_history_length,
                "missed_payments": application.missed_payments,
                "credit_utilization": application.credit_utilization,
                "num_open_accounts": application.num_open_accounts,
                "num_credit_inquiries": application.num_credit_inquiries,
                "payment_history": application.payment_history,
                "current_credit_limit": application.current_credit_limit,
                "self_reported_debt": application.self_reported_debt,
            }
            
            equifax_response = equifax_api.get_credit_report(user_data)
            
            if equifax_response.get("error"):
                logger.error(f"Error from Equifax API: {equifax_response.get('message')}")
            else:
                eq_report = equifax_response.get("creditReport", {})
                credit_score = eq_report.get("creditScore", application.credit_score)
                credit_report = CreditReport(
                    credit_score=credit_score,
                    credit_history_length=eq_report.get("creditHistoryLength", application.credit_history_length),
                    missed_payments=eq_report.get("missedPayments", application.missed_payments),
                    credit_utilization=eq_report.get("creditUtilization", application.credit_utilization),
                    open_accounts=eq_report.get("openAccounts", application.num_open_accounts),
                    credit_inquiries=eq_report.get("creditInquiries", application.num_credit_inquiries),
                    payment_history=eq_report.get("paymentHistory", application.payment_history),
                    total_credit_limit=eq_report.get("totalCreditLimit", application.current_credit_limit),
                    total_debt=eq_report.get("totalDebt", application.self_reported_debt)
                )
                application.credit_history_length = credit_report.credit_history_length
                application.missed_payments = credit_report.missed_payments
                application.credit_utilization = credit_report.credit_utilization
                application.num_open_accounts = credit_report.open_accounts
                application.num_credit_inquiries = credit_report.credit_inquiries
                application.self_reported_debt = credit_report.total_debt
        
        # ----- Compute Derived Features -----
        if application.annual_income > 0:
            monthly_income = application.annual_income / 12
            debt_to_income_ratio = (application.self_reported_debt + application.estimated_debt + 
                                    (application.requested_amount * 0.03)) / monthly_income
        else:
            debt_to_income_ratio = 0.0

        # Map employment status (if not already preprocessed)
        mapped_employment_status = application.employment_status  # retain string if training used strings
        # Alternatively, if training mapped to integers:
        # mapped_employment_status = EMPLOYMENT_MAPPING.get(application.employment_status, 0)

        # ----- Build Raw Input Dictionary -----
        # Note: Your training features were derived from the raw DataFrame. Here we build a dictionary
        # with the same columns (except target columns). If your training data included 'applicant_id', you might omit it.
        raw_features = {
            "annual_income": application.annual_income,
            "self_reported_debt": application.self_reported_debt,
            "self_reported_expenses": application.self_reported_expenses,
            "requested_amount": application.requested_amount,
            "age": application.age,
            "province": application.province,
            "employment_status": mapped_employment_status,
            "months_employed": application.months_employed,
            "credit_score": credit_score,
            "credit_utilization": application.credit_utilization,
            "num_open_accounts": application.num_open_accounts,
            "num_credit_inquiries": application.num_credit_inquiries,
            "payment_history": application.payment_history,
            "total_credit_limit": application.current_credit_limit,
            "monthly_expenses": application.monthly_expenses,
            "estimated_debt": application.estimated_debt,
            "DTI": debt_to_income_ratio
        }
        
        # Create a DataFrame from raw features
        input_df = pd.DataFrame([raw_features])

        # One-hot encode categorical variables (using drop_first=True as in training)
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Reindex to ensure the same order and columns as used during training.
        input_encoded = input_encoded.reindex(columns=encoded_feature_columns, fill_value=0)

        # (Optional: Debug check)
        assert list(input_encoded.columns) == encoded_feature_columns, "Column mismatch!"

        # Load the list of numeric columns that were scaled during training
        scaled_columns = joblib.load(os.path.join(MODEL_DIR, "scaled_columns.pkl"))

        # Create a copy to process the data
        input_processed = input_encoded.copy()

        # Apply the scaler only to the numeric (scaled) columns
        input_processed[scaled_columns] = scaler.transform(input_encoded[scaled_columns])

        # Optional: Log the processed input for debugging.
        logger.debug(f"Processed input for prediction: {input_processed.to_dict(orient='records')[0]}")

        # # Make predictions using the processed input
        # loan_approval = clf_model.predict(input_processed)[0]
        # Get approval probability instead of direct prediction
        approval_prob = clf_model.predict_proba(input_processed)[0][1]
        
        # Apply configurable threshold from .env
        loan_approval = 1 if approval_prob >= APPROVAL_THRESHOLD else 0

        estimated_loan_amount = None
        estimated_interest_rate = None

        if loan_approval == 1:
            estimated_loan_amount = reg_loan_model.predict(input_processed)[0]
            estimated_interest_rate = reg_interest_model.predict(input_processed)[0]
        else:
            estimated_loan_amount = 0.0
            estimated_interest_rate = 0.0

        return LoanPredictionResponse(
            loan_approved=bool(loan_approval),
            approved_amount=estimated_loan_amount,
            interest_rate=estimated_interest_rate,
            credit_report=credit_report
        )
    except Exception as e:
        logger.exception("Error predicting loan eligibility")
        raise HTTPException(status_code=500, detail=f"Error predicting loan eligibility: {str(e)}")
