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
from app.services.explanation_service import ExplanationService

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

# Initialize explanation service
explanation_service = ExplanationService(
    model_path=os.path.join(MODEL_DIR, "approval_model.pkl"),
    scaler_path=os.path.join(MODEL_DIR, "scaler.pkl"),
    feature_columns_path=os.path.join(MODEL_DIR, "encoded_feature_columns.pkl")
)

def apply_business_rules(
    loan_approval: int, 
    estimated_loan_amount: float, 
    estimated_interest_rate: float,
    credit_score: int,
    dti: float,
    application: LoanApplication
) -> tuple:
    """
    Apply business rules to ensure compliance with approval conditions, credit limits,
    and interest rate constraints.
    
    Parameters:
    -----------
    loan_approval : int
        Initial loan approval status (1 for approved, 0 for denied)
    estimated_loan_amount : float
        ML-predicted loan amount
    estimated_interest_rate : float
        ML-predicted interest rate
    credit_score : int
        Applicant's credit score
    dti : float
        Debt-to-income ratio as a percentage
    application : LoanApplication
        The loan application object containing all applicant information
    debt_to_income_ratio : float
        Raw DTI ratio (before percentage conversion)
        
    Returns:
    --------
    tuple
        (final_approval, final_amount, final_interest_rate)
    """
    # Enforce hard approval/denial rules
    if loan_approval == 0:
        return loan_approval, estimated_loan_amount, estimated_interest_rate
    
    # Deny if any of these conditions are met (regardless of model prediction)
    if (credit_score < 500 or 
        dti > 50 or 
        application.credit_utilization > 80):
        loan_approval = 0
        estimated_loan_amount = 0.0
        estimated_interest_rate = 0.0
        
    # Enforce credit limit constraints

    # Base limit based on credit score
    if credit_score >= 660:
        base_limit = application.annual_income * 0.5
    elif credit_score >= 500:
        base_limit = application.annual_income * 0.25
    else:
        base_limit = application.annual_income * 0.1
    
    # DTI adjustment
    if dti <= 30:
        dti_factor = 1.0
    elif dti <= 40:
        dti_factor = 0.75
    else:
        dti_factor = 0.5
    
    # Credit score cap
    if credit_score >= 750:
        credit_cap = 25000
    elif credit_score >= 660:
        credit_cap = 15000
    elif credit_score >= 500:
        credit_cap = 10000
    else:
        credit_cap = 5000
    
    # Employment bonus
    employment_bonus = 1.1 if (application.employment_status == "Full-time" and application.months_employed >= 12) else 1.0
    
    # Payment history penalty
    payment_penalty = 0.5 if application.payment_history == "Late >60" else 1.0
    
    # Credit utilization penalty
    utilization_penalty = 0.8 if application.credit_utilization > 50 else 1.0
    
    # Calculate adjusted credit limit
    adjusted_limit = min(
        base_limit * dti_factor * employment_bonus * payment_penalty * utilization_penalty,
        credit_cap
    )
    
    # Use the lower of ML prediction or rule-based limit
    estimated_loan_amount = min(estimated_loan_amount, adjusted_limit)
    estimated_loan_amount = min(estimated_loan_amount, application.requested_amount)
    
    # Interest rate adjustments - Option 1: Remove ALL redundant adjustments
    # The ML model has already learned these patterns, so we only enforce min/max bounds
    
    # Ensure interest rate is within specified range (3-15%)
    estimated_interest_rate = max(3.0, min(15.0, estimated_interest_rate))
    
    # Removed all adjustment rules including num_open_accounts > 5
    # to avoid redundancy with what the model has already learned

    return loan_approval, estimated_loan_amount, estimated_interest_rate

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
        mapped_employment_status = application.employment_status.value  # Use .value to get the string from enum
        # Alternatively, if training mapped to integers:
        # mapped_employment_status = EMPLOYMENT_MAPPING.get(application.employment_status.value, 0)

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
        print("raw_features", raw_features)
        # Create a DataFrame from raw features
        X = pd.DataFrame([raw_features])

        # One-hot encode categorical variables (using drop_first=True as in training)
        input_encoded = pd.get_dummies(X, drop_first=True)

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

        # Get model prediction
        approval_probability = clf_model.predict_proba(input_processed)[0][1]
        loan_approval = 1 if approval_probability >= APPROVAL_THRESHOLD else 0
        rejection_probability = 1 - approval_probability
        
        # Get explanation if loan is rejected
        explanation = None
        if loan_approval == 0:
            explanation = explanation_service.get_rejection_explanation(
                raw_features,  # Pass raw features instead of scaled
                rejection_probability
            )
        
        # Get loan amount and interest rate predictions only if approved
        if loan_approval == 1:
            estimated_loan_amount = float(reg_loan_model.predict(input_processed)[0])
            estimated_interest_rate = float(reg_interest_model.predict(input_processed)[0])
        else:
            estimated_loan_amount = 0.0
            estimated_interest_rate = 0.0
        
        # Calculate DTI percentage for business rules
        dti = debt_to_income_ratio * 100  # Convert to percentage
        
        # Apply business rules
        final_approval, final_amount, final_rate = apply_business_rules(
            loan_approval,
            estimated_loan_amount,
            estimated_interest_rate,
            credit_score,
            dti,
            application
        )
        
        # Ensure zero amounts for rejected loans
        if not final_approval:
            final_amount = 0.0
            final_rate = 0.0
        
        return LoanPredictionResponse(
            loan_approved=bool(final_approval),
            approved_amount=final_amount,
            interest_rate=final_rate,
            credit_report=credit_report,
            explanation=explanation,
            approval_probability=approval_probability,
            rejection_probability=rejection_probability,
            approval_threshold=APPROVAL_THRESHOLD
        )
    except Exception as e:
        logger.exception("Error predicting loan eligibility")
        raise HTTPException(status_code=500, detail=f"Error predicting loan eligibility: {str(e)}")
