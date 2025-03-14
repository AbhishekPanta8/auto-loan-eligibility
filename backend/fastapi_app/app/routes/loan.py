from fastapi import APIRouter, HTTPException
import numpy as np
import joblib
import os
from app.schemas.loan import LoanApplication, CreditReport, LoanPredictionResponse
from app.services.equifax_api import EquifaxAPI
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter()

# Correctly set the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "ml","models")
print("MODEL_DIR", MODEL_DIR)
# Load models
clf_model = joblib.load(os.path.join(MODEL_DIR, "approval_model.pkl"))
reg_loan_model = joblib.load(os.path.join(MODEL_DIR, "credit_limit_model.pkl"))
reg_interest_model = joblib.load(os.path.join(MODEL_DIR, "interest_rate_model.pkl"))

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
        # Check if user has consented to Equifax credit check
        credit_report = None
        credit_score = application.credit_score  # Default to self-reported score
        
        if application.equifax_consent:
            # Extract first and last name from full name
            name_parts = application.full_name.split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            # Extract house number, street name, and type from street_address if provided
            address_parts = application.street_address.split(" ") if application.street_address else []
            house_number = address_parts[0] if address_parts else ""
            street_type = address_parts[-1] if len(address_parts) > 1 else ""
            street_name = " ".join(address_parts[1:-1]) if len(address_parts) > 2 else ""
            
            # Prepare user data for Equifax API
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
                # Include self-reported data for fallback
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
            
            # Call Equifax API to get credit report
            equifax_response = equifax_api.get_credit_report(user_data)
            
            # Check if there was an error
            if equifax_response.get("error"):
                logger.error(f"Error from Equifax API: {equifax_response.get('message')}")
                # Fall back to self-reported credit score
            else:
                # Extract credit report data
                equifax_credit_report = equifax_response.get("creditReport", {})
                
                # Use Equifax credit score instead of self-reported
                credit_score = equifax_credit_report.get("creditScore", application.credit_score)
                
                # Create credit report response
                credit_report = CreditReport(
                    credit_score=credit_score,
                    credit_history_length=equifax_credit_report.get("creditHistoryLength", application.credit_history_length),
                    missed_payments=equifax_credit_report.get("missedPayments", application.missed_payments),
                    credit_utilization=equifax_credit_report.get("creditUtilization", application.credit_utilization),
                    open_accounts=equifax_credit_report.get("openAccounts", application.num_open_accounts),
                    credit_inquiries=equifax_credit_report.get("creditInquiries", application.num_credit_inquiries),
                    payment_history=equifax_credit_report.get("paymentHistory", application.payment_history),
                    total_credit_limit=equifax_credit_report.get("totalCreditLimit", application.current_credit_limit),
                    total_debt=equifax_credit_report.get("totalDebt", application.self_reported_debt)
                )
                
                # Update application data with Equifax data for model input
                application.credit_history_length = credit_report.credit_history_length
                application.missed_payments = credit_report.missed_payments
                application.credit_utilization = credit_report.credit_utilization
                application.num_open_accounts = credit_report.open_accounts
                application.num_credit_inquiries = credit_report.credit_inquiries
                application.self_reported_debt = credit_report.total_debt
        
        # Compute missing `debt_to_income_ratio`
        if application.annual_income > 0:
            # Calculate DTI according to the documentation:
            # DTI = (self_reported_debt + estimated_debt + (requested_amount * 0.03)) / (annual_income / 12)
            monthly_income = application.annual_income / 12
            debt_to_income_ratio = (application.self_reported_debt + application.estimated_debt + 
                                   (application.requested_amount * 0.03)) / monthly_income
        else:
            debt_to_income_ratio = 0.0

        # Ensure `employment_status` is converted to an integer if needed
        employment_status = EMPLOYMENT_MAPPING.get(application.employment_status, 0)

        # Prepare input data in the order expected by the model
        input_data = np.array([[
            application.age,
            employment_status,
            # application.months_employed,
            application.annual_income,
            application.self_reported_debt,
            # application.estimated_debt,
            debt_to_income_ratio,
            credit_score,
            application.credit_history_length,
            application.missed_payments,
            # application.credit_utilization,
            # application.num_open_accounts,
            # application.num_credit_inquiries,
            # 1 if application.payment_history == "On Time" else 0,  # Convert payment history to binary
            # application.current_credit_limit,
            # application.monthly_expenses,
            # application.self_reported_expenses,
            application.requested_amount,
            application.preferred_term_months,
            application.collateral_available
        ]])

        # Get approval probability instead of direct prediction
        approval_prob = clf_model.predict_proba(input_data)[0][1]
        
        # Apply configurable threshold from .env
        loan_approval = 1 if approval_prob >= APPROVAL_THRESHOLD else 0

        estimated_loan_amount = None
        estimated_interest_rate = None

        if loan_approval == 1:
            estimated_loan_amount = reg_loan_model.predict(input_data)[0]
            estimated_interest_rate = reg_interest_model.predict(input_data)[0]
        else:
            # 0 for rejected loans, can be made to None if desired
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
