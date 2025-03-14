# services/loan_service.py

def transform_loan_application(application):
    """
    Transform the LoanApplication Pydantic model instance into a dictionary
    of features expected by the ML model.
    """
    ml_features = {
        "annual_income": application.annual_income,
        "self_reported_debt": application.self_reported_debt,
        "self_reported_expenses": application.self_reported_expenses,
        "requested_amount": application.requested_amount,
        "age": application.age,
        "province": application.province,
        "employment_status": application.employment_status,
        "months_employed": application.months_employed,
        "credit_score": application.credit_score,
        "credit_utilization": application.credit_utilization,
        "num_open_accounts": application.num_open_accounts,
        "num_credit_inquiries": application.num_credit_inquiries,
        "payment_history": application.payment_history,
        "total_credit_limit": application.current_credit_limit,  # Mapping field name
        "monthly_expenses": application.monthly_expenses,
        "estimated_debt": application.estimated_debt,
        "DTI": application.debt_to_income_ratio  # Assuming DTI is provided
    }
    return ml_features
