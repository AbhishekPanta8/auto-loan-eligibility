from pydantic import BaseModel

class LoanApplication(BaseModel):
    age: int
    employment_status: int
    annual_income: float
    existing_debt: float
    debt_to_income_ratio: float
    credit_score: int
    credit_history_length: int
    missed_payments: int
    loan_amount_requested: float
    preferred_term_months: int
    collateral_available: int
