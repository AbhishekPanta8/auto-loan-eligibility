from pydantic import BaseModel, Field

class LoanApplication(BaseModel):
    age: int = Field(..., ge=0)
    employment_status: int  # <-- Add this line
    annual_income: float
    existing_debt: float
    debt_to_income_ratio: float
    credit_score: int
    credit_history_length: int
    missed_payments: int
    loan_amount_requested: float
    preferred_term_months: int
    collateral_available: int
