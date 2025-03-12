from pydantic import BaseModel, Field
from typing import Optional

class LoanApplication(BaseModel):
    full_name: str
    age: int = Field(..., ge=18, le=100)  # Age should be at least 18
    province: str
    employment_status: str  # Changed from int to str to match frontend
    months_employed: int = Field(..., ge=0)  # Previously missing
    annual_income: float
    self_reported_debt: float  # Replaces `existing_debt`
    debt_to_income_ratio: float  # Missing in frontend, could be computed
    credit_score: int
    credit_history_length: int  # Previously missing
    missed_payments: int  # Previously missing
    credit_utilization: float  # Previously missing
    num_open_accounts: int  # Previously missing
    num_credit_inquiries: int  # Previously missing
    payment_history: str  # Previously missing
    current_credit_limit: float  # Previously missing
    monthly_expenses: float  # Previously missing
    self_reported_expenses: float  # Previously missing
    estimated_debt: float  # Previously missing
    requested_amount: float  # Renamed from `loan_amount_requested`
    preferred_term_months: int  # Previously missing
    collateral_available: int  # Previously missing
    
    # New fields for Equifax integration
    equifax_consent: bool = False  # Whether the user consents to Equifax credit check
    sin: Optional[str] = None  # Social Insurance Number (required for Equifax API)
    date_of_birth: Optional[str] = None  # Date of birth in YYYY-MM-DD format
    street_address: Optional[str] = None  # Street address
    city: Optional[str] = None  # City
    postal_code: Optional[str] = None  # Postal code
    
class CreditReport(BaseModel):
    credit_score: int
    credit_history_length: int
    missed_payments: int
    credit_utilization: float
    open_accounts: int
    credit_inquiries: int
    payment_history: str
    total_credit_limit: float
    total_debt: float
    
class LoanPredictionResponse(BaseModel):
    loan_approved: bool
    approved_amount: float
    interest_rate: float
    credit_report: Optional[CreditReport] = None
