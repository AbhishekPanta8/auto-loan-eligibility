from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List, Dict, Any
from enum import Enum

class EmploymentStatus(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    UNEMPLOYED = "Unemployed"
    SELF_EMPLOYED = "Self-employed"
    RETIRED = "Retired"
    STUDENT = "Student"

class Province(str, Enum):
    AB = "AB"
    BC = "BC"
    MB = "MB"
    NB = "NB"
    NL = "NL"
    NS = "NS"
    NT = "NT"
    NU = "NU"
    ON = "ON"
    PE = "PE"
    QC = "QC"
    SK = "SK"
    YT = "YT"

class LoanApplication(BaseModel):
    full_name: str
    age: int = Field(..., ge=19, le=100, description="Applicant age between 19 and 100")
    province: Province = Field(..., description="Canadian province code")
    employment_status: EmploymentStatus = Field(..., description="Current employment status")
    months_employed: int = Field(..., ge=0, le=600, description="Months at current job (0-600)")
    annual_income: float = Field(..., ge=20000, le=200000, description="Total yearly income (20,000-200,000 CAD)")
    self_reported_debt: float = Field(..., ge=0, le=10000, description="Existing monthly debt (0-10,000 CAD)")
    debt_to_income_ratio: float = Field(..., ge=0, description="Debt to income ratio")
    credit_score: int = Field(..., ge=300, le=900, description="Credit score (300-900)")
    credit_history_length: int = Field(..., ge=0, description="Length of credit history in months")
    missed_payments: int = Field(..., ge=0, description="Number of missed payments")
    credit_utilization: float = Field(..., ge=0, le=100, description="Percentage of credit used (0-100%)")
    num_open_accounts: int = Field(..., ge=0, le=20, description="Number of active credit accounts (0-20)")
    num_credit_inquiries: int = Field(..., ge=0, description="Number of credit inquiries")
    payment_history: str
    current_credit_limit: float = Field(..., ge=0, description="Current credit limit")
    monthly_expenses: float = Field(..., ge=0, description="Monthly expenses")
    self_reported_expenses: float = Field(..., ge=0, le=10000, description="Self-reported expenses (0-10,000 CAD)")
    estimated_debt: float = Field(..., ge=0, description="Estimated debt")
    requested_amount: float = Field(..., ge=1000, le=50000, description="Requested loan amount (1,000-50,000 CAD)")
    preferred_term_months: int = Field(..., ge=0, description="Preferred loan term in months")
    collateral_available: int = Field(..., ge=0, description="Collateral available")
    
    # New fields for Equifax integration
    equifax_consent: bool = False  # Whether the user consents to Equifax credit check
    sin: Optional[str] = None  # Social Insurance Number (required for Equifax API)
    date_of_birth: Optional[str] = None  # Date of birth in YYYY-MM-DD format
    street_address: Optional[str] = None  # Street address
    city: Optional[str] = None  # City
    postal_code: Optional[str] = None  # Postal code
    
    @field_validator('sin')
    def validate_sin(cls, v, info):
        # Only validate SIN if equifax_consent is True
        if info.data.get('equifax_consent', False):
            if v is None or v == '':
                raise ValueError('SIN is required when consenting to Equifax check')
            if len(v) != 9 or not v.isdigit():
                raise ValueError('SIN must be a 9-digit number')
        return v
    
    @field_validator('date_of_birth')
    def validate_date_of_birth(cls, v, info):
        # Only validate date_of_birth if equifax_consent is True
        if info.data.get('equifax_consent', False):
            if v is None or v == '':
                raise ValueError('Date of birth is required when consenting to Equifax check')
            import re
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
                raise ValueError('Date of birth must be in YYYY-MM-DD format')
        return v
    
    @field_validator('street_address', 'city', 'postal_code')
    def validate_address_fields(cls, v, info):
        # Only validate address fields if equifax_consent is True
        if info.data.get('equifax_consent', False):
            if v is None or v == '':
                raise ValueError(f'{info.field_name} is required when consenting to Equifax check')
        return v

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
    
class ExplanationFactor(BaseModel):
    feature: str
    impact: float
    direction: str

class LoanExplanation(BaseModel):
    technical_details: Dict[str, Any]
    rejection_probability: float
    main_factors: List[str]
    improvement_suggestions: List[str]

class LoanPredictionResponse(BaseModel):
    loan_approved: bool
    approved_amount: float
    interest_rate: float
    credit_report: Optional[CreditReport] = None
    explanation: Optional[LoanExplanation] = None
    approval_probability: float
    rejection_probability: float
