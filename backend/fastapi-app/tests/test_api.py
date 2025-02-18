import requests

API_URL = "http://localhost:8000/predict/"

payload = {
    "age": 30,
    "employment_status": 2,  # 2 = Salaried, 1 = Self-employed, 0 = Unemployed
    "annual_income": 75000,
    "existing_debt": 10000,
    "debt_to_income_ratio": 0.2,
    "credit_score": 720,
    "credit_history_length": 10,
    "missed_payments": 0,
    "loan_amount_requested": 20000,
    "preferred_term_months": 36,
    "collateral_available": 1
}

response = requests.post(API_URL, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
