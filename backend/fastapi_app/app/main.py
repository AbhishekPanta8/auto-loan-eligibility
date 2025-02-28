from fastapi import FastAPI
from app.routes.loan import router as loan_router

app = FastAPI()

app.include_router(loan_router)

@app.get("/")
def root():
    return {"message": "Loan Eligibility API is running"}