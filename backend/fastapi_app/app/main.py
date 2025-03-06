from fastapi import FastAPI
from app.routes.loan import router as loan_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(loan_router)

@app.get("/")
def root():
    return {"message": "Loan Eligibility API is running"}