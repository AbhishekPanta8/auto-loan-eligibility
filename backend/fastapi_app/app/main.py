from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.loan import router as loan_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Add CORS middleware with permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(loan_router)

@app.get("/")
def root():
    return {"message": "Loan Eligibility API is running"}