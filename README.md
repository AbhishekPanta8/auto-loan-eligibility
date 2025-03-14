# auto-loan-eligibility
An automated system for analyzing customer financial data and providing preliminary eligibility decisions for loans and credit products. Built with Angular, Spring Boot, and ML to streamline pre-qualification and improve decision-making efficiency. 



### start API server
cd backend/fastapi_app && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

### Generate Dataset
cd backend/fastapi_app && python datasets/synthetic_data.py

### train ML model
cd backend/fastapi_app && python ml/train_synthetic.py
