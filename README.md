A. Initial Setup

1. Dependencies: Ensure Python 3.10+ and Docker are installed.

2. PostgreSQL: Start a local PostgreSQL instance and create a database named mlops_db.

3. Environment File: Create a file named .env in the project root:


` Change the user/password/host as needed for your local PG instance
DB_URL="postgresql://user:password@localhost:5432/mlops_db" `

4. Install Python requirements: pip install -r requirements.txt

B. Prepare Model and Database
Create Log Table: Execute the SQL code from data/sql/03_log_table_schema.sql against your mlops_db.

Train Model: Run the training script to generate the production model:
python src/train.py

C. Run the Service (Local)
Start API: Run the FastAPI server:

uvicorn api.main:app --reload
Test Endpoint: Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and test the /predict endpoint.

Good Request: {"feature_1": 0.5, "feature_2": 1.2, "loan_amount": 10000, "applicant_age": 35, "employment_type": "Professional"}

Edge Case Request (Pydantic Validation Test): {"feature_1": -1.0, "feature_2": 1.2, "loan_amount": 5000, "applicant_age": 15, "employment_type": "Freelancer"} (This should fail with a 422 error due to Pydantic schemas).

D. Run the Monitoring Job
While the API is running (and has logged some predictions), run the monitoring script:

python src/monitor.py
