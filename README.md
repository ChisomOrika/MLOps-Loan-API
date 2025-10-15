# 🚀 MLOps Loan Default Prediction API

This project demonstrates a robust, end-to-end MLOps pipeline using **FastAPI** for real-time predictions, **PostgreSQL** for logging and monitoring, and **scikit-learn** for the model, all containerized with **Docker**.

---

## 🛠️ Setup and Execution

### A. Initial Setup

1.  **Dependencies:** Ensure **Python 3.10+** and **Docker** are installed on your system.

2.  **PostgreSQL:** Start a local **PostgreSQL** instance and create an empty database named `mlops_db`.

3.  **Environment File:** Create a file named **.env** in the project root to configure your database connection.

    ```env
    # Change the user/password/host as needed for your local PG instance
    DB_URL="postgresql://user:password@localhost:5432/mlops_db"
    ```

4.  **Install Python requirements:**

    ```bash
    pip install -r requirements.txt
    ```

---

### B. Prepare Model and Database

1.  **Create Log Table:** Execute the SQL code from `data/sql/03_log_table_schema.sql` against your `mlops_db` to set up the prediction logging table.

2.  **Train Model:** Run the training script to generate the production model (`loan_model.pkl`), which includes preprocessing steps.

    ```bash
    python src/train.py
    ```

---

### C. Run the Service (Local)

1.  **Start API:** Run the FastAPI server locally. The `--reload` flag allows for development changes.

    ```bash
    uvicorn api.main:app --reload
    ```

2.  **Test Endpoint:** Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and test the `/predict` endpoint.

    * **✅ Good Request:** (This should return a prediction and log the request)
        ```json
        {
          "feature_1": 0.5,
          "feature_2": 1.2,
          "loan_amount": 10000,
          "applicant_age": 35,
          "employment_type": "Professional"
        }
        ```

    * **❌ Edge Case Request (Pydantic Validation Test):** (This should fail with a **422 Unprocessable Entity** error, demonstrating robust input validation via Pydantic schemas, as the age is too low and feature\_1 is negative).
        ```json
        {
          "feature_1": -1.0,
          "feature_2": 1.2,
          "loan_amount": 5000,
          "applicant_age": 15,
          "employment_type": "Freelancer"
        }
        ```

---

### D. Run the Monitoring Job

While the API is running (and has logged some predictions to PostgreSQL), run the monitoring script to check for data and prediction drift.

```bash
python src/monitor.py
