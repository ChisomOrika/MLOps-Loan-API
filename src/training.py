import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np

def train_and_save_model():
    """Trains a Logistic Regression model for loan default prediction and saves the pipeline."""
    
    # --- 1. Load Data ---
    data_path = 'data/training_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}. Please create the file.")
        return

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # --- 2. Define Preprocessing Steps (Edge Case: Handling categorical and numerical) ---
    numerical_features = ['feature_1', 'feature_2', 'loan_amount', 'applicant_age']
    categorical_features = ['employment_type']
    
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        # Edge Case: handle_unknown='ignore' prevents API crashes on unseen categories
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])
    
    # Create the Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not specified
    )
    
    # --- 3. Create and Train Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Splitting data (though in MLOps, we typically use the full dataset for deployment model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate (optional, but good practice)
    score = model_pipeline.score(X_test, y_test)
    print(f"Model trained. Accuracy on test set: {score:.4f}")
    
    # --- 4. Save Model ---
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(model_pipeline, os.path.join(model_dir, 'loan_model.pkl'))
    print(f"Model pipeline saved to {model_dir}/loan_model.pkl")

if __name__ == '__main__':
    train_and_save_model()
