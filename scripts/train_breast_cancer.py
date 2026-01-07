import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Breast_cancer_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'breast_cancer_model.pkl')

def train_breast_cancer_model():
    print("Loading Breast Cancer Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Preprocessing
    # Target is 'diagnosis' (0 or 1, or M/B)
    # Output shows 0, so likely already numeric or numeric-string.
    
    # Check if diagnosis is numeric
    if 'diagnosis' in df.columns:
        # Just in case it's not numeric
        if df['diagnosis'].dtype == 'object':
             # Maybe M=1, B=0? Or vice versa. Usually M=Malignant=1.
             # Need to assume or check values.
             # Based on previous get-content: 0. So likely already 0/1.
             pass
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Model for Breast Cancer...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_breast_cancer_model()
