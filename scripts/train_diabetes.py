import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_prediction_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_model.pkl')

def train_diabetes_model():
    print("Loading Diabetes Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Preprocessing
    # Headers: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
    
    # Handle categorical variables: gender, smoking_history
    # Using pd.get_dummies for one-hot encoding
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
    
    # Target column is 'diabetes'
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Model for Diabetes...")
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
    train_diabetes_model()
