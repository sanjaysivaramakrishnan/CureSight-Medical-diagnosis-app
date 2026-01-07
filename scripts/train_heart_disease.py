import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Heart_Disease_Prediction.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'heart_disease_model.pkl')

def train_heart_disease_model():
    print("Loading Heart Disease Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Preprocessing
    # Target column is 'Heart Disease' with values 'Presence'/'Absence'
    if 'Heart Disease' in df.columns:
        df['Target'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
        X = df.drop(['Heart Disease', 'Target'], axis=1)
        y = df['Target']
    else:
        # Fallback if column name is different/lowercase based on initial inspection
        # 'Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium,Heart Disease'
        print("Columns:", df.columns)
        # Assuming last column is target if name doesn't match exactly
        if 'Heart Disease' not in df.columns:
             # Try determining target from last column
             target_col = df.columns[-1]
             print(f"Assuming '{target_col}' is the target.")
             df['Target'] = df[target_col].map({'Presence': 1, 'Absence': 0})
             X = df.drop([target_col, 'Target'], axis=1)
             y = df['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Model...")
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
    train_heart_disease_model()
