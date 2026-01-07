import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'indian_liver_patient.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'liver_model.pkl')

def train_liver_model():
    print("Loading Liver Disease Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Preprocessing
    # Headers: Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Dataset
    
    # Handle missing values
    # Albumin_and_Globulin_Ratio sometimes has NaNs
    imputer = SimpleImputer(strategy='mean')
    # Fill numeric columns with mean
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Encode Gender
    if 'Gender' in df.columns:
        le = LabelEncoder()
        # Handle non-string/missing gender if any, though usually just Male/Female
        df['Gender'] = df['Gender'].astype(str)
        df['Gender'] = le.fit_transform(df['Gender'])

    # Target: Dataset (1: Liver Patient, 2: Non Liver Patient - verify this assumption or just map)
    # Typically 1 is yes, 2 is no. Let's map 2 -> 0, 1 -> 1.
    if 'Dataset' in df.columns:
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
        # If map fails (e.g. if values are floats 1.0/2.0), handle that
        # But after imputation they might be floats.
        # Let's check and ensure they are 0/1.
        # Recalculate mapping if imputation messed it up (it shouldn't if we treat it as numeric, but let's be safe)
        # Actually imputation on Dataset (target) is bad practice. We should separate X and y first or drop rows with missing target.
        # Assuming Dataset has no missing values.
        pass

    # Re-split X and y
    # But wait, I imputed everything including target in the block above if 'Dataset' is numeric.
    # Let's refine the logic.
    
    # Reload to be safe
    df = pd.read_csv(DATA_PATH)
    
    # Drop rows with missing target if any (unlikely for this dataset but good practice)
    df.dropna(subset=['Dataset'], inplace=True)
    
    # Encode Target: 1 -> 1, 2 -> 0
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
    
    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'].astype(str))
    
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    # Impute missing values in Features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Model for Liver Disease...")
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
    train_liver_model()
