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
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'kidney_disease.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'kidney_model.pkl')

def train_kidney_model():
    print("Loading Kidney Disease Data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Data Cleaning
    print("Cleaning Data...")
    
    # Drop id
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    # Handle missing values denoted by '?' or tabs
    df.replace('?', np.nan, inplace=True)
    df.replace('\t?', np.nan, inplace=True) # Just in case

    # Columns that should be numeric
    # pcv: packed cell volume, wc: white blood cell count, rc: red blood cell count
    for col in ['pcv', 'wc', 'rc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Target: classification
    if 'classification' in df.columns:
        # Clean classification column quirks (e.g. 'ckd\t')
        df['classification'] = df['classification'].astype(str).str.strip()
        df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})
        # If any remain (e.g. nan converted to 'nan'), drop or handle
        # Assuming only ckd/notckd are valid targets.
        df = df[df['classification'].isin([0, 1])]
        df['classification'] = df['classification'].astype(int)
    
    # Handle other categorical columns with potential whitespace
    # cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].str.strip()
    
    # Specific cleaning for 'dm' and 'cad' which are known to have typos
    if 'dm' in df.columns:
        df['dm'] = df['dm'].replace({' yes': 'yes', '\tyes': 'yes', ' yes': 'yes', '\tno': 'no'})
    if 'cad' in df.columns:
        df['cad'] = df['cad'].replace({'\tno': 'no'})

    # Split features and target
    X = df.drop('classification', axis=1)
    y = df['classification']

    # Impute missing values
    # Numerical: mean, Categorical: most_frequent
    # We'll split numeric and categorical columns first
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Impute Numeric
    num_imputer = SimpleImputer(strategy='mean')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # Impute Categorical
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # Encode Categorical
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training Random Forest Model for Kidney Disease...")
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
    train_kidney_model()
