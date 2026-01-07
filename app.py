from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
import pandas as pd # For DataFrame consistency if needed

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Models
print("Loading models...")
MODELS_DIR = 'models'

def load_pkl_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

heart_model = load_pkl_model('heart_disease_model.pkl')
diabetes_model = load_pkl_model('diabetes_model.pkl')
kidney_model = load_pkl_model('kidney_model.pkl')
liver_model = load_pkl_model('liver_model.pkl')
breast_cancer_model = load_pkl_model('breast_cancer_model.pkl')

# Deep Learning Models (Load lazily or upfront?)
# Upfront is better for API speed, but might consume memory while training.
# We'll load them if they exist.
# Note: Use the NEW filenames we defined in training scripts.
pneumonia_model_path = os.path.join(MODELS_DIR, 'pneumonia_model_new.h5')
malaria_model_path = os.path.join(MODELS_DIR, 'malaria_cnn_model_fast.h5')  # Fast model is compatible with current Keras

pneumonia_model = None
malaria_model = None

if os.path.exists(pneumonia_model_path):
    try:
        pneumonia_model = load_model(pneumonia_model_path)
        print("Pneumonia model loaded.")
    except Exception as e:
        print(f"Error loading Pneumonia model: {e}")

if os.path.exists(malaria_model_path):
    try:
        malaria_model = load_model(malaria_model_path)
        print("Malaria model loaded.")
    except Exception as e:
        print(f"Error loading Malaria model: {e}")

# Disease Insights - Predefined information for each disease
DISEASE_INSIGHTS = {
    'heart_disease': {
        'positive': {
            'title': 'Heart Disease Risk Detected',
            'description': 'Our AI analysis indicates potential cardiovascular risk factors in your profile.',
            'risk_factors': [
                'High blood pressure (Hypertension)',
                'Elevated cholesterol levels',
                'Smoking and tobacco use',
                'Obesity and sedentary lifestyle',
                'Family history of heart disease'
            ],
            'recommendations': [
                'Schedule an appointment with a cardiologist',
                'Monitor blood pressure regularly',
                'Adopt a heart-healthy diet (low sodium, low saturated fat)',
                'Engage in regular physical activity (150 min/week)',
                'Consider stress management techniques'
            ]
        },
        'negative': {
            'title': 'No Heart Disease Indicators',
            'description': 'Based on the provided data, no significant heart disease risk factors were detected.',
            'tips': [
                'Continue maintaining a healthy lifestyle',
                'Regular cardiovascular checkups recommended annually',
                'Keep cholesterol and blood pressure in healthy ranges',
                'Stay physically active and maintain healthy weight'
            ]
        }
    },
    'diabetes': {
        'positive': {
            'title': 'Diabetes Risk Detected',
            'description': 'The analysis suggests elevated risk for diabetes based on your health markers.',
            'risk_factors': [
                'Elevated blood glucose levels',
                'High HbA1c indicating poor glucose control',
                'Obesity (BMI > 30)',
                'Family history of diabetes',
                'Sedentary lifestyle'
            ],
            'recommendations': [
                'Consult an endocrinologist for comprehensive evaluation',
                'Monitor blood glucose levels regularly',
                'Follow a low-glycemic, balanced diet',
                'Increase physical activity to improve insulin sensitivity',
                'Consider weight management if overweight'
            ]
        },
        'negative': {
            'title': 'No Diabetes Indicators',
            'description': 'Your health markers do not indicate significant diabetes risk at this time.',
            'tips': [
                'Maintain a balanced diet with limited sugar intake',
                'Stay physically active',
                'Get annual blood glucose screenings',
                'Monitor weight and BMI regularly'
            ]
        }
    },
    'kidney_disease': {
        'positive': {
            'title': 'Kidney Disease Indicators Present',
            'description': 'The analysis detected markers that may indicate chronic kidney disease.',
            'risk_factors': [
                'Diabetes and high blood pressure',
                'Abnormal protein levels in urine',
                'Reduced glomerular filtration rate',
                'History of kidney infections',
                'Family history of kidney disease'
            ],
            'recommendations': [
                'Consult a nephrologist immediately',
                'Monitor kidney function tests regularly',
                'Control blood pressure and blood sugar',
                'Limit sodium and protein intake as advised',
                'Stay well hydrated with water'
            ]
        },
        'negative': {
            'title': 'Kidney Function Appears Normal',
            'description': 'No significant indicators of kidney disease were detected in your analysis.',
            'tips': [
                'Stay hydrated and drink adequate water',
                'Limit sodium intake',
                'Avoid overuse of NSAIDs and painkillers',
                'Get annual kidney function tests if at risk'
            ]
        }
    },
    'liver_disease': {
        'positive': {
            'title': 'Liver Disease Risk Detected',
            'description': 'Elevated liver enzyme levels or markers suggest potential liver concerns.',
            'risk_factors': [
                'Excessive alcohol consumption',
                'Hepatitis B or C infection',
                'Obesity and fatty liver disease',
                'Certain medications and toxins',
                'Autoimmune conditions'
            ],
            'recommendations': [
                'Consult a hepatologist or gastroenterologist',
                'Avoid alcohol completely',
                'Get tested for hepatitis infections',
                'Maintain a healthy weight',
                'Review medications with your doctor'
            ]
        },
        'negative': {
            'title': 'Liver Function Appears Normal',
            'description': 'Your liver enzyme levels appear within normal ranges.',
            'tips': [
                'Limit alcohol consumption',
                'Maintain a healthy weight',
                'Avoid unnecessary medications',
                'Get vaccinated for Hepatitis A and B'
            ]
        }
    },
    'breast_cancer': {
        'malignant': {
            'title': 'Malignant Tumor Characteristics Detected',
            'description': 'The cellular analysis suggests malignant characteristics. Immediate medical consultation is essential.',
            'next_steps': [
                'Consult an oncologist immediately',
                'Additional imaging (mammogram, MRI) may be needed',
                'Biopsy for definitive diagnosis',
                'Discuss treatment options with your medical team',
                'Seek support from cancer support groups'
            ]
        },
        'benign': {
            'title': 'Benign Tumor Characteristics',
            'description': 'The cellular analysis suggests benign (non-cancerous) characteristics.',
            'tips': [
                'Continue regular breast self-examinations',
                'Schedule routine mammograms as recommended',
                'Monitor for any changes in breast tissue',
                'Maintain a healthy lifestyle'
            ]
        }
    },
    'malaria': {
        'infected': {
            'title': 'Parasitized Cells Detected',
            'description': 'The blood smear analysis detected cells infected with malaria parasites.',
            'about': 'Malaria is caused by Plasmodium parasites transmitted through infected mosquito bites.',
            'symptoms': [
                'High fever and chills',
                'Severe headache',
                'Muscle and joint pain',
                'Fatigue and weakness',
                'Nausea and vomiting'
            ],
            'recommendations': [
                'Seek immediate medical treatment',
                'Antimalarial medication is essential',
                'Rest and stay hydrated',
                'Complete the full course of treatment',
                'Use mosquito prevention measures'
            ]
        },
        'uninfected': {
            'title': 'No Malaria Parasites Detected',
            'description': 'The blood smear analysis did not detect malaria-infected cells.',
            'tips': [
                'Continue using mosquito repellents',
                'Sleep under insecticide-treated bed nets',
                'Seek medical attention if symptoms develop',
                'Consider antimalarial prophylaxis when traveling to endemic areas'
            ]
        }
    },
    'pneumonia': {
        'positive': {
            'title': 'Pneumonia Indicators Detected',
            'description': 'The chest X-ray analysis suggests patterns consistent with pneumonia.',
            'about': 'Pneumonia is an infection that inflames air sacs in the lungs, which may fill with fluid.',
            'symptoms': [
                'Persistent cough with phlegm',
                'Fever and chills',
                'Difficulty breathing',
                'Chest pain when breathing',
                'Fatigue and weakness'
            ],
            'recommendations': [
                'Consult a pulmonologist or physician immediately',
                'Antibiotics may be prescribed if bacterial',
                'Rest and stay well hydrated',
                'Complete the full course of medication',
                'Get pneumonia vaccine to prevent future infections'
            ]
        },
        'negative': {
            'title': 'Lungs Appear Normal',
            'description': 'The chest X-ray analysis did not detect pneumonia patterns.',
            'tips': [
                'Practice good hand hygiene',
                'Get annual flu vaccination',
                'Consider pneumococcal vaccine if at risk',
                'Avoid smoking and secondhand smoke'
            ]
        }
    }
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- Heart Disease ---
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            # Get values from form
            # Features: Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['cp']),
                float(request.form['bp']),
                float(request.form['chol']),
                float(request.form['fbs']),
                float(request.form['ekg']),
                float(request.form['max_hr']),
                float(request.form['exang']),
                float(request.form['oldpeak']),
                float(request.form['slope']),
                float(request.form['ca']),
                float(request.form['thal'])
            ]
            
            prediction = heart_model.predict([features])[0]
            result = "Presence" if prediction == 1 else "Absence"
            insight_key = 'positive' if prediction == 1 else 'negative'
            insights = DISEASE_INSIGHTS['heart_disease'][insight_key]
            return render_template('heart.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('heart.html', error=str(e))
    return render_template('heart.html')

# --- Diabetes ---
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            # Features: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
            # Model Columns: ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever', 'smoking_history_former', 'smoking_history_never', 'smoking_history_not current']
            
            # 1. Base Features
            data = {
                'age': float(request.form['age']),
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'bmi': float(request.form['bmi']),
                'HbA1c_level': float(request.form['HbA1c_level']),
                'blood_glucose_level': float(request.form['blood_glucose_level']),
                'gender_Male': 0,
                'gender_Other': 0,
                'smoking_history_current': 0,
                'smoking_history_ever': 0,
                'smoking_history_former': 0,
                'smoking_history_never': 0,
                'smoking_history_not current': 0
            }
            
            # 2. Handle Gender
            gender = request.form['gender']
            if gender == 'Male':
                data['gender_Male'] = 1
            elif gender == 'Other':
                data['gender_Other'] = 1
            # Female is default (0,0)
            
            # 3. Handle Smoking
            smoking = request.form['smoking_history']
            if smoking == 'current':
                data['smoking_history_current'] = 1
            elif smoking == 'ever':
                data['smoking_history_ever'] = 1
            elif smoking == 'former':
                data['smoking_history_former'] = 1
            elif smoking == 'never':
                data['smoking_history_never'] = 1
            elif smoking == 'not current':
                data['smoking_history_not current'] = 1
            # 'No Info' is default (all 0)
            
            # Create feature vector in correct order
            feature_order = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                             'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever', 
                             'smoking_history_former', 'smoking_history_never', 'smoking_history_not current']
            
            feature_vector = [data[col] for col in feature_order]
            
            prediction = diabetes_model.predict([feature_vector])[0]
            result = "Positive" if prediction == 1 else "Negative"
            insight_key = 'positive' if prediction == 1 else 'negative'
            insights = DISEASE_INSIGHTS['diabetes'][insight_key]
            return render_template('diabetes.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('diabetes.html', error=str(e))
    return render_template('diabetes.html')

# --- Kidney Disease ---
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            # Inputs matching form names
            # Feature order logic from training script (cleaned):
            # age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
            # Encoding assumptions: LabelEncoder sorts alphabetically.
            # rbc: abnormal(0), normal(1) -> Form sends 0/1 directly?
            # Form sends: rbc: 1=Normal, 0=Abnormal. (Matches Normal > Abnormal? 'n' > 'a'. Yes. 1=Normal, 0=Abnormal)
            # pcc: notpresent(0), present(1). ('n' < 'p'). Correct.
            # ba: notpresent(0), present(1). Correct.
            # htn: no(0), yes(1). Correct.
            # dm: no(0), yes(1). Correct.
            # cad: no(0), yes(1). Correct.
            # appet: good(0), poor(1). ('g' < 'p'). Correct.
            # pe: no(0), yes(1). Correct.
            # ane: no(0), yes(1). Correct.
            
            features = [
                float(request.form['age']),
                float(request.form['bp']),
                float(request.form['sg']),
                float(request.form['al']),
                float(request.form['su']),
                float(request.form['rbc']), # 0 or 1
                float(request.form['pc']),
                float(request.form['pcc']),
                float(request.form['ba']),
                float(request.form['bgr']),
                float(request.form['bu']),
                float(request.form['sc']),
                float(request.form['sod']),
                float(request.form['pot']),
                float(request.form['hemo']),
                float(request.form['pcv']),
                float(request.form['wc']),
                float(request.form['rc']),
                float(request.form['htn']),
                float(request.form['dm']),
                float(request.form['cad']),
                float(request.form['appet']),
                float(request.form['pe']),
                float(request.form['ane'])
            ]
            
            prediction = kidney_model.predict([features])[0]
            # My training script: confirmed valid targets 0/1. 
            # I mapped 'ckd' -> 1. So 1 is Positive/Disease.
            result = "Positive" if prediction == 1 else "Negative"
            insight_key = 'positive' if prediction == 1 else 'negative'
            insights = DISEASE_INSIGHTS['kidney_disease'][insight_key]
            return render_template('kidney.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('kidney.html', error=str(e))
    return render_template('kidney.html')

# --- Liver Disease ---
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            # Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
            # Gender: Male/Female. Training: LabelEncoder. 'Female' < 'Male' -> Female=0, Male=1.
            gender_val = 1 if request.form['Gender'] == 'Male' else 0
            
            features = [
                float(request.form['Age']),
                gender_val,
                float(request.form['Total_Bilirubin']),
                float(request.form['Direct_Bilirubin']),
                float(request.form['Alkaline_Phosphotase']),
                float(request.form['Alamine_Aminotransferase']),
                float(request.form['Aspartate_Aminotransferase']),
                float(request.form['Total_Protiens']),
                float(request.form['Albumin']),
                float(request.form['Albumin_and_Globulin_Ratio'])
            ]
            
            prediction = liver_model.predict([features])[0]
            # Mapped 1->1 (Disease), 2->0 (No Disease).
            result = "High Probability" if prediction == 1 else "Low Probability"
            insight_key = 'positive' if prediction == 1 else 'negative'
            insights = DISEASE_INSIGHTS['liver_disease'][insight_key]
            return render_template('liver.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('liver.html', error=str(e))
    return render_template('liver.html')

# --- Breast Cancer ---
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        try:
            # 5 features matching the training data
            features = [
                float(request.form['mean_radius']),
                float(request.form['mean_texture']),
                float(request.form['mean_perimeter']),
                float(request.form['mean_area']),
                float(request.form['mean_smoothness'])
            ]
            
            prediction = breast_cancer_model.predict([features])[0]
            # In training data: 0 = Malignant, 1 = Benign
            result = "Malignant" if prediction == 0 else "Benign"
            insight_key = 'malignant' if prediction == 0 else 'benign'
            insights = DISEASE_INSIGHTS['breast_cancer'][insight_key]
            return render_template('breast_cancer.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('breast_cancer.html', error=str(e))
    return render_template('breast_cancer.html')

# --- Malaria ---
@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('malaria.html', error="No file part")
            file = request.files['file']
            if file.filename == '':
                return render_template('malaria.html', error="No selected file")
            
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Use the properly trained malaria model
                if malaria_model is None:
                    return render_template('malaria.html', error="Malaria model not loaded")

                # Preprocess - model uses 50x50 input (from original Kaggle notebook)
                img = keras_image.load_img(filepath, target_size=(50, 50))
                img_array = keras_image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                pred = malaria_model.predict(img_array)
                print(f"DEBUG Malaria Prediction: {pred}")
                
                # The model uses categorical crossentropy with 2 classes:
                # Class 0 = Uninfected, Class 1 = Parasitized (Infected)
                # Output is [prob_uninfected, prob_parasitized]
                pred_class = np.argmax(pred[0])  # Get class with highest probability
                result = "Infected" if pred_class == 1 else "Uninfected"
                insight_key = 'infected' if pred_class == 1 else 'uninfected'
                insights = DISEASE_INSIGHTS['malaria'][insight_key]
                return render_template('malaria.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('malaria.html', error=str(e))
    return render_template('malaria.html')

# --- Pneumonia ---
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('pneumonia.html', error="No file part")
            file = request.files['file']
            if file.filename == '':
                return render_template('pneumonia.html', error="No selected file")
            
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                if pneumonia_model is None:
                     return render_template('pneumonia.html', error="Model not loaded")

                # Preprocess
                # Trained with 150x150
                img = keras_image.load_img(filepath, target_size=(150, 150))
                img_array = keras_image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                pred = pneumonia_model.predict(img_array)
                # Indices: NORMAL, PNEUMONIA (Alphabetical: NORMAL=0, PNEUMONIA=1).
                # Pred > 0.5 -> 1 (Pneumonia).
                
                result = "Pneumonia Detected" if pred[0][0] > 0.5 else "Normal"
                insight_key = 'positive' if pred[0][0] > 0.5 else 'negative'
                insights = DISEASE_INSIGHTS['pneumonia'][insight_key]
                return render_template('pneumonia.html', prediction=result, insights=insights)
        except Exception as e:
            return render_template('pneumonia.html', error=str(e))
    return render_template('pneumonia.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
