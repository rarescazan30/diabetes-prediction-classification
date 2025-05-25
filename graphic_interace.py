import gradio as gr
import pandas as pd
import numpy as np
import joblib

# incarc modelul meu
model = joblib.load('model_antrenat_clasificare.pkl')
scaler = joblib.load('scaler_model.pkl')


def predict_diabetes(gender, heart_disease, hypertension, age, bmi, HbA1c_level, blood_glucose_level, physical_activity, smoking_history):
    
    # creez singur datele pe care le-am calculat in functie de alti factori ca sa ajut modelul
    is_obese = 1 if bmi >= 30 else 0
    age_group_30_50 = 1 if 0 <= age < 50 else 0
    age_group_50_70 = 1 if 50 <= age < 70 else 0
    age_group_70_plus = 1 if age >= 70 else 0
    # smoke history a fost one hot encoded, deci il transform aici manual
    gender = 1 if gender == 'Woman' else 0
    smoking_ever = 1 if smoking_history == 'ever' else 0
    smoking_former = 1 if smoking_history == 'former' else 0
    smoking_never = 1 if smoking_history == 'never' else 0
    smoking_not_current = 1 if smoking_history == 'not current' else 0
    
    # creez un dataframe cu ce a introdus user-ul
    input_df = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'smoking_history_ever': [smoking_ever],
        'smoking_history_former': [smoking_former],
        'smoking_history_never': [smoking_never],
        'smoking_history_not current': [smoking_not_current],
        'is_obese': [is_obese],
        'age_group_30-50': [age_group_30_50],
        'age_group_50-70': [age_group_50_70],
        'age_group_70+': [age_group_70_plus],
        'physical_activity': [physical_activity],
    })
    
    # normalizez datele cu acelasi scaler ca in dataset_manip.py
    numeric_cols = ['age', 'HbA1c_level', 'blood_glucose_level', 'bmi', 'physical_activity']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    # fac predictia
    pred = model.predict(input_df)[0]
    probabilitatea = model.predict_proba(input_df)[0]
    if pred == 1: pred = "Diabetes predicted!"
    else: pred = "No diabetes predicted!"
    return f"Prediction: {pred}", f"Probability: {probabilitatea}"

# input-urile din interfata gradio
inputs = [
    gr.Dropdown(choices=['Man', 'Woman'], label="Gender"),
    gr.Checkbox(label="Heart Disease"),
    gr.Checkbox(label="Hypertension"),
    gr.Number(label="Age"),
    gr.Number(label="BMI"),
    gr.Number(label="HbA1c Level"),
    gr.Number(label="Blood Glucose Level"),
    gr.Number(label="Physical Activity"),
    gr.Dropdown(choices=['ever', 'former', 'never', 'current', 'not current'], label="Smoking History")
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="Probabilities")
]

demo = gr.Interface(fn=predict_diabetes, inputs=inputs, outputs=outputs, title="Predictie Diabet")

demo.launch(share=True)
