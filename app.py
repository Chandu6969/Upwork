import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load model
model = joblib.load("model/heart_disease_model.pkl")

st.title("❤️ Advanced Heart Disease Prediction App")

def user_input():
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting BP (trestbps)", 90, 200, 120)
    chol = st.slider("Cholesterol (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0, 1])
    restecg = st.selectbox("Rest ECG (restecg)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate (thalach)", 60, 200, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]])
    return features

user_data = user_input()

if st.button("Predict"):
    prediction = model.predict(user_data)
    st.success("🔴 High Risk of Heart Disease" if prediction[0] == 1 else "🟢 Low Risk of Heart Disease")
