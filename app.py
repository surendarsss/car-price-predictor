import streamlit as st
import pandas as pd
import joblib

# Load your trained pipeline
model = joblib.load("final.pkl")

st.title("🚗 Car Price Predictor")

st.write("Enter your car's details below:")

# User inputs for all features you trained on:
dohcv = st.selectbox("DOHC Variable Valve Timing (dohcv)", [0, 1])
ohc = st.selectbox("Overhead Camshaft (ohc)", [0, 1])
rotor = st.selectbox("Rotary Engine (rotor)", [0, 1])
four = st.selectbox("4 Cylinders (four)", [0, 1])
eight = st.selectbox("8 Cylinders (eight)", [0, 1])
enginesize = st.number_input("Engine Size", min_value=50.0, max_value=500.0, value=120.0)
curbweight = st.number_input("Curb Weight", min_value=1000.0, max_value=5000.0, value=2500.0)
wheelbase = st.number_input("Wheelbase", min_value=70.0, max_value=150.0, value=95.7)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'dohcv': dohc,
        'ohc': ohc,
        'rotor': rotor,
        'four': four,
        'eight': eight,
        'enginesize': enginesize,
        'curbweight': curbweight,
        'wheelbase': wheelbase
    }])
    
    result = model.predict(input_data)[0]
    st.success(f"💡 Predicted Car Price: ${result:.2f}")
