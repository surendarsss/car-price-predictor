import streamlit as st
import pandas as pd
import joblib

# Load your trained pipeline
model = joblib.load("final.pkl")

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor")
st.write(
    "Enter your car's engine and body details below. "
    "The model will predict the estimated car price!"
)

st.header("Engine Configuration (binary flags)")
dohcv = st.selectbox(
    "DOHC with Variable Valve Timing (`dohcv`)",
    [0, 1],
    help="1 = True, 0 = False"
)

ohc = st.selectbox(
    "Overhead Camshaft (`ohc`)",
    [0, 1],
    help="1 = True, 0 = False"
)

rotor = st.selectbox(
    "Rotary Engine (`rotor`)",
    [0, 1],
    help="1 = True, 0 = False"
)

four = st.selectbox(
    "4 Cylinders (`four`)",
    [0, 1],
    help="1 = Engine has 4 cylinders, 0 = Not 4 cylinders"
)

eight = st.selectbox(
    "8 Cylinders (`eight`)",
    [0, 1],
    help="1 = Engine has 8 cylinders, 0 = Not 8 cylinders"
)

st.header("Numeric Specs")
enginesize = st.number_input(
    "Engine Size (`enginesize`) [cc]",
    min_value=50.0,
    max_value=500.0,
    value=120.0,
    step=1.0,
    help="Engine displacement in cubic centimeters (cc)"
)

curbweight = st.number_input(
    "Curb Weight (`curbweight`) [lbs]",
    min_value=1000.0,
    max_value=6000.0,
    value=2500.0,
    step=10.0,
    help="Car weight without passengers or cargo, in pounds (lbs)"
)

wheelbase = st.number_input(
    "Wheelbase (`wheelbase`) [inches]",
    min_value=70.0,
    max_value=150.0,
    value=95.7,
    step=0.1,
    help="Distance between front and rear axles, in inches"
)

if st.button("🔮 Predict Car Price"):
    # Build the input DataFrame
    input_data = pd.DataFrame([{
        'dohcv': dohcv,
        'ohc': ohc,
        'rotor': rotor,
        'four': four,
        'eight': eight,
        'enginesize': enginesize,
        'curbweight': curbweight,
        'wheelbase': wheelbase
    }])

    # Make the prediction
    result = model.predict(input_data)[0]

    st.success(f"💡 **Estimated Car Price:** ${result:,.2f}")
    st.info("Note: This is a prediction based on your inputs and trained data.")

st.write("---")
st.caption("🚗 Built with Streamlit • Ridge Regression ML Model")
