import pickle
from pathlib import Path

import numpy as np
import streamlit as st

MODEL_PATH = Path(__file__).parent / "Diabetesmodel.pkl"

st.set_page_config(page_title="AI Diabetes Risk Assessment", page_icon="🩺")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


st.title("AI Diabetes Risk Assessment")
st.write(
    "Enter the patient's health metrics below, then press **Predict** to "
    "estimate the likelihood of diabetes."
)

if not MODEL_PATH.exists():
    st.error("Diabetesmodel.pkl was not found. Run `python train.py` first to create it.")
    st.stop()

model = load_model()

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=None,
            placeholder="e.g. 120",
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.0, max_value=80.0, value=None,
            step=0.1, format="%.1f", placeholder="e.g. 25.0",
        )
    with col2:
        blood_pressure = st.number_input(
            "Blood Pressure (diastolic, mm Hg)", min_value=0.0, max_value=250.0,
            value=None, placeholder="e.g. 70",
        )
        age = st.number_input(
            "Age (years)", min_value=0, max_value=120, value=None, step=1,
            placeholder="e.g. 30",
        )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    inputs = {
        "Glucose": glucose,
        "Blood Pressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
    }
    missing = [name for name, value in inputs.items() if value is None]
    not_positive = [name for name, value in inputs.items() if value is not None and value <= 0]

    if missing:
        st.error(f"Please enter a value for: {', '.join(missing)}.")
    elif not_positive:
        st.error(f"{', '.join(not_positive)} must be greater than zero.")
    else:
        input_data = np.asarray(list(inputs.values()), dtype=float).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("You are **likely** to have diabetes.")
        else:
            st.success("You are **not likely** to have diabetes.")
        st.metric("Estimated risk", f"{probability * 100:.0f} %")
        st.progress(float(probability))

st.caption(
    "This tool is a classroom project for educational purposes only. It is not "
    "a medical diagnosis — please consult a healthcare professional."
)
