import streamlit as st
import pandas as pd
import joblib

st.title("Test Lung Cancer Model")

st.info("""
This model is trained on a small lung cancer dataset.
Results may appear very accurate due to the simplicity of the dataset.
""")

model = joblib.load("models/lung_model.pkl")
scaler = joblib.load("models/lung_scaler.pkl")

st.write("Enter patient information:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
smokes = st.number_input("Smokes", min_value=0, value=0)
areaq = st.number_input("AreaQ", min_value=0, value=0)
alkhol = st.number_input("Alkhol", min_value=0, value=0)

if st.button("Predict Lung Cancer Risk"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Smokes": smokes,
        "AreaQ": areaq,
        "Alkhol": alkhol
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: High Risk / Positive Result")
    else:
        st.success("Prediction: Low Risk / Negative Result")

    st.write(f"Probability of class 0: {probability[0]:.4f}")
    st.write(f"Probability of class 1: {probability[1]:.4f}")