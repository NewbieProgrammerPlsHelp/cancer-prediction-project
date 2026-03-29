import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

st.title("Test Machine Learning Model")

model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/scaler.pkl")

data = load_breast_cancer()
feature_names = list(data.feature_names)

st.write("Enter the tumor feature values below to test the ensemble machine learning model.")

sample_values = dict(zip(feature_names, data.data[0]))

if st.button("Use Sample Data for ML"):
    for feature in feature_names:
        st.session_state[f"ml_{feature}"] = float(sample_values[feature])

col1, col2 = st.columns(2)
input_data = {}

half = len(feature_names) // 2

for i, feature in enumerate(feature_names):
    target_col = col1 if i < half else col2
    with target_col:
        input_data[feature] = st.number_input(
            feature,
            value=float(st.session_state.get(f"ml_{feature}", 0.0)),
            format="%.6f",
            key=f"ml_input_{feature}"
        )

if st.button("Predict with Ensemble Model"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")

    st.write(f"Malignant Probability: {probability[0]:.4f}")
    st.write(f"Benign Probability: {probability[1]:.4f}")