import streamlit as st

st.set_page_config(page_title="Cancer Prediction Project", layout="wide")

st.title("Cancer Prediction Web Application")

st.write("""
This web application presents a breast cancer classification project using two artificial intelligence approaches:
""")

st.markdown("""
- **Ensemble Machine Learning**
- **Neural Network**
""")

st.subheader("Project Overview")
st.write("""
The purpose of this project is to classify breast tumors as benign or malignant
based on diagnostic features. The website includes:
- explanation pages for both models
- test pages for real-time prediction
- model comparison through performance results
""")

st.subheader("Models Included")
st.write("""
1. Ensemble Model:
   Logistic Regression + Random Forest + Support Vector Machine

2. Neural Network Model:
   Multi-Layer Perceptron (MLP)
""")

st.info("Use the sidebar to explore the project pages.")