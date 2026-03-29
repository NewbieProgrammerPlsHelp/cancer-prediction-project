import streamlit as st

st.title("Neural Network Model Information")

st.header("Project Objective")
st.write("""
This part of the project uses a neural network model to classify breast tumors
as benign or malignant. The goal is to compare a neural network approach with
ensemble machine learning.
""")

st.header("Dataset Description")
st.write("""
The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset.
It contains numerical features extracted from breast mass images.
These features represent important medical characteristics of the tumor samples.
""")

st.header("Target Variable")
st.write("""
The model predicts one of two classes:
- Benign
- Malignant
""")

st.header("Data Preparation")
st.write("""
The preprocessing steps include:
- Loading the dataset
- Splitting data into training and testing sets
- Standardizing feature values using StandardScaler
Feature scaling is important for neural network performance and convergence.
""")

st.header("Neural Network Theory")
st.write("""
A neural network is a model inspired by the human brain.
It learns patterns through layers of connected neurons.

In this project, the neural network model is a Multi-Layer Perceptron (MLP),
which is a feedforward neural network commonly used for classification problems.
""")

st.header("Model Architecture")
st.write("""
The neural network uses:
- Hidden Layer 1: 64 neurons
- Hidden Layer 2: 32 neurons
- Output Layer: binary classification output
""")

st.header("Model Development Process")
st.write("""
1. Load the dataset
2. Split into training and testing data
3. Standardize the features
4. Train the MLPClassifier model
5. Evaluate the model on test data
""")

st.header("Model Performance")
st.success("Neural Network Accuracy: 0.9649")

st.header("Dataset References")

st.write("""
**Dataset 1: Breast Cancer Wisconsin Diagnostic Dataset**
- Source: UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- Also available via scikit-learn

**Dataset 2: Lung Cancer Dataset (Survey-Based)**
- Source: Kaggle
- Description: This dataset contains information such as age, smoking habits,
  alcohol consumption, and environmental factors related to lung cancer risk.
- Link: https://www.kaggle.com/datasets/yusufdede/lung-cancer-dataset
""")