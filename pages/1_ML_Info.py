import streamlit as st

st.title("Machine Learning Model Information")

st.header("Project Objective")
st.write("""
The objective of this project is to predict whether a breast tumor is benign or malignant
using machine learning techniques. This helps demonstrate how artificial intelligence can
support medical data analysis and classification tasks.
""")

st.header("Dataset Description")
st.write("""
This project uses the Breast Cancer Wisconsin Diagnostic Dataset from scikit-learn.
The dataset contains features computed from digitized images of a breast mass.
These features describe characteristics of the cell nuclei, such as radius, texture,
perimeter, area, smoothness, compactness, concavity, and symmetry.
""")

st.header("Target Variable")
st.write("""
The target variable has two classes:
- Benign
- Malignant
""")

st.header("Data Preparation")
st.write("""
The following preprocessing steps were applied:
- Loaded the dataset
- Split the data into training and testing sets
- Standardized the feature values using StandardScaler
This scaling step is important because the features have different numerical ranges.
""")

st.header("Algorithms Used")
st.write("""
The ensemble machine learning model combines three algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

These models are combined using a soft Voting Classifier.
""")

st.header("Why Ensemble Learning?")
st.write("""
Ensemble learning improves prediction performance by combining multiple models.
Instead of relying on only one algorithm, the final prediction is made from the
combined strengths of all three models.
""")

st.header("Model Development Process")
st.write("""
1. Load the dataset
2. Split data into training and testing sets
3. Standardize the features
4. Train Logistic Regression, Random Forest, and SVM
5. Combine them using Voting Classifier
6. Evaluate the model using test data
""")

st.header("Model Performance")
st.success("Ensemble Accuracy: 0.9737")

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