# Cancer Prediction Web Application

## Project Description
This project is a web application for breast cancer classification using two artificial intelligence approaches:
- Ensemble Machine Learning
- Neural Network

The goal is to predict whether a tumor is benign or malignant based on diagnostic features.

## Models Used
### 1. Ensemble Machine Learning
- Logistic Regression
- Random Forest
- Support Vector Machine
- Combined using Voting Classifier

### 2. Neural Network
- Multi-Layer Perceptron (MLPClassifier)

## Dataset
- Breast Cancer Wisconsin Diagnostic Dataset from scikit-learn

## Performance
- Ensemble Accuracy: 0.9737
- Neural Network Accuracy: 0.9649

## How to Run
```bash
streamlit run app.py