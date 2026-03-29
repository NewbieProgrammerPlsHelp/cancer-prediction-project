# Cancer Prediction Web Application

## Project Description
This project is a web application for cancer classification using two artificial intelligence approaches:
- Ensemble Machine Learning
- Neural Network

The goal is to predict whether a tumor is benign or malignant based on diagnostic features.

This project also demonstrates how dataset size and complexity affect model performance, as seen in the difference between the breast cancer dataset and the smaller lung cancer dataset.

---

## Models Used

### 1. Ensemble Machine Learning
- Logistic Regression
- Random Forest
- Support Vector Machine
- Combined using Voting Classifier

### 2. Neural Network
- Multi-Layer Perceptron (MLPClassifier)

---

## Datasets

### Dataset 1: Breast Cancer Wisconsin Diagnostic Dataset
- Source: UCI Machine Learning Repository
- Available via: scikit-learn
- Description: Contains features computed from digitized images of breast mass samples

### Dataset 2: Lung Cancer Dataset (Survey-Based)
- Source: Kaggle
- Description: Contains features such as age, smoking habits, alcohol consumption, and environmental factors
- Note: This dataset is relatively small and simple, which may result in very high model accuracy

---

## Performance

### Breast Cancer Dataset
- Ensemble Accuracy: 0.9737
- Neural Network Accuracy: 0.9649

### Lung Cancer Dataset
- Accuracy may reach up to 1.0 due to dataset simplicity

---

## Project Structure
```text
cancer_project/
├─ app.py
├─ dataset/
├─ models/
├─ train/
└─ pages/
```

---

## How to Run
```bash
streamlit run app.py
```

---

## Requirements
```bash
pip install -r requirements.txt
```
