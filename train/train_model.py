import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

# Neural Network (MLP)
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🔵 Ensemble Model
# =========================
lr = LogisticRegression(max_iter=5000)
rf = RandomForestClassifier(n_estimators=200)
svm = SVC(probability=True)

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("svm", svm)
    ],
    voting="soft"
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(ensemble, "models/ensemble_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# =========================
# 🔴 Neural Network (MLP)
# =========================
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)

print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_nn))

# Save model
joblib.dump(mlp, "models/nn_model.pkl")

print("✅ Models saved successfully!")