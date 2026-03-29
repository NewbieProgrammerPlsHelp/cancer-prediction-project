import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/lung_cancer.csv")

# Target column
target_column = "Result"

# Drop text columns that are not useful
df = df.drop(columns=["Name", "Surname"])

# Features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
for rs in [1, 7, 21, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rs, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("random_state =", rs, "accuracy =", accuracy_score(y_test, y_pred))

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural Network model
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Lung Cancer Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "models/lung_model.pkl")
joblib.dump(scaler, "models/lung_scaler.pkl")

print("✅ Lung model saved!")