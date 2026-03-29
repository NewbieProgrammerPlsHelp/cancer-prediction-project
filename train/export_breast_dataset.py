from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

# Save to CSV
df.to_csv("dataset/breast_cancer.csv", index=False)

print("✅ Breast cancer dataset saved!")