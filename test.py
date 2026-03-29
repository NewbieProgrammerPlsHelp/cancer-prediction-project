import pandas as pd

df = pd.read_csv("dataset/lung_cancer.csv")
print(df.shape)
print(df["Result"].value_counts())
print("Duplicate rows:", df.duplicated().sum())
print(df.head())
print(df.columns)