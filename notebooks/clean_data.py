import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Before cleaning:", df.shape)

df = df.dropna()

print("After cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())