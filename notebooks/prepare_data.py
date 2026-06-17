import pandas as pd

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# Remove customerID
df = df.drop("customerID", axis=1)

# Convert target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())