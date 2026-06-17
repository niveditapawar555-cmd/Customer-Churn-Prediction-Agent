import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)

print(X.columns.tolist())