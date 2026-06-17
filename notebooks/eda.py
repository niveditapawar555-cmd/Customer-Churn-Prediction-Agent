import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

print(df["Churn"].value_counts())

print("\nPercentage:")
print(
    round(
        df["Churn"].value_counts(normalize=True) * 100,
        2
    )
)

print("\nContract vs Churn:\n")

print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)
print("\nAverage Tenure by Churn:\n")

print(
    df.groupby("Churn")["tenure"].mean()
)
print("\nAverage Monthly Charges by Churn:\n")

print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)