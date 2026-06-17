import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

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
y = df["Churn"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "../model/churn_model.pkl")
joblib.dump(X.columns.tolist(), "../model/model_columns.pkl")

print("Model and columns saved!")