import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna()

# Remove ID
df = df.drop("customerID", axis=1)

# Encode target
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "../model/churn_model.pkl")

print("Model saved successfully!")