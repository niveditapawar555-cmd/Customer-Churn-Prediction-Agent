from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and columns
model = joblib.load("../model/churn_model.pkl")
model_columns = joblib.load("../model/model_columns.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Create dataframe from input
    df = pd.DataFrame([data])

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep correct column order
    df = df[model_columns]

    prediction = model.predict(df)

    return jsonify({
        "churn_prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)