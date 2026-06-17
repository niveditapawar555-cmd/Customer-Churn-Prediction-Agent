from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and columns
model = joblib.load(os.path.join(BASE_DIR, "../model/churn_model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "../model/model_columns.pkl"))

@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    prediction = model.predict(df)

    return jsonify({
        "churn_prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)