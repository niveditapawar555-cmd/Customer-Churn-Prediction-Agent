# Customer Churn Prediction Agent

## Project Overview
This project predicts whether a telecom customer is likely to churn using Machine Learning.

## Technologies Used
- Python
- Pandas
- Scikit-Learn
- Flask
- n8n
- Joblib

## Workflow
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Flask API Development
6. Workflow Automation using n8n

## Model Performance
Accuracy: 78.54%

## API Example

POST /predict

Input:

{
  "tenure": 2,
  "MonthlyCharges": 75
}

Output:

{
  "churn_prediction": 1
}

## Project Structure

Customer_Churn_Project/
├── data
├── notebooks
├── model
├── api
└── README.md