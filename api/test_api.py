import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "tenure": 2,
    "MonthlyCharges": 75
}

response = requests.post(url, json=data)

print(response.text)