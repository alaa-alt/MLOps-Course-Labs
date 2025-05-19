from fastapi.testclient import TestClient
from main import app  # or app.py if that's your filename

client = TestClient(app)

def test_predict():
    payload = {
        "CreditScore": 650,
        "Age": 35,
        "Tenure": 3,
        "Balance": 10000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
        "Geography": "Spain",
        "Gender": "Male"
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)