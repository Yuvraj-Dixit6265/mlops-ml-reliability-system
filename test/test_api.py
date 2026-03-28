from fastapi.testclient import TestClient
from app.main import app
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
client = TestClient(app)

# ✅ Test root
def test_root():
    response = client.get("/")
    assert response.status_code == 200

# ✅ Test prediction
def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()

# ✅ Test drift
def test_drift():
    response = client.get("/drift")

    assert response.status_code == 200
    assert "drift" in response.json()
    assert "alert" in response.json()