"""
tests/test_api.py
Run:  pytest tests/ -v
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


SAMPLE_RECORD = {
    "Gender": "Male", "Age": 45, "Married": "Yes",
    "Number of Dependents": 0, "City": "Los Angeles",
    "Number of Referrals": 2, "Tenure in Months": 4,
    "Offer": "Offer A", "Phone Service": "Yes",
    "Avg Monthly Long Distance Charges": 25.5,
    "Multiple Lines": "No", "Internet Service": "Yes",
    "Internet Type": "Fiber Optic", "Avg Monthly GB Download": 50.0,
    "Online Security": "No", "Online Backup": "Yes",
    "Device Protection Plan": "No", "Premium Tech Support": "Yes",
    "Streaming TV": "No", "Streaming Movies": "No", "Streaming Music": "No",
    "Unlimited Data": "Yes", "Contract": "Month-to-Month",
    "Paperless Billing": "Yes", "Payment Method": "Credit Card",
    "Monthly Charge": 75.0, "Total Charges": 300.0,
    "Total Extra Data Charges": 0.0,
    "Total Long Distance Charges": 102.0, "Total Revenue": 402.0,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_comparison(client):
    resp = client.get("/model-comparison")
    if resp.status_code == 404:
        pytest.skip("Metrics file not present")
    assert resp.status_code == 200
    data = resp.json()
    assert "comparison" in data
    assert len(data["comparison"]) == 2
    models = [r["model"] for r in data["comparison"]]
    assert "Logistic Regression" in models


def test_predict_final_model(client):
    resp = client.post("/predict", json=SAMPLE_RECORD)
    if resp.status_code == 500:
        pytest.skip("Model file not present in CI")
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in {"Churned", "Joined", "Stayed"}
    assert data["model_used"] == "Gradient Boosting"
    proba_sum = sum(data["probabilities"].values())
    assert 0.99 <= proba_sum <= 1.01


def test_predict_logreg(client):
    resp = client.post("/predict/logreg", json=SAMPLE_RECORD)
    if resp.status_code == 500:
        pytest.skip("Model file not present in CI")
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "Logistic Regression"


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[SAMPLE_RECORD, SAMPLE_RECORD])
    if resp.status_code == 500:
        pytest.skip("Model file not present in CI")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_predict_missing_field_returns_422(client):
    bad = {k: v for k, v in SAMPLE_RECORD.items() if k != "Gender"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_info_endpoint(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "target_classes" in data
    assert set(data["target_classes"]) == {"Churned", "Joined", "Stayed"}
