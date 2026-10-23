import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "ok"
    assert data["service"] == "employee-attrition-prediction"


def test_prediction(client):
    employee_data = {
        "age": 30,
        "experience_years": 5,
        "salary": 50000,
        "department": "IT",
        "education": "Bachelor",
        "job_satisfaction": 4,
        "work_life_balance": 4,
        "training_hours": 20
    }

    response = client.post(
        "/predict",
        json=employee_data
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data


def test_missing_field(client):
    employee_data = {
        "age": 30,
        "experience_years": 5,
        "salary": 50000
    }

    response = client.post(
        "/predict",
        json=employee_data
    )

    assert response.status_code == 400
