
import json
import unittest

from app import app


class TestEmployeeAttritionAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_prediction(self):
        payload = {
            "age": 35,
            "experience_years": 4,
            "salary": 65000,
            "department": "Sales",
            "education": "Bachelors",
            "job_satisfaction": 3,
            "work_life_balance": 3,
            "training_hours": 20
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("attrition_prediction", response.get_json())

    def test_missing_features(self):
        response = self.client.post(
            "/predict",
            json={"age": 35}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
