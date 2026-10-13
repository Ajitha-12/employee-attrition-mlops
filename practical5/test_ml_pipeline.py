import json
import os
import unittest
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("employee_attrition.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("employee_attrition_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("employee_attrition_model.pkl")

        sample = pd.DataFrame([{
            "age": 40,
            "experience_years": 15,
            "salary": 100000,
            "department": "IT",
            "education": "Masters",
            "job_satisfaction": 3,
            "work_life_balance": 3,
            "training_hours": 40
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_low_risk_employee(self):
        model = joblib.load("employee_attrition_model.pkl")

        sample = pd.DataFrame([{
            "age": 55,
            "experience_years": 25,
            "salary": 120000,
            "department": "IT",
            "education": "Masters",
            "job_satisfaction": 5,
            "work_life_balance": 5,
            "training_hours": 50
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)

    def test_high_risk_employee(self):
        model = joblib.load("employee_attrition_model.pkl")

        sample = pd.DataFrame([{
            "age": 28,
            "experience_years": 3,
            "salary": 50000,
            "department": "Sales",
            "education": "Bachelors",
            "job_satisfaction": 1,
            "work_life_balance": 1,
            "training_hours": 20
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)


if __name__ == "__main__":
    unittest.main()
