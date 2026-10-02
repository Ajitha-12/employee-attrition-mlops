import unittest
from attrition_logic import predict_attrition


class TestAttritionLogic(unittest.TestCase):

    def test_attrition_risk(self):
        result = predict_attrition(1, 2)
        self.assertEqual(result, "ATTRITION_RISK")

    def test_stay_due_to_satisfaction(self):
        result = predict_attrition(4, 2)
        self.assertEqual(result, "STAY")

    def test_stay_due_to_work_life_balance(self):
        result = predict_attrition(2, 4)
        self.assertEqual(result, "STAY")


if __name__ == "__main__":
    unittest.main()
