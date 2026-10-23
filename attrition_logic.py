def predict_attrition(job_satisfaction, work_life_balance):
    if job_satisfaction <= 2 and work_life_balance <= 2:
        return "ATTRITION_RISK"
    else:
        return "STAY"


if __name__ == "__main__":
    result = predict_attrition(1, 2)
    print("Predicted Employee Status:", result)
