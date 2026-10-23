from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILE = "employee_attrition_model.pkl"

FEATURES = [
    "age",
    "experience_years",
    "salary",
    "department",
    "education",
    "job_satisfaction",
    "work_life_balance",
    "training_hours",
]


def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found."
        )

    return joblib.load(MODEL_FILE)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "employee-attrition-prediction"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        input_data = {
            feature: data[feature]
            for feature in FEATURES
        }

        input_df = pd.DataFrame([input_data])

        model = load_model()

        prediction = model.predict(input_df)[0]

        prediction_text = str(prediction)

        return jsonify({
            "prediction": prediction_text
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
