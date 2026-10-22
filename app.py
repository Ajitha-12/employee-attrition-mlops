
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "attrition_model.pkl"

model = joblib.load(MODEL_FILE)

FEATURES = [
    "age",
    "experience_years",
    "salary",
    "department",
    "education",
    "job_satisfaction",
    "work_life_balance",
    "training_hours"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Employee Attrition Prediction API",
        "health": "/health",
        "prediction": "/predict"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing
        }), 400

    try:
        employee = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(employee)[0])

        return jsonify({
            "attrition_prediction": prediction,
            "prediction_label": (
                "Attrition" if prediction == 1 else "No Attrition"
            )
        }), 200

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
