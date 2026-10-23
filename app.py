import logging
import time
import uuid
from flask import Flask, request, jsonify, g
from pathlib import Path
import pandas as pd
import joblib

# Log messages to the console so Render can capture them
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "attrition_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    logger.info("Employee attrition model loaded successfully")
except Exception:
    logger.exception("Failed to load employee attrition model")
    raise

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


@app.before_request
def log_request_start():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()
    logger.info(
        "Request started id=%s method=%s path=%s",
        g.request_id, request.method, request.path
    )


@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.time())
    duration_ms = round((time.time() - start_time) * 1000, 2)

    if response.status_code >= 400:
        logger.warning(
            "Request completed id=%s status=%s duration_ms=%s",
            request_id, response.status_code, duration_ms
        )
    else:
        logger.info(
            "Request completed id=%s status=%s duration_ms=%s",
            request_id, response.status_code, duration_ms
        )

    response.headers["X-Request-ID"] = request_id
    return response


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Employee Attrition Prediction API",
        "health": "/health",
        "prediction": "/predict"
    })


@app.route("/health", methods=["GET"])
def health():
    logger.info("Health check successful")
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "Invalid JSON request id=%s",
            getattr(g, "request_id", "unknown")
        )
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        logger.warning(
            "Missing prediction features id=%s missing=%s",
            getattr(g, "request_id", "unknown"), missing
        )
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

        label = "Attrition" if prediction == 1 else "No Attrition"
        logger.info(
            "Prediction successful id=%s label=%s",
            getattr(g, "request_id", "unknown"), label
        )

        return jsonify({
            "attrition_prediction": prediction,
            "prediction_label": label
        }), 200

    except (ValueError, TypeError) as error:
        logger.warning(
            "Invalid prediction input id=%s error_type=%s",
            getattr(g, "request_id", "unknown"),
            type(error).__name__
        )
        return jsonify({"error": str(error)}), 400

    except Exception:
        logger.exception(
            "Prediction failed id=%s",
            getattr(g, "request_id", "unknown")
        )
        return jsonify({"error": "Internal prediction error"}), 500


if __name__ == "__main__":
    logger.info("Starting Employee Attrition Prediction API")
    app.run(host="0.0.0.0", port=5000)
