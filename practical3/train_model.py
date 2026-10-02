import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "employee_attrition.csv"
MODEL_FILE = "employee_attrition_model.pkl"
METRICS_FILE = "metrics.json"


def train_model():
    print("Loading employee attrition dataset...")
    data = pd.read_csv(DATA_FILE)

    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    target = "attrition"
    features = [
        "age",
        "experience_years",
        "salary",
        "department",
        "education",
        "job_satisfaction",
        "work_life_balance",
        "training_hours",
    ]

    X = data[features]
    y = data[target]

    categorical_features = ["department", "education"]
    numeric_features = [
        "age",
        "experience_years",
        "salary",
        "job_satisfaction",
        "work_life_balance",
        "training_hours",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )

    print("Training employee attrition model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, MODEL_FILE)
    print("\nModel saved as", MODEL_FILE)

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "total_records": len(data),
    }

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as", METRICS_FILE)
    return accuracy


if __name__ == "__main__":
    train_model()
