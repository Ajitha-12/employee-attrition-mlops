
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mlops_employee_attrition_dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = ARTIFACT_DIR / "employee_attrition_pipeline.joblib"
REPORT_PATH = OUTPUT_DIR / "pipeline_report.txt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTION_PATH = OUTPUT_DIR / "sample_prediction.csv"


def load_data():
    print("\nSTEP 1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)
    return df


def prepare_data(df):
    print("\nSTEP 2: Preparing data...")
    y = df["attrition"]
    X = df.drop("attrition", axis=1)
    X = X.drop(["employee_id"], axis=1)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    print("Number of input features:", X.shape[1])
    print("Numerical columns:", list(num_cols))
    print("Categorical columns:", list(cat_cols))
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y, num_cols, cat_cols


def split_data(X, y):
    print("\nSTEP 3: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training records:", X_train.shape[0])
    print("Testing records :", X_test.shape[0])
    return X_train, X_test, y_train, y_test


def build_pipeline(num_cols, cat_cols):
    print("\nSTEP 4: Building ML pipeline...")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    classifier = LogisticRegression(
        C=0.5, max_iter=2000, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("Pipeline created successfully")
    return pipeline


def train_model(pipeline, X_train, y_train):
    print("\nSTEP 5: Training model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed")
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    print("\nSTEP 6: Evaluating model...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }

    print("\nMODEL PERFORMANCE")
    print("-------------------------")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    return metrics


def save_model(pipeline):
    print("\nSTEP 7: Saving trained pipeline...")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("Pipeline saved:", MODEL_PATH)


def test_saved_model(X_test):
    print("\nSTEP 8: Loading saved pipeline...")
    loaded_pipeline = joblib.load(MODEL_PATH)

    sample = X_test.iloc[[0]].copy()
    prediction = loaded_pipeline.predict(sample)[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    print("\nSample employee:")
    print(sample)
    print("\nPrediction:", prediction)

    result = sample.copy()
    result["Prediction"] = prediction

    OUTPUT_DIR.mkdir(exist_ok=True)
    result.to_csv(PREDICTION_PATH, index=False)
    print("Prediction saved:", PREDICTION_PATH)

    return prediction


def save_report(metrics, prediction):
    print("\nSTEP 9: Saving execution results...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(REPORT_PATH, "w") as f:
        f.write("MODULE 2 - PRACTICE 2\n")
        f.write("END-TO-END ML PIPELINE IMPLEMENTATION\n")
        f.write("=======================================\n\n")
        f.write("Dataset: mlops_employee_attrition_dataset(1).csv\n")
        f.write("Target: attrition\n")
        f.write("Model: Logistic Regression\n")
        f.write("Pipeline: Preprocessing + Training + Prediction\n\n")
        f.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision : {metrics['precision']:.4f}\n")
        f.write(f"Recall    : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score  : {metrics['f1_score']:.4f}\n")
        f.write(f"\nSample Prediction: {prediction}\n")
        f.write("\nPipeline executed successfully.\n")

    print("Metrics saved:", METRICS_PATH)
    print("Report saved :", REPORT_PATH)


def main():
    print("\n==========================================")
    print(" EMPLOYEE ATTRITION END-TO-END ML PIPELINE")
    print("==========================================")

    df = load_data()
    X, y, num_cols, cat_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline = train_model(pipeline, X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline)
    prediction = test_saved_model(X_test)
    save_report(metrics, prediction)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
