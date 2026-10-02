# Employee Attrition MLOps Practicals

Dataset: employee_attrition.csv
Records: 500
Columns: 10
Target: attrition (0 = stayed, 1 = left)

## Practical 1
Basic GitHub Actions CI with Python unit tests.

## Practical 2
ML CI: load the real dataset, preprocess categorical/numeric features, train GradientBoostingClassifier, evaluate, save model and metrics, then run tests.

## Practical 3
ML quality gate with minimum accuracy 0.85.

## Airflow
Retry DAG that intentionally fails once during preprocessing and succeeds on the next attempt.
