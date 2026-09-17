import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-continuous-training")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris-classifier")
RANDOM_STATE = 42

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5],
    "criterion": ["gini", "entropy"],
}


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["species"])
    y = df["species"]
    return X, y


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.", file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=PARAM_GRID,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run() as run:
        print(f"MLflow run: {run.info.run_id}")

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")
        print(f"Best mean CV f1_weighted: {grid.best_score_:.4f}")

        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        test_acc = accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred))

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1_weighted")
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_metric("best_cv_f1_weighted", grid.best_score_)
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.log_metric("test_accuracy", test_acc)

        mlflow.log_artifact(DATA_PATH, artifact_path="data")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(5),
        )

        print(f"Model logged and registered as '{MODEL_NAME}'.")

main()