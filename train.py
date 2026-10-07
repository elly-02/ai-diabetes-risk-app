"""Train the diabetes risk models and save the one used by the app.

Follows the preprocessing in the Kaggle notebook (Diabetes_prediction.ipynb):
zeros treated as missing and replaced with the column mean, MinMax scaling,
then the four algorithms from the notebook. KNN (the model the notebook
saves) is written to Diabetesmodel.pkl with its scaler so the app can pass
raw values straight in. The test metrics of all four models are written to
model_metrics.json for the app's comparison tab.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes-risk-assessment-scikitlearn-2gd-v1" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "Diabetesmodel.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DEPLOYED_MODEL = "KNN"

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
}


def main():
    dataset = pd.read_csv(DATA_PATH)
    dataset[ZERO_AS_MISSING] = dataset[ZERO_AS_MISSING].replace(0, np.nan)
    dataset[ZERO_AS_MISSING] = dataset[ZERO_AS_MISSING].fillna(dataset[ZERO_AS_MISSING].mean())

    X = dataset[FEATURES].values
    Y = dataset["Outcome"].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y
    )

    models = {}
    metrics = {}
    for name, classifier in CLASSIFIERS.items():
        model = make_pipeline(MinMaxScaler(feature_range=(0, 1)), classifier)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        models[name] = model
        metrics[name] = {
            "accuracy": accuracy_score(Y_test, Y_pred),
            "precision": precision_score(Y_test, Y_pred),
            "recall": recall_score(Y_test, Y_pred),
            "f1": f1_score(Y_test, Y_pred),
            "confusion_matrix": confusion_matrix(Y_test, Y_pred).tolist(),
        }
        print(
            f"{name:<20} accuracy {metrics[name]['accuracy'] * 100:5.2f} %   "
            f"precision {metrics[name]['precision']:.3f}   "
            f"recall {metrics[name]['recall']:.3f}   f1 {metrics[name]['f1']:.3f}"
        )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models[DEPLOYED_MODEL], f)
    print(f"Saved {DEPLOYED_MODEL} model to {MODEL_PATH.name}")

    summary = {
        "features": FEATURES,
        "deployed_model": DEPLOYED_MODEL,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
