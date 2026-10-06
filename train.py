"""Train the diabetes risk model and save it as Diabetesmodel.pkl.

Follows the preprocessing in the Kaggle notebook (Diabetes_prediction.ipynb):
zeros treated as missing and replaced with the column mean, MinMax scaling,
KNN with 24 neighbours. The scaler is saved together with the model so the
app can pass raw values straight in.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes-risk-assessment-scikitlearn-2gd-v1" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "Diabetesmodel.pkl"

FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main():
    dataset = pd.read_csv(DATA_PATH)
    dataset[ZERO_AS_MISSING] = dataset[ZERO_AS_MISSING].replace(0, np.nan)
    dataset[ZERO_AS_MISSING] = dataset[ZERO_AS_MISSING].fillna(dataset[ZERO_AS_MISSING].mean())

    X = dataset[FEATURES].values
    Y = dataset["Outcome"].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y
    )

    model = make_pipeline(
        MinMaxScaler(feature_range=(0, 1)),
        KNeighborsClassifier(n_neighbors=24, metric="minkowski", p=2),
    )
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print(f"Features: {FEATURES}")
    print(f"Test accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f} %")
    print(classification_report(Y_test, Y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
