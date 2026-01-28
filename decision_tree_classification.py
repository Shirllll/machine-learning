#!/usr/bin/env python3
"""Decision tree classification on the buys_computer dataset."""

from __future__ import annotations

import pathlib

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = pathlib.Path(__file__).resolve().parent / "data" / "buys_computer.csv"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please create data/buys_computer.csv."
        )
    return pd.read_csv(DATA_PATH)


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["age", "income", "student", "credit_rating"]
    target_col = "buys_computer"
    encoders: dict[str, LabelEncoder] = {}
    encoded = df.copy()

    for col in feature_cols + [target_col]:
        encoder = LabelEncoder()
        encoded[col] = encoder.fit_transform(encoded[col])
        encoders[col] = encoder

    X = encoded[feature_cols]
    y = encoded[target_col]
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=4)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)


def main() -> None:
    df = load_dataset()
    X, y = encode_features(df)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
