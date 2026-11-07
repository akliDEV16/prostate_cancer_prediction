"""
Évaluation du modèle SVM entraîné.
"""

import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Chemins
ROOT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT_DIR / "data" / "final" / "test"
MODEL_PATH = ROOT_DIR / "models" / "prostate_cancer_model.pkl"


def evaluate_model():

    print("Chargement du modèle...")

    model = joblib.load(MODEL_PATH)

    print("Chargement des données de test...")

    X_test = pd.read_csv(TEST_DIR / "X_test.csv")
    y_test = pd.read_csv(TEST_DIR / "y_test.csv").iloc[:, 0]

    print("\nPrédiction...")

    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== ACCURACY ==========")
    print(f"Accuracy : {accuracy * 100:.2f}%")

    # Rapport de classification
    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Matrice de confusion
    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()