"""
Évaluation des modèles de prédiction du cancer de la prostate.

Modèles évalués :
1. SVM
2. Naive Bayes
3. KNN
4. Random Forest

Métriques :
- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- Matrice de confusion
"""

import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CHEMINS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT_DIR / "data" / "final" / "test"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CHARGEMENT DES DONNEES
# ==========================================

def load_test_data():

    print("Chargement des données de test...")

    X_test = pd.read_csv(TEST_DIR / "X_test.csv")
    y_test = pd.read_csv(TEST_DIR / "y_test.csv").iloc[:, 0]

    print("Dimensions X_test :", X_test.shape)
    print("Dimensions y_test :", y_test.shape)

    return X_test, y_test


# ==========================================
# EVALUATION D'UN MODELE
# ==========================================

def evaluate_model(name, model, X_test, y_test):

    print("\n======================================")
    print("ÉVALUATION :", name)
    print("======================================")

    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Affichage des résultats
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision Macro   : {precision:.4f}")
    print(f"Recall Macro      : {recall:.4f}")
    print(f"F1-score Macro    : {f1:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")

    print("\nRapport de classification :")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    print("\nMatrice de confusion :")
    print(cm)

    # Sauvegarde des prédictions
    predictions = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    predictions.to_csv(
        RESULT_DIR / f"{name}_predictions.csv",
        index=False
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Balanced Accuracy": balanced_acc
    }, cm


# ==========================================
# EVALUATION DE TOUS LES MODELES
# ==========================================

def evaluate_all_models():

    X_test, y_test = load_test_data()

    model_names = [
        "SVM",
        "Naive_Bayes",
        "KNN",
        "Random_Forest"
    ]

    results = []

    for name in model_names:

        model_path = MODEL_DIR / f"{name}.pkl"

        if not model_path.exists():
            print(f"\nModèle introuvable : {model_path}")
            continue

        print(f"\nChargement du modèle : {name}")

        model = joblib.load(model_path)

        metrics, cm = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(metrics)

    # Tableau comparatif
    if results:

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="F1-score",
            ascending=False
        )

        print("\n======================================")
        print("COMPARAISON DES MODELES")
        print("======================================")

        print(results_df.to_string(index=False))

        results_df.to_csv(
            RESULT_DIR / "comparison_results.csv",
            index=False
        )

        print("\nRésultats sauvegardés dans :", RESULT_DIR)

    else:
        print("\nAucun modèle disponible pour l'évaluation.")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    evaluate_all_models()