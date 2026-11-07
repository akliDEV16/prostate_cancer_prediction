"""
Entraînement du modèle de prédiction du cancer de la prostate.

Pipeline :
1. Imputation des valeurs manquantes.
2. Sélection des gènes avec ANOVA.
3. Normalisation des données.
4. Rééquilibrage des classes avec SMOTE-ENN.
5. Réduction de dimension avec LDA.
6. Classification avec SVM.
7. Sauvegarde du modèle.
"""

import pandas as pd
import joblib

from pathlib import Path

from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from src.features.imputation import make_imputer
from src.features.scaling import make_scaler
from src.features.dimensionality_reduction import make_lda
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# ==========================================
# CHEMINS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT_DIR / "data" / "final" / "train"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "prostate_cancer_model.pkl"


# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================

def load_training_data():

    print("Chargement des données d'entraînement...")

    X_train = pd.read_csv(TRAIN_DIR / "X_train.csv")
    y_train = pd.read_csv(TRAIN_DIR / "y_train.csv").iloc[:, 0]

    print("Dimensions de X_train :", X_train.shape)
    print("Dimensions de y_train :", y_train.shape)

    print("\nRépartition des classes :")
    print(y_train.value_counts().sort_index())

    return X_train, y_train


# ==========================================
# CONSTRUCTION DU PIPELINE
# ==========================================

def build_pipeline(y_train):

    # Nombre d'échantillons dans la classe minoritaire
    min_class_count = y_train.value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            "SMOTE nécessite au moins deux échantillons "
            "dans chaque classe."
        )

    k_neighbors = min(5, min_class_count - 1)


    pipeline = Pipeline([

        # 1. Imputation des valeurs manquantes
        ("imputer", make_imputer()),

        # 2. Suppression des gènes constants
        ("variance", VarianceThreshold(threshold=0)),

        # 3. Sélection des 500 meilleurs gènes
        ("anova", SelectKBest(
            score_func=f_classif,
            k=300
        )),

        # 4. Normalisation
        ("scaling", make_scaler()),

        # 5. Rééquilibrage des classes

        # 6. Réduction de dimension avec LDA

        # 7. Classification avec SVM
        ("classifier", SVC(
            kernel="linear",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        ))
    ])

    return pipeline


# ==========================================
# ENTRAÎNEMENT
# ==========================================

def train_model():

    X_train, y_train = load_training_data()

    print("\nConstruction du pipeline...")

    model = build_pipeline(y_train)

    print("\nDébut de l'entraînement...")

    model.fit(X_train, y_train)

    print("\nEntraînement terminé.")

    # Sauvegarde du modèle
    joblib.dump(model, MODEL_PATH)

    print("\nModèle sauvegardé :", MODEL_PATH)

    return model


# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================

if __name__ == "__main__":

    train_model()