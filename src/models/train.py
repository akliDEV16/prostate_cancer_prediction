"""
Entraînement de 4 modèles de prédiction du cancer de la prostate.

Pipeline :
1. Imputation des valeurs manquantes.
2. Suppression des caractéristiques constantes.
3. Sélection des gènes avec ANOVA.
4. Normalisation des données.
5. Classification avec 4 algorithmes.
6. Sauvegarde des modèles.
"""

import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from src.features.log_transform import log2p, keep_genes


# ==========================================
# CHEMINS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT_DIR / "data" / "final" / "train"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CHARGEMENT DES DONNEES
# ==========================================

def load_training_data():

    print("Chargement des données...")

    X_train = pd.read_csv(TRAIN_DIR / "X_train.csv")
    y_train = pd.read_csv(TRAIN_DIR / "y_train.csv").iloc[:, 0]

    print("Dimensions X_train :", X_train.shape)
    print("Dimensions y_train :", y_train.shape)

    print("\nRépartition des classes :")
    print(y_train.value_counts().sort_index())

    return X_train, y_train


# ==========================================
# CONSTRUCTION DES MODELES
# ==========================================

def build_models():

    models = {

        # 1. SVM
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("anova", SelectKBest(score_func=f_classif, k=10000)),
            ("scaling", MinMaxScaler()),
            ("classifier", SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        # 2. NAIVE BAYES
        "Naive_Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("anova", SelectKBest(score_func=f_classif, k=10000)),
            ("scaling", MinMaxScaler()),
            ("classifier", GaussianNB())
        ]),

        # 3. KNN
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("anova", SelectKBest(score_func=f_classif, k=10000)),
            ("scaling", MinMaxScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            ))
        ]),

        # 4. RANDOM FOREST
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("anova", SelectKBest(score_func=f_classif, k=10000)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

    }

    return models


# ==========================================
# ENTRAINEMENT
# ==========================================

def train_models():

    X_train, y_train = load_training_data()

    models = build_models()

    for name, model in models.items():

        print("\n====================================")
        print("ENTRAINEMENT :", name)
        print("====================================")

        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{name}.pkl"

        joblib.dump(model, model_path)

        print("Entraînement terminé.")
        print("Modèle sauvegardé :", model_path)

    print("\nTous les modèles ont été entraînés.")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    train_models()