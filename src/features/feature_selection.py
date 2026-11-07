"""
Sélection des gènes avec ANOVA et Random Forest.

Étapes :
1. Charger les données d'entraînement.
2. Remplacer les valeurs manquantes.
3. Supprimer les gènes constants.
4. Sélectionner les meilleurs gènes avec ANOVA.
5. Sélectionner les meilleurs gènes avec Random Forest.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    f_classif,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.features.imputation import make_imputer


# ==========================================
# SELECTION ANOVA
# ==========================================

def make_anova(k=300):
    """
    Sélectionne les k gènes les plus pertinents avec ANOVA.

    - Accepte les valeurs négatives.
    - Supprime les gènes constants avant ANOVA.
    """

    return Pipeline([
        ("variance", VarianceThreshold(threshold=0)),
        ("anova", SelectKBest(score_func=f_classif, k=k))
    ])


# ==========================================
# SELECTION RANDOM FOREST
# ==========================================

def make_rf_selector(k=300, n_estimators=200, random_state=42):
    """
    Sélectionne les gènes les plus importants avec Random Forest.
    """

    return SelectFromModel(
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        ),
        max_features=k,
        threshold=-np.inf
    )


# ==========================================
# EXECUTION ET TEST
# ==========================================

if __name__ == "__main__":

    # Chemins du projet
    ROOT_DIR = Path(__file__).resolve().parents[2]
    TRAIN_DIR = ROOT_DIR / "data" / "final" / "train"

    # Chargement des données
    print("Chargement des données...")

    X = pd.read_csv(TRAIN_DIR / "X_train.csv")
    y = pd.read_csv(TRAIN_DIR / "y_train.csv").iloc[:, 0]

    print("Dimensions initiales :", X.shape)
    print("Valeurs manquantes :", X.isnull().sum().sum())
    print("Valeurs négatives :", (X < 0).sum().sum())

    # ======================================
    # SELECTION ANOVA
    # ======================================

    print("\nSélection ANOVA...")

    anova_pipeline = Pipeline([
        ("imputer", make_imputer()),
        ("selector", make_anova(k=300))
    ])

    X_anova = anova_pipeline.fit_transform(X, y)

    print("Après ANOVA :", X_anova.shape)

    # ======================================
    # SELECTION RANDOM FOREST
    # ======================================

    print("\nSélection Random Forest...")

    rf_pipeline = Pipeline([
        ("imputer", make_imputer()),
        ("selector", make_rf_selector(k=300))
    ])

    X_rf = rf_pipeline.fit_transform(X, y)

    print("Après Random Forest :", X_rf.shape)

    print("\nSélection terminée.")