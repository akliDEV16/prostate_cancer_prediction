"""
Construction des pipelines : un par classifieur.

Le pipeline: sélection de gènes -> MinMax -> SMOTE-ENN -> LDA -> classifieur.
Il est réajusté à chaque pli de la validation croisée : le pli de test
ne sert jamais à apprendre une étape.
"""
from imblearn.pipeline import Pipeline  # gère les étapes de rééquilibrage
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_selection import make_chi2, make_rf_selector
from src.features.scaling import make_scaler
from src.features.imbalance import make_smote_enn
from src.features.dimensionality_reduction import make_lda

CLASSIFIERS = {
    "naive_bayes": GaussianNB(),
    "svm": SVC(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=0),
}

SELECTORS = {
    "chi2": make_chi2,
    "rf": make_rf_selector,
}


def build_pipeline(name, selector="chi2", k=300):
    """
    Assemble le pipeline complet pour un classifieur.

    name     : "naive_bayes", "svm", "knn" ou "random_forest"
    selector : "chi2" ou "rf" (méthode de sélection des gènes)
    k        : nombre de gènes gardés
    """
    return Pipeline([
        ("select", SELECTORS[selector](k)),
        ("scale", make_scaler()),
        ("smote_enn", make_smote_enn()),
        ("lda", make_lda()),
        ("clf", clone(CLASSIFIERS[name])),  # instance neuve à chaque appel
    ])