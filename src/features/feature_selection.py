"""
Sélection de gènes (Chi-2 et Random Forest).

Chaque fonction renvoie un sélecteur non entraîné. Il est placé dans le
Pipeline (voir train.py)
"""

import numpy as np

from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.ensemble import RandomForestClassifier


def make_chi2(k=300):
    """
    Garde les k gènes les mieux notés par le test du Chi-2.
    Exige des valeurs >= 0 (vrai après log2(x+1)).
    """
    return SelectKBest(score_func=chi2, k=k)

def make_rf_selector(k=300, n_estimators=200, random_state=0):
    """
    Garde les k gènes les plus importants selon un Random Forest.
    Plus lente que make_chi2, car il entraîne des arbres sur tous les gènes.
    """
    return SelectFromModel(
        RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1 # utilise tous les coeurs du processeur
        ),
        max_features=k,
        threshold=-np.inf,
    )
