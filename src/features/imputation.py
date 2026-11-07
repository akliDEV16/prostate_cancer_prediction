"""
Traitement des valeurs manquantes dans les données génétiques.
"""

from sklearn.impute import SimpleImputer


def make_imputer(strategy="median"):
    """
    Crée un imputer pour remplacer les valeurs manquantes.

    strategy :
        - median : remplacer par la médiane
        - mean   : remplacer par la moyenne
    """

    return SimpleImputer(strategy=strategy)