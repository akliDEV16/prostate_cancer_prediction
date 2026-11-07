"""
Rééquilibrage des classes (SMOTE-ENN).

L'objet renvoyé est placé dans le Pipeline. 
Il n'agit qu'à l'entraînement. 
"""
from imblearn.combine import SMOTEENN


def make_smote_enn(random_state=0):
    """
    SMOTE crée des patients synthétiques de la classe minoritaire,
    puis ENN supprime les patients dont les voisins sont d'une autre classe.
    """
    return SMOTEENN(random_state=random_state)