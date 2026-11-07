"""
Mise à l'échelle des gènes.

Le SVM et le KNN reposent sur des distances : un gène aux grandes valeurs
pèserait plus que les autres. Le scaler est placé dans le Pipeline, donc
ses minimums et maximums sont calculés sur le train de chaque pli seulement.
"""
from sklearn.preprocessing import MinMaxScaler


def make_scaler():
    """Ramène chaque gène entre 0 et 1"""
    return MinMaxScaler()