"""
Réduction de dimension par LDA (analyse discriminante linéaire).
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def make_lda():
    """
    Projette les patients sur les axes qui séparent le mieux les classes.

    Le nombre d'axes est au plus (nombre de classes - 1) : un seul en cible binaire. 
    Le shrinkage régularise le calcul, utile quand le nombre de
    gènes reste élevé par rapport au nombre de patients.
    """
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")