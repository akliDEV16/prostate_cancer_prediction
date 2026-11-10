import numpy as np

def log2p(A):
    return np.log2(np.clip(A, 0, None) + 1)

def keep_genes(X):
    """Garde uniquement les colonnes de gènes (ENSG...)."""
    return X[[c for c in X.columns if str(c).startswith("ENSG")]]