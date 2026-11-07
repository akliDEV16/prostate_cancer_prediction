"""
Évaluation des pipelines par validation croisée stratifiée.

Tous les modèles passent par le même protocole (mêmes plis, même graine),
ce qui rend la comparaison honnête.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

from src.models.train import CLASSIFIERS, build_pipeline


def evaluate(pipe, X, y, n_splits=5, seed=0):
    """Renvoie ({métrique: (moyenne, écart-type)}, matrice de confusion).

    À chaque pli, le pipeline est cloné, 
    entraîné sur 4/5 des patients, puis noté sur le 1/5 restant qu'il n'a jamais vu.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    y_pred_all = np.empty(len(y), dtype=y.dtype)

    for train_idx, test_idx in cv.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_true = y.iloc[test_idx]
        y_pred = model.predict(X.iloc[test_idx])
        y_pred_all[test_idx] = y_pred

        scores["accuracy"].append(accuracy_score(y_true, y_pred))
        scores["precision"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        scores["recall"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
        scores["f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))

    summary = {m: (np.mean(v), np.std(v)) for m, v in scores.items()}
    return summary, confusion_matrix(y, y_pred_all)


def evaluate_all(X, y, n_splits=5, **pipeline_kwargs):
    """
    Évalue les 4 classifieurs. 
    Renvoie (tableau, {modèle: matrice de confusion}).
    """
    rows, matrices = [], {}
    for name in CLASSIFIERS:
        print(f"Évaluation : {name} ...")
        summary, cm = evaluate(build_pipeline(name, **pipeline_kwargs), X, y, n_splits)
        rows.append({"modele": name,
                     **{m: f"{mean:.3f} ± {std:.3f}" for m, (mean, std) in summary.items()}})
        matrices[name] = cm
    return pd.DataFrame(rows).set_index("modele"), matrices