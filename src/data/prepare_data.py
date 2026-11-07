"""
Module : prepare_dataset.py


Objectif :
Préparer les données génétiques et cliniques pour
les algorithmes de Machine Learning.

Opérations :
- Chargement du dataset fusionné.
- Nettoyage des noms de colonnes.
- Suppression des doublons.
- Suppression des identifiants inutiles.
- Suppression des variables responsables du Data Leakage.
- Définition de la variable cible.
- Conversion des caractéristiques en format numérique.
- Vérification des valeurs manquantes.
- Séparation des variables X et y.
- Exportation des datasets préparés.

Remarque :
La division Train/Test, l'imputation et la standardisation
seront réalisées dans des modules séparés.
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "merged_data.csv"

OUTPUT_X = PROCESSED_DIR / "X.csv"
OUTPUT_Y = PROCESSED_DIR / "y.csv"
OUTPUT_IDS = PROCESSED_DIR / "patient_ids.csv"
OUTPUT_FEATURES = PROCESSED_DIR / "feature_names.json"


# ============================================================
# 2. PARAMÈTRES DU PROJET
# ============================================================

TARGET_COLUMN = "GLEASON_SCORE"

ID_COLUMN = "PATIENT_ID"

# Variables directement liées au score de Gleason.
# Elles doivent être exclues des caractéristiques X.

LEAKAGE_COLUMNS = [
    "GLEASON_PATTERN_PRIMARY",
    "GLEASON_PATTERN_SECONDARY",
    "GLEASON_PATTERN_TERTIARY"
]


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

def load_dataset():

    """
    Charge le dataset fusionné contenant les données
    génétiques et cliniques.
    """

    print("\n========== CHARGEMENT DES DONNÉES ==========")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Dimensions initiales :", df.shape)

    print("Nombre de patients :", len(df))

    return df


# ============================================================
# 4. NETTOYAGE DES COLONNES
# ============================================================

def clean_column_names(df):

    """
    Nettoie les noms des colonnes.

    Supprime les espaces inutiles au début et à la fin
    des noms des variables.
    """

    print("\n========== NETTOYAGE DES COLONNES ==========")

    df = df.copy()

    df.columns = df.columns.astype(str).str.strip()

    # Vérifier les noms de colonnes dupliqués.

    duplicated_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicated_columns:

        raise ValueError(
            f"Colonnes dupliquées détectées : {duplicated_columns}"
        )

    print("Nettoyage des noms de colonnes terminé.")

    return df


# ============================================================
# 5. NETTOYAGE DU DATASET
# ============================================================

def clean_dataset(df):

    """
    Nettoie le dataset avant la séparation de X et y.

    - Supprime les lignes dupliquées.
    - Vérifie les identifiants des patients.
    - Supprime les identifiants des caractéristiques.
    - Supprime les variables responsables du Data Leakage.
    - Supprime les patients sans score Gleason.
    """

    print("\n========== NETTOYAGE DU DATASET ==========")

    df = df.copy()

    # --------------------------------------------------------
    # 1. Suppression des lignes dupliquées
    # --------------------------------------------------------

    duplicated_count = df.duplicated().sum()

    print("Lignes dupliquées :", duplicated_count)

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # 2. Vérification des identifiants
    # --------------------------------------------------------

    if ID_COLUMN in df.columns:

        df[ID_COLUMN] = df[ID_COLUMN].astype("string").str.strip()

        if df[ID_COLUMN].isna().any():

            raise ValueError(
                "Des identifiants de patients sont manquants."
            )

        duplicated_ids = df[ID_COLUMN].duplicated().sum()

        if duplicated_ids > 0:

            raise ValueError(
                f"{duplicated_ids} identifiants patients sont dupliqués."
            )

    # --------------------------------------------------------
    # 3. Vérification de la variable cible
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"La variable cible {TARGET_COLUMN} est absente."
        )

    # Conversion du Gleason Score en numérique.

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # Suppression des patients sans score Gleason.

    missing_target = df[TARGET_COLUMN].isna().sum()

    print("Scores Gleason manquants :", missing_target)

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    # Vérifier que les scores sont des nombres entiers.

    if not (df[TARGET_COLUMN] % 1 == 0).all():

        raise ValueError(
            "Certains scores Gleason ne sont pas des entiers."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # 4. Suppression des variables responsables du Data Leakage
    # --------------------------------------------------------

    existing_leakage = [
        col for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_leakage
    )

    print(
        "Variables supprimées pour éviter le Data Leakage :",
        existing_leakage
    )

    # --------------------------------------------------------
    # 5. Vérification finale
    # --------------------------------------------------------

    print("Dimensions après nettoyage :", df.shape)

    print("\nDistribution du Gleason Score :")

    print(
        df[TARGET_COLUMN].value_counts().sort_index()
    )

    return df


# ============================================================
# 6. SÉPARATION DES VARIABLES X ET Y
# ============================================================

def separate_features_target(df):

    """
    Sépare les variables explicatives X
    et la variable cible y.

    X : caractéristiques génétiques et cliniques.
    y : Gleason Score.

    Les identifiants des patients sont conservés séparément
    pour assurer la traçabilité des observations.
    """

    print("\n========== SÉPARATION X ET Y ==========")

    # --------------------------------------------------------
    # 1. Conserver les identifiants séparément
    # --------------------------------------------------------

    if ID_COLUMN in df.columns:

        patient_ids = df[ID_COLUMN].copy()

    else:

        patient_ids = pd.Series(
            df.index,
            name=ID_COLUMN
        )

    # --------------------------------------------------------
    # 2. Définir la variable cible
    # --------------------------------------------------------

    y = df[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # 3. Définir les caractéristiques
    # --------------------------------------------------------

    columns_to_drop = [
        TARGET_COLUMN,
        ID_COLUMN
    ]

    X = df.drop(
        columns=[
            col for col in columns_to_drop
            if col in df.columns
        ]
    ).copy()

    # --------------------------------------------------------
    # 4. Conversion numérique des caractéristiques
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # 5. Suppression des colonnes entièrement manquantes
    # --------------------------------------------------------

    all_missing_columns = X.columns[
        X.isna().all()
    ].tolist()

    X = X.drop(
        columns=all_missing_columns
    )

    print(
        "Colonnes entièrement manquantes supprimées :",
        len(all_missing_columns)
    )

    # --------------------------------------------------------
    # 6. Vérification des données
    # --------------------------------------------------------

    print("\nDimensions de X :", X.shape)

    print("Dimensions de y :", y.shape)

    print(
        "Valeurs manquantes dans X :",
        X.isna().sum().sum()
    )

    print("\nDistribution de la cible :")

    print(
        y.value_counts().sort_index()
    )

    return X, y, patient_ids


# ============================================================
# 7. EXPORTATION DES DONNÉES PRÉPARÉES
# ============================================================

def save_dataset(X, y, patient_ids):

    """
    Enregistre X, y et les identifiants des patients.

    Les valeurs manquantes restantes dans X ne sont pas
    remplacées ici. L'imputation sera réalisée dans
    le pipeline de préparation des caractéristiques.
    """

    print("\n========== EXPORTATION DES DONNÉES ==========")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Exporter les caractéristiques X
    # --------------------------------------------------------

    X.to_csv(
        OUTPUT_X,
        index=False
    )

    # --------------------------------------------------------
    # 2. Exporter la cible y
    # --------------------------------------------------------

    y.to_csv(
        OUTPUT_Y,
        index=False,
        header=[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # 3. Exporter les identifiants des patients
    # --------------------------------------------------------

    patient_ids.to_csv(
        OUTPUT_IDS,
        index=False,
        header=[ID_COLUMN]
    )

    # --------------------------------------------------------
    # 4. Exporter les noms des caractéristiques
    # --------------------------------------------------------

    feature_names = X.columns.tolist()

    with open(
        OUTPUT_FEATURES,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_names,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nFichiers enregistrés :")

    print(OUTPUT_X)

    print(OUTPUT_Y)

    print(OUTPUT_IDS)

    print(OUTPUT_FEATURES)

    print("\nExportation terminée avec succès.")


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n============================================")

    print("       PRÉPARATION DU DATASET TCGA-PRAD")

    print("============================================")

    # Étape 1 : Chargement des données

    df = load_dataset()

    # Étape 2 : Nettoyage des noms de colonnes

    df = clean_column_names(df)

    # Étape 3 : Nettoyage du dataset

    df = clean_dataset(df)

    # Étape 4 : Séparation X et y

    X, y, patient_ids = separate_features_target(df)

    # Étape 5 : Exportation

    save_dataset(
        X,
        y,
        patient_ids
    )

    print("\n============================================")

    print("       PRÉPARATION TERMINÉE AVEC SUCCÈS")

    print("============================================")


# ============================================================
# 9. POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":

    main()