"""
Module : clean_data.py


Objectif :
Nettoyer les données génétiques et cliniques avant
leur fusion et leur utilisation dans le Machine Learning.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GENE_FILE = PROCESSED_DIR / "gene_expression.csv"
CLINICAL_FILE = PROCESSED_DIR / "clinical_data.csv"

GENE_OUTPUT = PROCESSED_DIR / "gene_expression_clean.csv"
CLINICAL_OUTPUT = PROCESSED_DIR / "clinical_data_clean.csv"


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

def load_data():

    """
    Charge les données génétiques et cliniques.
    """

    if not GENE_FILE.exists():
        raise FileNotFoundError(
            f"Fichier génétique introuvable : {GENE_FILE}"
        )

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier clinique introuvable : {CLINICAL_FILE}"
        )

    gene_df = pd.read_csv(GENE_FILE, index_col=0)

    clinical_df = pd.read_csv(CLINICAL_FILE)

    print("\n========== CHARGEMENT DES DONNÉES ==========")

    print("Données génétiques :", gene_df.shape)

    print("Données cliniques :", clinical_df.shape)

    return gene_df, clinical_df


# ============================================================
# 3. NETTOYAGE DES IDENTIFIANTS
# ============================================================

def clean_patient_ids(gene_df, clinical_df):

    """
    Nettoie les identifiants des patients.

    Supprime les espaces inutiles et les identifiants vides.
    """

    print("\n========== NETTOYAGE DES IDENTIFIANTS ==========")

    # Nettoyage des identifiants génétiques
    gene_df.index = gene_df.index.astype(str).str.strip()

    # Nettoyage des identifiants cliniques
    clinical_df["PATIENT_ID"] = (
        clinical_df["PATIENT_ID"]
        .astype("string")
        .str.strip()
    )

    # Suppression des identifiants vides
    clinical_df["PATIENT_ID"] = clinical_df[
        "PATIENT_ID"
    ].replace("", pd.NA)

    # Suppression des lignes sans identifiant clinique
    missing_ids = clinical_df["PATIENT_ID"].isna().sum()

    clinical_df = clinical_df.dropna(
        subset=["PATIENT_ID"]
    ).copy()

    print("Identifiants cliniques manquants supprimés :", missing_ids)

    print("Nettoyage des identifiants terminé.")

    return gene_df, clinical_df


# ============================================================
# 4. VÉRIFICATION DES DOUBLONS
# ============================================================

def check_duplicates(gene_df, clinical_df):

    """
    Vérifie si des patients apparaissent plusieurs fois.

    Ne supprime pas automatiquement les doublons :
    ils doivent être examinés avant toute suppression.
    """

    print("\n========== VÉRIFICATION DES DOUBLONS ==========")

    gene_duplicates = gene_df.index.duplicated().sum()

    clinical_duplicates = clinical_df[
        "PATIENT_ID"
    ].duplicated().sum()

    print("Doublons génétiques :", gene_duplicates)

    print("Doublons cliniques :", clinical_duplicates)

    if gene_duplicates > 0:
        raise ValueError(
            "Des identifiants génétiques sont dupliqués. "
            "Il faut les examiner avant de continuer."
        )

    if clinical_duplicates > 0:
        raise ValueError(
            "Des PATIENT_ID sont dupliqués dans les données "
            "cliniques. Il faut les examiner avant de continuer."
        )

    print("Aucun doublon détecté.")


# ============================================================
# 5. CONVERSION DES VARIABLES NUMÉRIQUES
# ============================================================

def convert_numeric_columns(clinical_df):

    """
    Convertit les variables cliniques numériques
    dans un format numérique adapté.

    Les valeurs impossibles à convertir deviennent NaN.
    """

    print("\n========== CONVERSION DES VARIABLES ==========")

    numeric_columns = [
        "GLEASON_SCORE",
        "GLEASON_PATTERN_PRIMARY",
        "GLEASON_PATTERN_SECONDARY",
        "GLEASON_PATTERN_TERTIARY",
        "AGE",
        "DAYS_TO_BIRTH",
        "OS_MONTHS",
        "DFS_MONTHS"
    ]

    for column in numeric_columns:

        if column in clinical_df.columns:

            clinical_df[column] = pd.to_numeric(
                clinical_df[column],
                errors="coerce"
            )

            print(f"{column} : conversion terminée.")

    return clinical_df


# ============================================================
# 6. ANALYSE DES VALEURS MANQUANTES
# ============================================================

def check_missing_values(gene_df, clinical_df):

    """
    Calcule le nombre de valeurs manquantes
    dans les deux datasets.
    """

    print("\n========== VALEURS MANQUANTES ==========")

    gene_missing = gene_df.isna().sum().sum()

    clinical_missing = clinical_df.isna().sum().sum()

    print("Valeurs manquantes génétiques :", gene_missing)

    print("Valeurs manquantes cliniques :", clinical_missing)

    print("\nColonnes cliniques avec des valeurs manquantes :")

    missing_columns = clinical_df.isna().sum()

    missing_columns = missing_columns[
        missing_columns > 0
    ].sort_values(ascending=False)

    if missing_columns.empty:

        print("Aucune valeur manquante.")

    else:

        print(missing_columns)

    print("\nAttention : aucune valeur manquante n'est remplacée.")


# ============================================================
# 7. EXPORTATION DES DONNÉES NETTOYÉES
# ============================================================

def save_clean_data(gene_df, clinical_df):

    """
    Enregistre les données nettoyées.

    Les fichiers originaux restent inchangés.
    """

    gene_df.to_csv(GENE_OUTPUT, index=True)

    clinical_df.to_csv(CLINICAL_OUTPUT, index=False)

    print("\n========== EXPORTATION ==========")

    print("Données génétiques :", GENE_OUTPUT)

    print("Données cliniques :", CLINICAL_OUTPUT)


# ============================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n============================================")

    print("       NETTOYAGE DES DONNÉES TCGA-PRAD")

    print("============================================")

    # Étape 1 : Chargement
    gene_df, clinical_df = load_data()

    # Étape 2 : Nettoyage des identifiants
    gene_df, clinical_df = clean_patient_ids(
        gene_df,
        clinical_df
    )

    # Étape 3 : Vérification des doublons
    check_duplicates(gene_df, clinical_df)

    # Étape 4 : Conversion des variables numériques
    clinical_df = convert_numeric_columns(clinical_df)

    # Étape 5 : Analyse des valeurs manquantes
    check_missing_values(gene_df, clinical_df)

    # Étape 6 : Exportation
    save_clean_data(gene_df, clinical_df)

    print("\n============================================")

    print("       NETTOYAGE TERMINÉ AVEC SUCCÈS")

    print("============================================")


if __name__ == "__main__":

    main()