"""
Module : inspect_alignment.py


Objectif :
Vérifier la correspondance entre les données génétiques
et les données cliniques avant leur fusion.
"""

from pathlib import Path
import pandas as pd


# =====================================================
# 1. CONFIGURATION DES CHEMINS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GENE_FILE = PROCESSED_DIR / "gene_expression.csv"
CLINICAL_FILE = PROCESSED_DIR / "clinical_data.csv"


# =====================================================
# 2. CHARGEMENT DES DONNÉES
# =====================================================

def load_data():
    """
    Charge les données génétiques et cliniques.

    Returns:
        gene_df : données génétiques
        clinical_df : données cliniques
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

    if "PATIENT_ID" not in clinical_df.columns:
        raise ValueError(
            "La colonne PATIENT_ID est absente du fichier clinique."
        )

    return gene_df, clinical_df


# =====================================================
# 3. CORRESPONDANCE DES IDENTIFIANTS
# =====================================================

def check_patient_ids(gene_df, clinical_df):
    """
    Compare les identifiants des deux datasets.
    """

    print("\n========== IDENTIFIANTS DES PATIENTS ==========")

    gene_ids = set(gene_df.index.dropna().astype(str))

    clinical_ids = set(
        clinical_df["PATIENT_ID"].dropna().astype(str)
    )

    # Patients présents dans les deux fichiers
    common_ids = gene_ids.intersection(clinical_ids)

    # Patients uniquement dans le fichier génétique
    gene_only = gene_ids - clinical_ids

    # Patients uniquement dans le fichier clinique
    clinical_only = clinical_ids - gene_ids

    print("Patients génétiques :", len(gene_ids))

    print("Patients cliniques :", len(clinical_ids))

    print("Patients communs :", len(common_ids))

    print("Uniquement génétiques :", len(gene_only))

    print("Uniquement cliniques :", len(clinical_only))

    print("\nExemples uniquement cliniques :")
    print(sorted(clinical_only)[:10])

    print("\nExemples uniquement génétiques :")
    print(sorted(gene_only)[:10])

    return common_ids


# =====================================================
# 4. RECHERCHE DES DOUBLONS
# =====================================================

def check_duplicates(gene_df, clinical_df):

    print("\n========== RECHERCHE DES DOUBLONS ==========")

    gene_duplicates = gene_df.index.duplicated().sum()

    clinical_duplicates = clinical_df[
        "PATIENT_ID"
    ].duplicated().sum()

    print("Doublons génétiques :", gene_duplicates)

    print("Doublons cliniques :", clinical_duplicates)


# =====================================================
# 5. VALEURS MANQUANTES
# =====================================================

def check_missing_values(clinical_df):

    print("\n========== VALEURS MANQUANTES ==========")

    missing = clinical_df.isna().sum()

    total_missing = missing.sum()

    print("Total des valeurs manquantes :", total_missing)

    print("\nColonnes contenant des valeurs manquantes :")

    missing_columns = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing_columns.empty:
        print("Aucune valeur manquante.")

    else:
        print(missing_columns)


# =====================================================
# 6. DISTRIBUTION DU SCORE DE GLEASON
# =====================================================

def check_gleason_distribution(clinical_df):

    print("\n========== DISTRIBUTION DU GLEASON SCORE ==========")

    columns = [
        "GLEASON_SCORE",
        "GLEASON_PATTERN_PRIMARY",
        "GLEASON_PATTERN_SECONDARY"
    ]

    for column in columns:

        if column not in clinical_df.columns:
            print(f"\nColonne absente : {column}")
            continue

        print(f"\nDistribution de {column} :")

        print(
            clinical_df[column].value_counts(
                dropna=False
            ).sort_index()
        )


# =====================================================
# 7. FONCTION PRINCIPALE
# =====================================================

def main():

    print("\n============================================")
    print("       ANALYSE DES DONNÉES TCGA-PRAD")
    print("============================================")

    # Chargement
    gene_df, clinical_df = load_data()

    # Correspondance des patients
    common_ids = check_patient_ids(
        gene_df,
        clinical_df
    )

    # Doublons
    check_duplicates(
        gene_df,
        clinical_df
    )

    # Valeurs manquantes
    check_missing_values(clinical_df)

    # Scores de Gleason
    check_gleason_distribution(clinical_df)

    print("\n========== RÉSUMÉ FINAL ==========")

    print("Patients génétiques :", len(gene_df))

    print("Patients cliniques :", len(clinical_df))

    print("Patients communs :", len(common_ids))

    print("\nInspection terminée avec succès.")


# =====================================================
# 8. EXÉCUTION
# =====================================================

if __name__ == "__main__":
    main()