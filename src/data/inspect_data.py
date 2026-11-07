"""
Module : inspect_data.py

Objectif :
    Inspecter les données génétiques et cliniques
    avant de commencer leur nettoyage.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GENE_FILE = PROCESSED_DIR / "gene_expression.csv"

CLINICAL_FILE = PROCESSED_DIR / "clinical_data.csv"


# ============================================================
# 2. INSPECTION DES DONNÉES GÉNÉTIQUES
# ============================================================

def inspect_gene_data():

    print("\n========== DONNÉES GÉNÉTIQUES ==========")

    if not GENE_FILE.exists():
        raise FileNotFoundError(GENE_FILE)

    df = pd.read_csv(
        GENE_FILE,
        index_col=0
    )

    print("\nDimensions :", df.shape)

    print("\nPremiers identifiants :")
    print(df.index[:10].tolist())

    print("\nAperçu :")
    print(df.iloc[:5, :5])

    print("\nValeurs manquantes :")
    print("Total :", df.isna().sum().sum())

    print("\nIdentifiants dupliqués :")
    print(df.index.duplicated().sum())

    print("\nTypes de données :")
    print(df.dtypes.value_counts())

    return df


# ============================================================
# 3. INSPECTION DES DONNÉES CLINIQUES
# ============================================================

def inspect_clinical_data():

    print("\n========== DONNÉES CLINIQUES ==========")

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(CLINICAL_FILE)

    df = pd.read_csv(CLINICAL_FILE)

    print("\nDimensions :", df.shape)

    print("\nColonnes disponibles :")
    for column in df.columns:
        print("-", column)

    print("\nPremières lignes :")
    print(df.head())

    print("\nValeurs manquantes par colonne :")
    print(
        df.isna().sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nNombre de lignes dupliquées :")
    print(df.duplicated().sum())

    return df


# ============================================================
# 4. EXÉCUTION PRINCIPALE
# ============================================================

def main():

    print("\n======================================")
    print("     INSPECTION DU DATASET TCGA-PRAD")
    print("======================================")

    gene_df = inspect_gene_data()

    clinical_df = inspect_clinical_data()

    print("\n========== RÉSUMÉ GLOBAL ==========")

    print("Nombre d'échantillons génétiques :", gene_df.shape[0])

    print("Nombre de gènes :", gene_df.shape[1])

    print("Nombre de lignes cliniques :", clinical_df.shape[0])

    print("\nInspection terminée.")


if __name__ == "__main__":
    main()