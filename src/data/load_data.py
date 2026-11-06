"""
Module : load_data.py

Objectif :
    Charger les données génétiques et cliniques TCGA-PRAD
    et les enregistrer dans le dossier data/processed/.

Sources :
    - prad_tcga_genes.zip
    - prad_tcga_clinical_data.xlsx

Résultats :
    - gene_expression.csv
    - clinical_data.csv

Remarque :
    Aucune opération de nettoyage ou de preprocessing
    n'est effectuée dans ce module.
"""

import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossier des données originales
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Dossier de destination
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Fichiers sources
GENES_ZIP = RAW_DIR / "prad_tcga_genes.zip"

CLINICAL_FILE = RAW_DIR / "prad_tcga_clinical_data.xlsx"


# ============================================================
# 2. CHARGEMENT DES DONNÉES GÉNÉTIQUES
# ============================================================

def load_gene_expression():
    """
    Charge les données génétiques depuis le fichier ZIP.

    Le fichier CSV contient :
        - Lignes : gènes
        - Colonnes : patients

    Après transposition :
        - Lignes : patients
        - Colonnes : gènes

    Retourne :
        DataFrame des données génétiques.
    """

    print("\n========== DONNÉES GÉNÉTIQUES ==========")

    if not GENES_ZIP.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {GENES_ZIP}"
        )

    with zipfile.ZipFile(GENES_ZIP, "r") as zip_file:

        # Rechercher les fichiers CSV dans le ZIP
        csv_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "Aucun fichier CSV trouvé dans le ZIP."
            )

        if len(csv_files) > 1:
            raise ValueError(
                f"Plusieurs fichiers CSV trouvés : {csv_files}. "
                "Précisez lequel utiliser."
            )

        csv_file = csv_files[0]

        print("Fichier sélectionné :", csv_file)

        # Lire le CSV directement depuis le ZIP
        with zip_file.open(csv_file) as file:
            gene_df = pd.read_csv(file)

    if gene_df.empty:
        raise ValueError("Le fichier génétique est vide.")

    print("Dimensions originales :", gene_df.shape)

    # Utiliser la première colonne comme identifiant des gènes
    gene_df = gene_df.set_index(gene_df.columns[0])

    # Transposer la matrice
    gene_df = gene_df.transpose()

    gene_df.index.name = "SAMPLE_ID"

    print("Dimensions après transposition :", gene_df.shape)

    return gene_df


# ============================================================
# 3. CHARGEMENT DES DONNÉES CLINIQUES
# ============================================================

def load_clinical_data():
    """
    Charge les données cliniques depuis le fichier Excel.

    Retourne :
        DataFrame des données cliniques.
    """

    print("\n========== DONNÉES CLINIQUES ==========")

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CLINICAL_FILE}"
        )

    clinical_df = pd.read_excel(
        CLINICAL_FILE,
        engine="openpyxl"
    )

    if clinical_df.empty:
        raise ValueError("Le fichier clinique est vide.")

    print("Dimensions cliniques :", clinical_df.shape)

    return clinical_df


# ============================================================
# 4. ENREGISTREMENT DANS DATA/PROCESSED
# ============================================================

def save_processed_data(gene_df, clinical_df):
    """
    Enregistre les deux DataFrames dans data/processed/.

    Aucun nettoyage ni modification des valeurs.
    """

    print("\n========== ENREGISTREMENT DES DONNÉES ==========")

    # Créer le dossier s'il n'existe pas
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Chemins des fichiers de destination
    gene_output = PROCESSED_DIR / "gene_expression.csv"

    clinical_output = PROCESSED_DIR / "clinical_data.csv"

    # Enregistrer les données génétiques
    gene_df.to_csv(gene_output, index=True)

    # Enregistrer les données cliniques
    clinical_df.to_csv(clinical_output, index=False)

    print("Données génétiques enregistrées :", gene_output)

    print("Données cliniques enregistrées :", clinical_output)


# ============================================================
# 5. FONCTION PRINCIPALE
# ============================================================

def load_all_data():
    """
    Charge les deux sources et les enregistre
    dans le dossier data/processed/.
    """

    print("\n============================================")
    print("       CHARGEMENT GLOBAL TCGA-PRAD")
    print("============================================")

    # Charger les données génétiques
    gene_df = load_gene_expression()

    # Charger les données cliniques
    clinical_df = load_clinical_data()

    # Enregistrer les deux datasets
    save_processed_data(
        gene_df,
        clinical_df
    )

    print("\n============================================")
    print("       CHARGEMENT TERMINÉ AVEC SUCCÈS")
    print("============================================")

    print("\nRésumé :")
    print("Échantillons génétiques :", gene_df.shape[0])
    print("Nombre de gènes :", gene_df.shape[1])
    print("Lignes cliniques :", clinical_df.shape[0])

    return gene_df, clinical_df


# ============================================================
# 6. EXÉCUTION DU MODULE
# ============================================================

if __name__ == "__main__":

    load_all_data()