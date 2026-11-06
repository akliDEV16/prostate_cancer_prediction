"""
Module : load_data.py

Objectif :
    Charger les données d'expression génétique et les données
    cliniques du projet de prédiction du score de Gleason.

Données utilisées :
    - prad_tcga_genes.zip
    - prad_tcga_clinical_data.xlsx
"""

import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

# Chemin racine du projet
# Le fichier load_data.py se trouve dans src/data/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dossier contenant les données brutes
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Fichier ZIP contenant les données génétiques
GENES_ZIP = RAW_DIR / "prad_tcga_genes.zip"

# Fichier Excel contenant les données cliniques
CLINICAL_FILE = RAW_DIR / "prad_tcga_clinical_data.xlsx"


# ============================================================
# 2. EXTRACTION ET CHARGEMENT DES DONNÉES GÉNÉTIQUES
# ============================================================

def load_gene_expression():
    """
    Charge les données d'expression génétique depuis le fichier ZIP.

    Le fichier CSV contient normalement :
        - Les gènes en lignes.
        - Les patients en colonnes.

    On transpose le tableau pour obtenir :
        - Les patients en lignes.
        - Les gènes en colonnes.

    Returns:
        pandas.DataFrame : données génétiques.
    """

    print("\n========== CHARGEMENT DES DONNÉES GÉNÉTIQUES ==========")

    # Vérifier que le fichier ZIP existe
    if not GENES_ZIP.exists():
        raise FileNotFoundError(
            f"Fichier ZIP introuvable : {GENES_ZIP}"
        )

    # Ouvrir le fichier ZIP sans extraire inutilement son contenu
    with zipfile.ZipFile(GENES_ZIP, "r") as zip_file:

        # Récupérer les fichiers CSV présents dans le ZIP
        csv_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
        ]

        # Vérifier la présence d'un CSV
        if not csv_files:
            raise FileNotFoundError(
                "Aucun fichier CSV trouvé dans le fichier ZIP."
            )

        # Afficher les fichiers CSV disponibles
        print("Fichiers CSV trouvés :")
        for file in csv_files:
            print(" -", file)

        # Pour cette première version, on utilise le premier CSV
        csv_file = csv_files[0]

        print("\nFichier sélectionné :", csv_file)

        # Lire directement le CSV depuis le ZIP
        with zip_file.open(csv_file) as file:
            gene_df = pd.read_csv(file)

    # Vérifier que le tableau n'est pas vide
    if gene_df.empty:
        raise ValueError(
            "Le fichier d'expression génétique est vide."
        )

    # Afficher les dimensions avant la transposition
    print("\nDimensions avant transposition :", gene_df.shape)

    # Vérifier la première colonne
    print("Premières colonnes :", gene_df.columns[:5].tolist())

    # La première colonne contient normalement les identifiants des gènes
    gene_df = gene_df.set_index(gene_df.columns[0])

    # Transposer :
    # Avant : lignes = gènes, colonnes = patients
    # Après : lignes = patients, colonnes = gènes
    gene_df = gene_df.T

    # Convertir les valeurs en nombres
    gene_df = gene_df.apply(pd.to_numeric, errors="coerce")

    # Afficher les dimensions finales
    print("\nDimensions après transposition :", gene_df.shape)

    print("\nExemple des données génétiques :")
    print(gene_df.iloc[:5, :5])

    print("\nChargement des données génétiques terminé.")

    return gene_df


# ============================================================
# 3. CHARGEMENT DES DONNÉES CLINIQUES
# ============================================================

def load_clinical_data():
    """
    Charge les données cliniques depuis le fichier Excel.

    Returns:
        pandas.DataFrame : données cliniques.
    """

    print("\n========== CHARGEMENT DES DONNÉES CLINIQUES ==========")

    # Vérifier que le fichier Excel existe
    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {CLINICAL_FILE}"
        )

    # Lire le fichier Excel
    clinical_df = pd.read_excel(CLINICAL_FILE, engine="openpyxl")

    # Vérifier que le tableau n'est pas vide
    if clinical_df.empty:
        raise ValueError(
            "Le fichier clinique est vide."
        )

    # Afficher les dimensions
    print("\nDimensions des données cliniques :", clinical_df.shape)

    # Afficher les noms des colonnes
    print("\nColonnes disponibles :")
    for column in clinical_df.columns:
        print(" -", column)

    # Afficher les premières lignes
    print("\nAperçu des données cliniques :")
    print(clinical_df.head())

    print("\nChargement des données cliniques terminé.")

    return clinical_df


# ============================================================
# 4. FONCTION PRINCIPALE DE CHARGEMENT
# ============================================================

def load_all_data():
    """
    Charge les deux sources de données du projet.

    Returns:
        gene_df     : données d'expression génétique.
        clinical_df : données cliniques.
    """

    print("\n==============================================")
    print("       CHARGEMENT GLOBAL DES DONNÉES")
    print("==============================================")

    # Charger les données génétiques
    gene_df = load_gene_expression()

    # Charger les données cliniques
    clinical_df = load_clinical_data()

    print("\n==============================================")
    print("       CHARGEMENT TERMINÉ")
    print("==============================================")

    return gene_df, clinical_df


# ============================================================
# 5. TEST DU MODULE
# ============================================================

if __name__ == "__main__":

    # Charger les deux jeux de données
    gene_df, clinical_df = load_all_data()

    # Afficher un résumé final
    print("\nRésumé du chargement :")

    print("Nombre de patients dans les données génétiques :",
          gene_df.shape[0])

    print("Nombre de gènes :", gene_df.shape[1])

    print("Nombre de lignes cliniques :", clinical_df.shape[0])

    print("\nModule load_data.py exécuté avec succès.")