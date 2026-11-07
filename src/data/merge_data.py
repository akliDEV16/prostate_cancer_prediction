"""
Module : merge_data.py


Objectif :
Fusionner les données génétiques et cliniques
en utilisant les identifiants des patients.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. CONFIGURATION DES CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

GENE_FILE = PROCESSED_DIR / "gene_expression_clean.csv"
CLINICAL_FILE = PROCESSED_DIR / "clinical_data_clean.csv"

OUTPUT_FILE = PROCESSED_DIR / "merged_data.csv"


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

def load_data():

    """
    Charge les données génétiques et cliniques nettoyées.
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

    # Nettoyer les noms des colonnes
    gene_df.columns = gene_df.columns.astype(str).str.strip()
    clinical_df.columns = clinical_df.columns.astype(str).str.strip()

    print("\n========== CHARGEMENT DES DONNÉES ==========")

    print("Données génétiques :", gene_df.shape)

    print("Données cliniques :", clinical_df.shape)

    return gene_df, clinical_df


# ============================================================
# 3. PRÉPARATION DES IDENTIFIANTS
# ============================================================

def prepare_patient_ids(gene_df, clinical_df):

    """
    Prépare les identifiants pour garantir
    leur compatibilité avant la fusion.
    """

    print("\n========== PRÉPARATION DES IDENTIFIANTS ==========")

    # Transformer l'index génétique en colonne PATIENT_ID
    gene_df = gene_df.copy()

    gene_df.index = gene_df.index.astype("string").str.strip()

    gene_df.index.name = "PATIENT_ID"

    gene_df = gene_df.reset_index()

    # Nettoyer les identifiants cliniques
    clinical_df = clinical_df.copy()

    clinical_df["PATIENT_ID"] = (
        clinical_df["PATIENT_ID"]
        .astype("string")
        .str.strip()
    )

    # Vérifier les identifiants manquants
    if gene_df["PATIENT_ID"].isna().any():
        raise ValueError(
            "Des identifiants génétiques sont manquants."
        )

    if clinical_df["PATIENT_ID"].isna().any():
        raise ValueError(
            "Des identifiants cliniques sont manquants."
        )

    # Vérifier les doublons
    if gene_df["PATIENT_ID"].duplicated().any():
        raise ValueError(
            "Des identifiants génétiques sont dupliqués."
        )

    if clinical_df["PATIENT_ID"].duplicated().any():
        raise ValueError(
            "Des identifiants cliniques sont dupliqués."
        )

    print("Identifiants préparés avec succès.")

    return gene_df, clinical_df


# ============================================================
# 4. FUSION DES DATASETS
# ============================================================

def merge_datasets(gene_df, clinical_df):

    """
    Fusionne les données génétiques et cliniques.

    La fusion interne (inner join) conserve uniquement
    les patients présents dans les deux datasets.

    En cas de colonnes communes, on conserve les versions
    provenant du fichier clinique.
    """

    print("\n========== FUSION DES DATASETS ==========")

    # Identifier les patients communs
    gene_ids = set(gene_df["PATIENT_ID"])

    clinical_ids = set(clinical_df["PATIENT_ID"])

    common_ids = gene_ids.intersection(clinical_ids)

    gene_only = gene_ids - clinical_ids

    clinical_only = clinical_ids - gene_ids

    print("Patients génétiques :", len(gene_ids))

    print("Patients cliniques :", len(clinical_ids))

    print("Patients communs :", len(common_ids))

    print("Patients uniquement génétiques :", len(gene_only))

    print("Patients uniquement cliniques :", len(clinical_only))

    # --------------------------------------------------------
    # Identifier les colonnes communes
    # --------------------------------------------------------

    # PATIENT_ID est exclue car elle sert à la fusion.
    common_columns = (
        gene_df.columns
        .intersection(clinical_df.columns)
        .difference(["PATIENT_ID"])
    )

    print("\n========== COLONNES EN DOUBLE ==========")

    print("Colonnes communes :", common_columns.tolist())

    # Supprimer les colonnes communes du fichier génétique.
    # On conserve leurs versions provenant du fichier clinique.
    gene_df = gene_df.drop(columns=common_columns)

    print("Colonnes communes supprimées du fichier génétique.")

    # --------------------------------------------------------
    # Fusionner les deux datasets
    # --------------------------------------------------------

    merged_df = pd.merge(
        gene_df,
        clinical_df,
        on="PATIENT_ID",
        how="inner",
        validate="one_to_one"
    )

    print("\nFusion terminée.")

    return merged_df


# ============================================================
# 5. VÉRIFICATION DU DATASET FUSIONNÉ
# ============================================================

def check_merged_data(merged_df):

    """
    Vérifie la structure et la qualité
    du dataset après fusion.
    """

    print("\n========== VÉRIFICATION APRÈS FUSION ==========")

    print("Dimensions du dataset :", merged_df.shape)

    print("Nombre de patients :", len(merged_df))

    print(
        "Nombre de doublons PATIENT_ID :",
        merged_df["PATIENT_ID"].duplicated().sum()
    )

    print(
        "Nombre de valeurs manquantes :",
        merged_df.isna().sum().sum()
    )

    # Vérifier la présence de la variable cible
    if "GLEASON_SCORE" in merged_df.columns:

        print("\nGLEASON_SCORE est bien présente.")

        print("\nDistribution du GLEASON_SCORE :")

        print(
            merged_df["GLEASON_SCORE"]
            .value_counts(dropna=False)
            .sort_index()
        )

    else:

        raise ValueError(
            "ERREUR : GLEASON_SCORE est absente du dataset."
        )

    # Vérifier qu'aucune colonne n'a reçu de suffixe
    # indiquant un conflit de noms après la fusion.
    suffix_columns = [
        col for col in merged_df.columns
        if col.endswith("_x") or col.endswith("_y")
    ]

    if suffix_columns:
        print("\nAttention : colonnes avec suffixes détectées :")
        print(suffix_columns)
    else:
        print("\nAucun conflit de noms de colonnes détecté.")


# ============================================================
# 6. EXPORTATION DU DATASET FUSIONNÉ
# ============================================================

def save_merged_data(merged_df):

    """
    Enregistre le dataset fusionné au format CSV.
    """

    # Créer le dossier de sortie s'il n'existe pas
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== EXPORTATION ==========")

    print("Dataset fusionné enregistré dans :")

    print(OUTPUT_FILE)


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("\n============================================")

    print("       FUSION DES DONNÉES TCGA-PRAD")

    print("============================================")

    # Étape 1 : Chargement
    gene_df, clinical_df = load_data()

    # Étape 2 : Préparation des identifiants
    gene_df, clinical_df = prepare_patient_ids(
        gene_df,
        clinical_df
    )

    # Étape 3 : Fusion
    merged_df = merge_datasets(
        gene_df,
        clinical_df
    )

    # Étape 4 : Vérification
    check_merged_data(merged_df)

    # Étape 5 : Exportation
    save_merged_data(merged_df)

    print("\n============================================")

    print("       FUSION TERMINÉE AVEC SUCCÈS")

    print("============================================")


# ============================================================
# 8. POINT D'ENTRÉE DU PROGRAMME
# ============================================================

if __name__ == "__main__":

    main()
