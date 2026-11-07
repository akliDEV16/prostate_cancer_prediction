from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FINAL = ROOT / "data" / "final"

# Chargement des données
X = pd.read_csv(PROCESSED / "X.csv")
y = pd.read_csv(PROCESSED / "y.csv").iloc[:, 0]
ids = pd.read_csv(PROCESSED / "patient_ids.csv")

# Division 80 % entraînement / 20 % test
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Création des dossiers
(FINAL / "train").mkdir(parents=True, exist_ok=True)
(FINAL / "test").mkdir(parents=True, exist_ok=True)

# Sauvegarde
X_train.to_csv(FINAL / "train/X_train.csv", index=False)
y_train.to_frame().to_csv(FINAL / "train/y_train.csv", index=False)
X_test.to_csv(FINAL / "test/X_test.csv", index=False)
y_test.to_frame().to_csv(FINAL / "test/y_test.csv", index=False)

ids_train.to_csv(FINAL / "train/patient_ids_train.csv", index=False)
ids_test.to_csv(FINAL / "test/patient_ids_test.csv", index=False)

print("Division terminée avec succès.")
print("Train :", len(X_train), "patients")
print("Test :", len(X_test), "patients")