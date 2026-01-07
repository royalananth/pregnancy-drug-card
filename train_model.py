import os
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CSV_PATH = "Master table_260 drugs_ADME_Protox.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- adjust these to match your CSV ----
SMILES_COL = "SMILES"

# Pick a few numeric outputs first (you can expand later)
# Replace these with exact column names from your table
TARGET_COLS = [
    # ADMET-style numeric probabilities
    "BBB",                 # e.g., BBB probability
    "P-gp substrate",      # probability 0-1
    "P-gp inhibitor",      # probability 0-1
    "Hepatotoxicity",      # probability 0-1
    "Neurotoxicity",       # probability 0-1
]

def mol_from_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    return Chem.MolFromSmiles(s.strip())

def featurize(mol):
    # ECFP4 fingerprint (2048 bits)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=np.float32)
    rdMolDescriptors.ConvertToNumpyArray(fp, fp_arr)

    # Small descriptor set (stable and useful)
    desc = np.array([
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.MolLogP(mol),
    ], dtype=np.float32)

    return np.concatenate([fp_arr, desc], axis=0)

df = pd.read_csv(CSV_PATH)

# Basic clean
df = df.dropna(subset=[SMILES_COL]).copy()

# Keep rows where targets exist (for first training round)
df = df.dropna(subset=[c for c in TARGET_COLS if c in df.columns]).copy()

missing = [c for c in TARGET_COLS if c not in df.columns]
if missing:
    raise ValueError(f"These TARGET_COLS are not in your CSV. Fix names: {missing}")

# Build X
X_list = []
keep_idx = []
for i, s in enumerate(df[SMILES_COL].tolist()):
    mol = mol_from_smiles(s)
    if mol is None:
        continue
    X_list.append(featurize(mol))
    keep_idx.append(i)

X = np.vstack(X_list)
Y = df.iloc[keep_idx][TARGET_COLS].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.18, random_state=42)

# Model (simple + reliable for small data)
model = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # fingerprints are sparse-like; with_mean=False is safer
    ("rf", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=600,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

# quick sanity metrics
r2 = r2_score(y_test, pred, multioutput="raw_values")
mae = mean_absolute_error(y_test, pred, multioutput="raw_values")

print("Targets:", TARGET_COLS)
print("R2 per target:", dict(zip(TARGET_COLS, r2)))
print("MAE per target:", dict(zip(TARGET_COLS, mae)))

joblib.dump(
    {
        "model": model,
        "target_cols": TARGET_COLS,
        "smiles_col": SMILES_COL,
    },
    os.path.join(OUT_DIR, "pregdrug_multioutput_rf.joblib")
)

print("Saved:", os.path.join(OUT_DIR, "pregdrug_multioutput_rf.joblib"))
