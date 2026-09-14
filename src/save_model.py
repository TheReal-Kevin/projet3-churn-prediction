"""
Script de sauvegarde du modèle final.
À lancer une seule fois depuis la racine du projet :
    python src/save_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Chargement des données ──────────────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned_telco.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Entraînement du modèle final ────────────────────────────────────────────
# Hyperparamètres retenus après GridSearchCV (notebook 03_models.ipynb).
# class_weight="balanced" compense le déséquilibre des classes (~27 % de churn).
modele = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
)

modele.fit(X_train, y_train)

# ── Sauvegarde ───────────────────────────────────────────────────────────────
joblib.dump(modele, "models/random_forest_final.pkl")
# L'ordre des colonnes est sauvegardé car le modèle attend un vecteur de features
# dans le même ordre qu'à l'entraînement (utilisé notamment par l'API).
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("Modele sauvegarde : models/random_forest_final.pkl")
print("Colonnes sauvegardees : models/feature_names.pkl")
print(f"Nombre de features : {len(X.columns)}")
