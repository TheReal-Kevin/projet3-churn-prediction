"""Tests sur le modèle sauvegardé — vérifie qu'il existe et fonctionne."""

import os
import joblib
import pandas as pd


MODEL_PATH = "models/random_forest_final.pkl"
FEATURES_PATH = "models/feature_names.pkl"


def test_fichier_modele_existe():
    """Le fichier .pkl du modèle doit exister après l'entraînement."""
    assert os.path.exists(MODEL_PATH), f"Modèle introuvable : {MODEL_PATH}"


def test_fichier_features_existe():
    """Le fichier des noms de colonnes doit exister pour que l'API fonctionne."""
    assert os.path.exists(FEATURES_PATH), f"Features introuvables : {FEATURES_PATH}"


def test_modele_se_charge():
    """Le modèle doit pouvoir être chargé sans erreur."""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_modele_fait_une_prediction():
    """Le modèle doit retourner 0 ou 1 pour une ligne de données valide."""
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    # On crée une ligne avec toutes les colonnes à 0 (client par défaut)
    ligne = pd.DataFrame([{col: 0 for col in feature_names}])

    prediction = model.predict(ligne)[0]
    assert prediction in [0, 1], f"Prédiction inattendue : {prediction}"


def test_modele_retourne_une_probabilite():
    """La probabilité de churn doit être comprise entre 0 et 1."""
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    ligne = pd.DataFrame([{col: 0 for col in feature_names}])

    proba = model.predict_proba(ligne)[0][1]
    assert 0.0 <= proba <= 1.0, f"Probabilité hors limites : {proba}"


def test_nombre_de_features():
    """Le modèle doit attendre exactement 30 features (issues du preprocessing)."""
    feature_names = joblib.load(FEATURES_PATH)
    assert len(feature_names) == 30, f"Nombre de features inattendu : {len(feature_names)}"
