"""
API de prédiction du Churn Client — FastAPI
Lancement : uvicorn app.api:app --reload
Documentation : http://127.0.0.1:8000/docs
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.preprocessing import NUMERIC_COLS

# ── Chargement du modèle au démarrage ───────────────────────────────────────
# Chargé une seule fois au démarrage du serveur et réutilisé pour chaque requête.
model = joblib.load("models/random_forest_final.pkl")
feature_names = joblib.load("models/feature_names.pkl")
# Scaler entraîné sur les mêmes colonnes (tenure, MonthlyCharges, TotalCharges) que
# le modèle : indispensable pour lui fournir des valeurs standardisées, cohérentes
# avec ce qu'il a appris à l'entraînement.
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de résiliation d'un client.",
    version="1.0.0",
)


# ── Schéma des données d'entrée ─────────────────────────────────────────────
class ClientData(BaseModel):
    """Schéma de validation des données client reçues par la route /predict."""

    # Requis car le modèle a été entraîné avec "gender_Male" comme feature (One-Hot
    # Encoding) : sans ce champ, chaque prédiction supposerait silencieusement la
    # modalité de référence ("Female"), quel que soit le client réel.
    gender: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Contract: str
    InternetService: str
    PaymentMethod: str
    PaperlessBilling: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "SeniorCitizen": 0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "PaperlessBilling": 1,
                "Partner": 0,
                "Dependents": 0,
                "PhoneService": 1,
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            }
        }
    )


# ── Fonction de préparation des données ─────────────────────────────────────
def preparer_features(data: ClientData) -> pd.DataFrame:
    """
    Transforme les données brutes du client en un vecteur de features
    compatible avec le modèle entraîné.
    """
    # Toutes les colonnes attendues par le modèle démarrent à 0 (schéma One-Hot).
    row = {col: 0 for col in feature_names}

    row["SeniorCitizen"] = data.SeniorCitizen
    row["Partner"] = data.Partner
    row["Dependents"] = data.Dependents
    row["PhoneService"] = data.PhoneService
    row["PaperlessBilling"] = data.PaperlessBilling

    # Normalisation des colonnes continues avec le scaler entraîné : le modèle a
    # appris sur des valeurs standardisées, pas sur les valeurs brutes du client.
    valeurs_brutes = pd.DataFrame(
        [[data.tenure, data.MonthlyCharges, data.TotalCharges]], columns=NUMERIC_COLS
    )
    valeurs_normalisees = scaler.transform(valeurs_brutes)[0]
    for nom_colonne, valeur in zip(NUMERIC_COLS, valeurs_normalisees):
        row[nom_colonne] = valeur

    # Colonnes catégorielles : reconstruit le nom généré par le One-Hot Encoding
    # à l'entraînement (format "NomColonne_Valeur") pour activer le bon indicateur.
    colonnes_ohe = {
        "gender": data.gender,
        "MultipleLines": data.MultipleLines,
        "InternetService": data.InternetService,
        "OnlineSecurity": data.OnlineSecurity,
        "OnlineBackup": data.OnlineBackup,
        "DeviceProtection": data.DeviceProtection,
        "TechSupport": data.TechSupport,
        "StreamingTV": data.StreamingTV,
        "StreamingMovies": data.StreamingMovies,
        "Contract": data.Contract,
        "PaymentMethod": data.PaymentMethod,
    }

    for colonne, valeur in colonnes_ohe.items():
        nom_col_encodee = f"{colonne}_{valeur}"
        # Absent du dictionnaire = modalité de référence supprimée par
        # drop_first=True à l'entraînement ; on laisse alors la valeur à 0.
        if nom_col_encodee in row:
            row[nom_col_encodee] = 1

    return pd.DataFrame([row])


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def accueil():
    """Point d'entrée de l'API."""
    return {"message": "API Churn Prediction — voir /docs pour tester"}


@app.post("/predict")
def predire_churn(data: ClientData):
    """
    Prédit si un client va churner.
    Retourne la prédiction (0 ou 1) et la probabilité de churn.
    """
    features = preparer_features(data)

    prediction = int(model.predict(features)[0])
    probabilite = round(float(model.predict_proba(features)[0][1]), 4)

    return {
        "prediction": prediction,
        "churn": "Oui" if prediction == 1 else "Non",
        "probabilite_churn": probabilite,
        "risque": "Élevé" if probabilite > 0.6 else "Moyen" if probabilite > 0.3 else "Faible",
    }


@app.get("/health")
def health_check():
    """Vérifie que l'API tourne et que le modèle est bien chargé."""
    return {"status": "ok", "features_attendues": len(feature_names)}
