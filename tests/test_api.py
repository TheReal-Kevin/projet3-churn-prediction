"""Tests sur les routes de l'API FastAPI — sans avoir besoin de lancer le serveur."""

from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)

# Exemple de client utilisé dans tous les tests POST
CLIENT_EXEMPLE = {
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


def test_route_accueil():
    """GET / doit retourner un statut 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_route_health():
    """GET /health doit retourner status: ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_route_predict_statut_200():
    """POST /predict doit retourner un statut 200 pour des données valides."""
    response = client.post("/predict", json=CLIENT_EXEMPLE)
    assert response.status_code == 200


def test_route_predict_champs_presents():
    """La réponse de /predict doit contenir les 4 champs attendus."""
    response = client.post("/predict", json=CLIENT_EXEMPLE)
    data = response.json()

    assert "prediction" in data
    assert "churn" in data
    assert "probabilite_churn" in data
    assert "risque" in data


def test_route_predict_valeurs_valides():
    """La prédiction doit être 0 ou 1, et le churn 'Oui' ou 'Non'."""
    response = client.post("/predict", json=CLIENT_EXEMPLE)
    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["churn"] in ["Oui", "Non"]
    assert 0.0 <= data["probabilite_churn"] <= 1.0
    assert data["risque"] in ["Élevé", "Moyen", "Faible"]
