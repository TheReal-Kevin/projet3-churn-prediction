# Prédiction du Churn Client — Télécommunications

Projet #3 — Introduction au Machine Learning  
Kouame Adamou Kevin — AIA01 — L'École Multimédia

---

## Contexte

Développement d'un pipeline de Machine Learning pour prédire le churn (résiliation d'abonnement) de clients d'une entreprise de télécommunications, à partir du dataset **Telco Customer Churn** (Kaggle).

---

## Structure du projet

```
Projet_3/
├── .github/
│   └── workflows/
│       └── ci.yml                        # Pipeline CI/CD GitHub Actions
├── app/
│   ├── api.py                            # API de prédiction (FastAPI)
│   └── dashboard.py                      # Dashboard interactif (Dash)
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv      # Dataset brut (non versionné)
│   └── processed/
│       └── cleaned_telco.csv             # Dataset nettoyé
├── models/
│   ├── random_forest_final.pkl           # Modèle sauvegardé (non versionné)
│   └── feature_names.pkl                 # Noms des colonnes (non versionné)
├── notebooks/
│   ├── 01_preprocessing.ipynb            # Nettoyage & préparation des données
│   ├── 02_eda.ipynb                      # Analyse exploratoire (EDA)
│   ├── 03_models.ipynb                   # Entraînement & comparaison des modèles
│   └── 04_validation.ipynb               # Validation finale & monitoring Evidently
├── reports/
│   ├── data_drift_report.html            # Rapport dérive des données (Evidently)
│   └── model_performance_report.html     # Rapport performance modèle (Evidently)
├── src/
│   ├── preprocessing.py                  # Fonctions de nettoyage
│   ├── features.py                       # Feature engineering
│   ├── evaluation.py                     # Métriques & visualisations
│   └── save_model.py                     # Script de sauvegarde du modèle
├── tests/
│   ├── test_code_quality.py              # flake8 + black + pylint
│   ├── test_preprocessing.py             # Tests unitaires preprocessing
│   ├── test_model.py                     # Tests sur le modèle sauvegardé
│   └── test_api.py                       # Tests sur les routes FastAPI
├── requirements.txt
└── README.md
```

---

## Résultats des modèles

| Modèle | AUC-ROC | F1-Score |
|--------|---------|----------|
| Régression Logistique | 0.84 | 0.61 |
| Ridge Classifier | 0.83 | 0.59 |
| Arbre de Décision | 0.73 | 0.55 |
| Random Forest (optimisé) | 0.83 | 0.60 |

> Le modèle retenu est le **Random Forest optimisé** (GridSearchCV, StratifiedKFold 5 splits).  
> Note : un surapprentissage est observé (F1 train = 0.98 vs F1 test = 0.61) — documenté dans `04_validation.ipynb`.

---

## Installation

> Toutes les commandes suivantes se lancent depuis la **racine du projet** (`Projet_3/`).

```bash
# 1. Cloner le repo
git clone <url-du-repo>
cd Projet_3

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

# 4. Installer les dépendances
pip install -r requirements.txt
```

---

## Guide d'utilisation — étapes complètes

### Étape 1 — Préparer les données et explorer

> Placer le fichier brut `telco_customer_churn.csv` dans `data/raw/` avant de commencer.

```bash
# Depuis la racine du projet
jupyter lab
```

Exécuter les notebooks dans cet ordre :

| Notebook | Contenu |
|----------|---------|
| `notebooks/01_preprocessing.ipynb` | Nettoyage, encodage, normalisation → génère `data/processed/cleaned_telco.csv` |
| `notebooks/02_eda.ipynb` | Analyse exploratoire, visualisations, corrélations |
| `notebooks/03_models.ipynb` | Entraînement des 4 modèles, GridSearchCV, courbes ROC |
| `notebooks/04_validation.ipynb` | Validation finale, courbes d'apprentissage, rapport Evidently |

---

### Étape 2 — Sauvegarder le modèle final

> À faire après avoir exécuté les notebooks.

```bash
# Depuis la racine du projet
python src/save_model.py
```

Génère :
- `models/random_forest_final.pkl`
- `models/feature_names.pkl`

---

### Étape 3 — Lancer les tests

```bash
# Depuis la racine du projet
python -m pytest tests/ -v
```

| Fichier de test | Ce qui est vérifié |
|---|---|
| `test_code_quality.py` | flake8 (PEP8), black (formatage), pylint (qualité) |
| `test_preprocessing.py` | Fonctions de nettoyage des données |
| `test_model.py` | Chargement et prédiction du modèle |
| `test_api.py` | Routes GET et POST de l'API |

---

### Étape 4 — Lancer l'API de prédiction

```bash
# Depuis la racine du projet
uvicorn app.api:app --reload
```

- Documentation interactive : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Route principale : `POST /predict` — envoyer les données d'un client, recevoir la prédiction

Exemple de réponse :
```json
{
  "prediction": 0,
  "churn": "Non",
  "probabilite_churn": 0.275,
  "risque": "Faible"
}
```

---

### Étape 5 — Lancer le dashboard

```bash
# Depuis la racine du projet
python app/dashboard.py
```

- Dashboard : [http://127.0.0.1:8050](http://127.0.0.1:8050)
- Visualisations interactives : distribution du churn, charges mensuelles, ancienneté, filtres par type de contrat

---

## CI/CD — GitHub Actions

À chaque `push` ou `pull_request` sur `main` ou `dev`, le pipeline CI lance automatiquement :

```
pytest tests/test_code_quality.py -v
```

Configuration : `.github/workflows/ci.yml`

---

## Dépendances principales

- Python 3.11
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- imbalanced-learn
- evidently
- fastapi, uvicorn
- dash

Installation complète : `pip install -r requirements.txt`

---

## Auteur

Kouame Adamou Kevin — AIA01 — L'École Multimédia
