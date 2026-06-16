# Prédiction du Churn Client — Télécommunications

Projet #3 — Introduction au Machine Learning  

## Contexte

Développement d'un pipeline de Machine Learning pour prédire le churn (résiliation d'abonnement) de clients d'une entreprise de télécommunications, à partir du dataset **Telco Customer Churn** (Kaggle).

## Structure du projet

```
Projet_3/
├── data/
│   ├── telco_customer_churn.csv      # Dataset brut (ignoré par git)
│   └── cleaned_telco.csv             # Dataset nettoyé
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Nettoyage & préparation des données
│   ├── 02_eda.ipynb                  # Analyse exploratoire (EDA)
│   ├── 03_models.ipynb               # Entraînement & comparaison des modèles
│   └── 04_validation.ipynb           # Validation finale & conclusions
├── src/
│   ├── preprocessing.py              # Fonctions de nettoyage
│   ├── features.py                   # Feature engineering
│   └── evaluation.py                 # Métriques & visualisations
├── docs/
│   └── rapport.md                    # Rapport final
└── README.md
```

## Installation

```bash
# Cloner le repo
git clone <url-du-repo>
cd Projet_3

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## Usage

Lancer Jupyter Lab et exécuter les notebooks dans l'ordre :

```bash
jupyter lab
```

1. `01_preprocessing.ipynb` — Nettoyage et préparation des données
2. `02_eda.ipynb` — Analyse exploratoire
3. `03_models.ipynb` — Modèles ML et optimisation
4. `04_validation.ipynb` — Validation finale

## Modèles testés

| Modèle | AUC-ROC | F1-Score |
|--------|---------|----------|
| Régression Logistique | — | — |
| Arbre de Décision | — | — |
| Random Forest | — | — |

> Les résultats seront complétés après entraînement.

## Résultats

> À compléter après la phase de modélisation.

## Dépendances principales

- Python 3.11
- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- imbalanced-learn
- evidently

## Auteur

Kouame Adamou Kevin — AIA01
