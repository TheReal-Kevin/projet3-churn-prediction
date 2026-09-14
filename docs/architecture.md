# Architecture du Projet — Prédiction du Churn Client

---

## 1. Pipeline de données (Data Flow)

```mermaid
flowchart TD
    A[("📂 telco_customer_churn.csv\n7 043 clients · 21 colonnes\nKaggle / IBM Dataset")]

    subgraph PREP ["🔧 Preprocessing — src/preprocessing.py"]
        B1["Nettoyage\nTotalCharges → float\nSuppr. customerID"]
        B2["Encodage\nChurn Yes/No → 1/0\nOHE drop_first=True"]
        B3["Normalisation\nStandardScaler\ntenure · MonthlyCharges · TotalCharges"]
    end

    C[("💾 cleaned_telco.csv\n30 features · 7 043 lignes")]

    subgraph SPLIT ["✂️ Séparation stratifiée"]
        D1["X_train · y_train\n5 634 lignes · 80 %"]
        D2["X_test · y_test\n1 409 lignes · 20 %"]
    end

    subgraph MODELS ["🤖 Modélisation — notebooks/03_models.ipynb"]
        E1["Régression Logistique\nAUC 0.84 · F1 0.61"]
        E2["Ridge Classifier\nAUC 0.84 · F1 0.61"]
        E3["Arbre de Décision\nAUC 0.82 · F1 0.62"]
        E4["Random Forest\nAUC 0.82 · F1 0.55"]
    end

    subgraph OPTIM ["⚙️ Optimisation — GridSearchCV"]
        F["StratifiedKFold n=5\n24 combinaisons · 120 runs\nMétrique : F1-Score"]
    end

    G["🏆 Random Forest Optimisé\nn_estimators=200 · max_depth=20\nAUC 0.83 · F1 0.56"]

    subgraph VALID ["✅ Validation — notebooks/04_validation.ipynb"]
        H1["Courbes d'apprentissage\nSurapprentissage détecté\nF1 train=0.996 vs test=0.56"]
        H2["Monitoring Evidently\nData Drift : 0/4 colonnes\nPas de dérive détectée"]
    end

    I[("💾 random_forest_final.pkl\nfeature_names.pkl")]

    subgraph DEPLOY ["🚀 Déploiement"]
        J1["API FastAPI\nPOST /predict\n:8000/docs"]
        J2["Dashboard Dash\n4 KPIs · 6 graphiques\n:8050"]
    end

    A --> PREP
    B1 --> B2 --> B3
    PREP --> C
    C --> SPLIT
    D1 --> MODELS
    MODELS --> OPTIM
    OPTIM --> G
    G --> VALID
    G --> I
    I --> J1
    C --> J2
    D2 --> H1
    D2 --> H2
```

---

## 2. Architecture technique (Stack)

```mermaid
flowchart LR
    subgraph DATA ["📊 Données"]
        d1["CSV brut\ndata/raw/"]
        d2["CSV nettoyé\ndata/processed/"]
        d3["Modèle .pkl\nmodels/"]
    end

    subgraph CODE ["🐍 Code Python"]
        subgraph SRC ["src/"]
            s1["preprocessing.py\nnettoyage & encodage"]
            s2["features.py\nfeature engineering"]
            s3["evaluation.py\nmétriques & graphiques"]
            s4["save_model.py\nsauvegarde joblib"]
        end
        subgraph NB ["notebooks/"]
            n1["01_preprocessing"]
            n2["02_eda"]
            n3["03_models"]
            n4["04_validation"]
        end
    end

    subgraph DEPLOY2 ["🚀 Applications"]
        a1["app/api.py\nFastAPI + Uvicorn\nREST API :8000"]
        a2["app/dashboard.py\nDash + Plotly\nDashboard :8050"]
    end

    subgraph QUALITY ["🧪 Qualité & CI/CD"]
        q1["tests/\nflake8 · black · pylint\npytest · 19 tests"]
        q2[".github/workflows/ci.yml\nGitHub Actions\nPush → Tests auto"]
    end

    subgraph MONITORING ["📈 Monitoring"]
        m1["Evidently 0.7.21\nData Drift Report\nClassification Report"]
        m2["reports/\nHTML générés"]
    end

    subgraph LIBS ["📦 Librairies principales"]
        l1["pandas · numpy"]
        l2["scikit-learn"]
        l3["matplotlib · seaborn · plotly"]
        l4["fastapi · uvicorn · pydantic"]
        l5["dash · plotly"]
        l6["evidently"]
        l7["joblib"]
    end

    d1 --> SRC
    SRC --> d2
    NB --> SRC
    d2 --> NB
    d3 --> a1
    d1 --> a2
    SRC --> QUALITY
    DEPLOY2 --> QUALITY
    QUALITY --> q2
    NB --> MONITORING
    MONITORING --> m2
    LIBS --> CODE
    LIBS --> DEPLOY2
```

---

## 3. Architecture de l'API (FastAPI)

```mermaid
sequenceDiagram
    participant Client as 👤 Client
    participant API as ⚡ FastAPI (app/api.py)
    participant Model as 🤖 RandomForest (.pkl)

    Client->>API: POST /predict\n{ tenure, Contract, MonthlyCharges... }
    API->>API: Validation Pydantic\n(18 champs typés)
    API->>API: preparer_features()\n18 champs → 30 colonnes OHE
    API->>Model: model.predict(features)
    Model-->>API: prediction [0 ou 1]
    API->>Model: model.predict_proba(features)
    Model-->>API: probabilité [0.0 → 1.0]
    API-->>Client: { prediction, churn, probabilite_churn, risque }
```

---

## 4. Structure des dossiers

```
Projet_3/
│
├── .github/workflows/ci.yml     ← CI/CD GitHub Actions
│
├── app/
│   ├── api.py                   ← API REST (FastAPI + Uvicorn)
│   └── dashboard.py             ← Dashboard interactif (Dash + Plotly)
│
├── data/
│   ├── raw/                     ← Dataset brut Kaggle (non versionné)
│   └── processed/               ← CSV nettoyé (versionné)
│
├── docs/
│   ├── rapport.md               ← Rapport final & justification des choix
│   ├── competences.md           ← Mapping compétences brief ↔ réalisations
│   └── architecture.md          ← Ce fichier
│
├── models/
│   ├── random_forest_final.pkl  ← Modèle entraîné (non versionné)
│   └── feature_names.pkl        ← Noms des 30 features (non versionné)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb   ← Nettoyage & préparation
│   ├── 02_eda.ipynb             ← Analyse exploratoire
│   ├── 03_models.ipynb          ← Entraînement & comparaison
│   └── 04_validation.ipynb      ← Validation & monitoring
│
├── reports/
│   ├── data_drift_report.html   ← Rapport Evidently (dérive)
│   └── model_performance_report.html
│
├── src/
│   ├── preprocessing.py         ← Fonctions de nettoyage
│   ├── features.py              ← Feature engineering
│   ├── evaluation.py            ← Métriques & visualisations
│   └── save_model.py            ← Script de sauvegarde
│
├── tests/
│   ├── test_code_quality.py     ← flake8 + black + pylint
│   ├── test_preprocessing.py    ← Tests unitaires preprocessing
│   ├── test_model.py            ← Tests modèle sauvegardé
│   └── test_api.py              ← Tests routes FastAPI
│
├── requirements.txt
└── README.md
```
