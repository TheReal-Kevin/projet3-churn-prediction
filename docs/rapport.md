# Rapport Final — Prédiction du Churn Client

**Projet #3 — Introduction au Machine Learning**  
**Auteur :** Kouame Adamou Kevin — AIA01  
**École :** L'École Multimédia  
**Date :** Juin 2026

---

## 1. Contexte et objectif

Une entreprise de télécommunications constate que certains clients résilient leur abonnement (phénomène appelé **churn**). Acquérir un nouveau client coûte en moyenne 5 à 7 fois plus cher que de retenir un client existant. L'objectif de ce projet est de construire un modèle de Machine Learning capable de **prédire si un client va résilier son abonnement**, afin de permettre à l'entreprise d'agir en amont (offre de rétention, contact commercial, etc.).

---

## 2. Source des données

**Dataset :** Telco Customer Churn — IBM Sample Dataset  
**Source :** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Taille :** 7 043 clients, 21 colonnes  
**Variable cible :** `Churn` (Yes / No)

### Description des variables principales

| Variable | Type | Description |
|---|---|---|
| `tenure` | Numérique | Ancienneté du client en mois |
| `MonthlyCharges` | Numérique | Frais mensuels (€) |
| `TotalCharges` | Numérique | Total facturé depuis le début |
| `Contract` | Catégorielle | Type de contrat (mensuel, 1 an, 2 ans) |
| `InternetService` | Catégorielle | Type de connexion (DSL, Fibre, Aucun) |
| `PaymentMethod` | Catégorielle | Mode de paiement |
| `SeniorCitizen` | Binaire | Client senior (1) ou non (0) |
| `Churn` | Binaire cible | Résiliation (Yes) ou non (No) |

### Distribution de la cible

Le dataset présente un **déséquilibre de classes** :
- Non-Churn (0) : ~73 % des clients
- Churn (1) : ~27 % des clients

Ce déséquilibre a été pris en compte dans tous les modèles via le paramètre `class_weight="balanced"`.

---

## 3. Analyse exploratoire (EDA)

L'analyse exploratoire (notebook `02_eda.ipynb`) a révélé plusieurs facteurs de risque de churn :

**Contrat mensuel (Month-to-month)**
Les clients en contrat mensuel ont un taux de churn bien supérieur à ceux en contrat 1 an ou 2 ans. Un contrat long engage le client et réduit le risque de résiliation.

**Ancienneté (tenure)**
Les clients qui churent ont une ancienneté très faible (pic dans les premiers mois). Un client fidèle depuis longtemps a très peu de chances de partir.

**Frais mensuels**
Les clients résiliés paient en moyenne des frais mensuels plus élevés. La fibre optique, plus chère, est associée à un churn plus fort.

**Service Internet (Fiber optic)**
Les abonnés fibre optique churent significativement plus que les abonnés DSL ou sans internet. Ce service est plus cher et a davantage de concurrence.

**Clients seniors**
Les seniors (SeniorCitizen = 1) churent à un taux environ deux fois supérieur aux non-seniors.

**Variables peu corrélées**
`gender` (homme/femme) n'a quasiment aucune corrélation avec le churn — cette variable a été gardée mais n'a pas d'impact significatif sur les prédictions.

---

## 4. Préparation des données (Preprocessing)

### 4.1 Nettoyage

- **TotalCharges** : la colonne contient des espaces vides pour les nouveaux clients (tenure = 0). Ces valeurs ont été converties en `float` avec `pd.to_numeric(errors='coerce')` et les NaN remplacés par `0.0`.
- **customerID** : supprimé — identifiant unique sans valeur prédictive.

### 4.2 Encodage de la variable cible

`Churn` : `"Yes"` → `1`, `"No"` → `0`.

### 4.3 Encodage des variables binaires

Les colonnes contenant uniquement `"Yes"` / `"No"` (Partner, Dependents, PhoneService, PaperlessBilling) ont été encodées en `1` / `0` directement.

### 4.4 Encodage One-Hot (OHE)

Les variables catégorielles à plus de 2 modalités (Contract, InternetService, PaymentMethod, MultipleLines, etc.) ont été transformées en colonnes binaires avec `pd.get_dummies(drop_first=True)`.

**Pourquoi `drop_first=True` ?** Pour éviter la multicolinéarité — si une variable a 3 modalités, 2 colonnes suffisent à tout représenter. La troisième est redondante.

### 4.5 Normalisation

Les trois colonnes numériques continues (`tenure`, `MonthlyCharges`, `TotalCharges`) ont été normalisées avec `StandardScaler`.

**Pourquoi StandardScaler ?** La régression logistique et le Ridge Classifier sont sensibles aux différences d'échelle entre variables. Sans normalisation, `TotalCharges` (valeurs jusqu'à 8000) dominerait `SeniorCitizen` (valeurs 0 ou 1). Le Random Forest et l'Arbre de Décision ne sont pas sensibles à l'échelle, mais l'appliquer uniformément assure la cohérence du pipeline.

### 4.6 Séparation train / test

Division stratifiée 80 % / 20 % avec `stratify=y` pour conserver la même proportion de churn dans les deux ensembles.

**Résultat final :** 30 colonnes de features, 5 634 lignes d'entraînement, 1 409 lignes de test.

---

## 5. Feature Engineering

Deux features supplémentaires ont été implémentées dans `src/features.py`, avec une justification métier :

- **`service_count`** : nombre de services souscrits par le client (PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies). Hypothèse : plus un client est engagé dans des services, moins il risque de partir.
- **`charge_per_month`** : `TotalCharges / tenure` (repli sur `MonthlyCharges` si `tenure = 0`). Hypothèse : capter un écart entre la dépense mensuelle actuelle et la moyenne historique du client (ex. fin de promotion).

**Test empirique et décision** : ces deux features ont été intégrées au pipeline et le modèle final a été réentraîné sur 3 variantes, évaluées sur le même jeu de test (split stratifié 80/20, `random_state=42`) :

| Variante | Nb features | AUC-ROC | F1-score (test) | F1-score (train) |
|---|---|---|---|---|
| **Baseline (retenue)** | 30 | **0.8415** | **0.6191** | 0.7719 |
| + `service_count` + `charge_per_month` | 32 | 0.8402 | 0.6259 | 0.7747 |
| + `service_count` seul | 31 | 0.8424 | 0.6287 | 0.7689 |

Les deux variantes enrichies n'apportent qu'un gain marginal de F1 sur le jeu de test (+0.007 avec les deux features, +0.010 avec `service_count` seul), du même ordre que la variabilité liée à un simple découpage train/test, et l'AUC ne bouge pas (0.84 dans les trois cas). À performance quasi égale, je garde le modèle le plus simple (principe de parcimonie) ; une validation croisée sur plusieurs découpages serait nécessaire avant de conclure à un vrai gain. `charge_per_month` est probablement redondante avec les variables déjà présentes (`tenure`, `MonthlyCharges`, `TotalCharges` sont mathématiquement liées), ce qui ajoute de la complexité sans apporter d'information réellement nouvelle au modèle.

**Décision finale : ces deux features ne sont pas intégrées au modèle en production.** Le code reste disponible dans `src/features.py` à titre de piste explorée et documentée, plutôt que supprimé.

---

## 6. Modèles testés

### 6.1 Choix des algorithmes

Quatre modèles ont été évalués, choisis pour leur complémentarité :

| Modèle | Raison du choix |
|---|---|
| **Régression Logistique** | Modèle de référence (baseline). Simple, interprétable, rapide. Bon point de départ pour comparer. |
| **Ridge Classifier** | Variante régularisée de la régression logistique. Utile pour voir si la pénalité L2 améliore la généralisation. |
| **Arbre de Décision** | Très interprétable — on peut visualiser les règles de décision. Permet de comprendre quels critères séparent les churners. |
| **Random Forest** | Ensemble d'arbres de décision. Généralement plus robuste que l'arbre seul, capable de capturer des interactions complexes entre variables. |

Tous les modèles ont été configurés avec `class_weight="balanced"` pour compenser le déséquilibre de classes (73% Non-Churn / 27% Churn).

### 6.2 Résultats comparatifs

| Modèle | AUC-ROC | F1-Score (test) | Précision | Rappel |
|---|---|---|---|---|
| Régression Logistique | **0.84** | 0.61 | 0.51 | **0.78** |
| Ridge Classifier | 0.84 | 0.61 | 0.50 | 0.79 |
| Arbre de Décision (profondeur 4) | 0.82 | **0.62** | 0.51 | 0.78 |
| Random Forest (base) | 0.82 | 0.55 | 0.64 | 0.49 |
| **Random Forest (optimisé, max_depth=10)** | **0.84** | **0.62** | 0.54 | 0.72 |

*Valeurs issues des sorties du notebook `03_models.ipynb` (même split stratifié 80/20, `random_state=42`).*

On observe deux profils : le Random Forest de base (arbres non limités en profondeur) a une **précision élevée** (0.64) mais un **rappel faible** (0.49) et surapprend fortement ; les modèles linéaires, l'arbre peu profond et le Random Forest optimisé (`max_depth=10`) ont un **rappel élevé** (0.72 à 0.79) pour une précision autour de 0.50-0.54. L'optimisation par GridSearchCV transforme le Random Forest : F1 0.55 → 0.62 et rappel 0.49 → 0.72, ce qui en fait le meilleur modèle testé sur le F1, à égalité avec la régression logistique sur l'AUC (0.84).

**Pourquoi l'AUC-ROC comme métrique principale ?**  
L'AUC-ROC mesure la capacité du modèle à distinguer les churners des non-churners **quelle que soit le seuil de décision**. Dans un contexte métier, on peut ajuster ce seuil selon la tolérance au risque. Le F1-Score est suivi en complément car il prend en compte le déséquilibre de classes.

### 6.3 Visualisation de l'Arbre de Décision

L'arbre de décision (profondeur 4) a permis d'identifier les règles les plus importantes :
1. Si `Contract_Two year = 1` → très faible probabilité de churn
2. Si `tenure` faible ET `MonthlyCharges` élevé → risque de churn élevé
3. Si `InternetService_Fiber optic = 1` ET `OnlineSecurity = 0` → risque accru

---

## 7. Optimisation du modèle

### 7.1 GridSearchCV

Le Random Forest a été optimisé avec une recherche par grille exhaustive :

```
Paramètres testés :
- n_estimators : [100, 200]
- max_depth : [None, 10, 20]
- min_samples_split : [2, 5]
- min_samples_leaf : [1, 2]
→ 24 combinaisons × 5 folds = 120 entraînements
```

**Métrique d'optimisation :** F1-Score (pertinent pour les classes déséquilibrées)  
**Validation croisée :** StratifiedKFold avec 5 splits (assure la représentation du churn dans chaque fold)

**Meilleurs hyperparamètres trouvés par GridSearchCV (notebook 03) :**
```
n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1
→ F1 (validation croisée) = 0.634 ; F1 test = 0.62, rappel = 0.72
```

Ces hyperparamètres sont ceux du modèle final (notebook 04 et `src/save_model.py`). Lors de la relecture finale, j'ai constaté que le modèle déployé avait d'abord été entraîné avec `max_depth=20` (F1 test 0.56, rappel 0.50, F1 train 0.996) : je l'ai réaligné sur le résultat de la recherche en grille, ce qui améliore nettement la généralisation (F1 test 0.62, rappel 0.72, F1 train 0.77).

---

## 8. Validation finale

### 8.1 Modèle retenu

Le **Random Forest optimisé** (`max_depth=10`) est retenu comme modèle final : c'est le meilleur F1 des modèles testés (0.62), avec un AUC égal à celui de la régression logistique (0.84), un rappel de 0.72 (près de trois churners sur quatre repérés) et une précision de 0.54. Il capture des interactions non linéaires entre variables et fournit une importance des variables directement exploitable pour l'équipe commerciale. La régression logistique reste une alternative crédible (rappel 0.78, aucun surapprentissage) : les deux modèles sont proches, et le choix final dépend du coût métier relatif d'un départ non détecté et d'une fausse alerte.

### 8.2 Surapprentissage détecté

| Métrique | Train | Test | Écart |
|---|---|---|---|
| F1-Score | 0.7719 | 0.6191 | **0.1528** |

Un surapprentissage (overfitting) modéré subsiste — bien moindre qu'avec `max_depth=20` (écart de 0.44 sur le F1). Les courbes d'apprentissage (learning curves) montrent que les scores d'entraînement et de validation se rapprochent quand le volume de données augmente.

**Causes probables de l'écart restant :**
- `min_samples_leaf=1` permet encore des feuilles avec un seul exemple
- 200 arbres de profondeur 10 restent un modèle très flexible pour 5 634 exemples

**Pistes d'amélioration :**
- Réduire `max_depth` (4 à 8)
- Augmenter `min_samples_leaf` (5 à 10)
- Essayer un modèle XGBoost ou LightGBM avec régularisation native

### 8.3 Monitoring avec Evidently

L'outil **Evidently** (version 0.7.21) a été utilisé pour analyser la dérive des données entre le jeu d'entraînement et le jeu de test.

**Pourquoi Evidently et pas Aporia ?**  
Evidently est open-source, installable localement (`pip install evidently`), sans compte ni clé API. Aporia est un SaaS (Software as a Service) payant orienté production. Pour un projet académique de Machine Learning, Evidently est le choix naturel.

**Résultats du rapport de dérive (`reports/data_drift_report.html`) :**
- Colonnes analysées : `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
- Dérive détectée : **0 / 4 colonnes**
- Conclusion : pas de dérive significative entre train et test — la séparation aléatoire stratifiée garantit des distributions similaires.

---

## 9. Déploiement

### 9.1 API de prédiction — FastAPI

**Pourquoi FastAPI et pas Flask ?**  
FastAPI génère automatiquement une documentation interactive (Swagger UI à `/docs`), valide les données d'entrée via Pydantic, et est nativement asynchrone. Flask nécessite des extensions supplémentaires pour ces fonctionnalités. Pour exposer un modèle ML en production, FastAPI est aujourd'hui le standard.

**Fonctionnement de l'API (`app/api.py`) :**
1. Chargement du modèle au démarrage (`models/random_forest_final.pkl`)
2. Route `POST /predict` : reçoit les données d'un client en JSON, prépare les 30 features, retourne la prédiction
3. Route `GET /health` : vérification que l'API est opérationnelle

**Exemple de réponse :**
```json
{
  "prediction": 0,
  "churn": "Non",
  "probabilite_churn": 0.275,
  "risque": "Faible"
}
```

Le niveau de risque est calculé ainsi :
- `probabilite_churn > 0.6` → Risque **Élevé**
- `probabilite_churn > 0.3` → Risque **Moyen**
- Sinon → Risque **Faible**

### 9.2 Dashboard interactif — Dash

**Pourquoi Dash ?**  
Dash (by Plotly) permet de créer des dashboards interactifs entièrement en Python, sans écrire de JavaScript. Les graphiques sont interactifs nativement (zoom, hover, filtres). C'est la solution la plus cohérente avec un workflow Data Science Python.

**Fonctionnalités du dashboard (`app/dashboard.py`) :**
- 4 KPIs dynamiques (total clients, résiliés, taux de churn, charge mensuelle)
- Filtre par type de contrat (mise à jour en temps réel de tous les graphiques)
- 6 visualisations : répartition churn, taux par contrat, frais mensuels, ancienneté, service internet, profil senior

---

## 10. CI/CD — GitHub Actions

Le fichier `.github/workflows/ci.yml` déclenche automatiquement les tests à chaque `push` ou `pull_request` sur les branches `main` et `dev`.

**Pourquoi GitHub Actions et pas GitLab CI ?**  
Le projet est hébergé sur GitHub. GitHub Actions est natif à la plateforme — pas de configuration de runner externe, pas d'account séparé. GitLab CI est excellent mais nécessite un compte GitLab ou un self-hosted runner.

**Ce que le pipeline vérifie :**
- `flake8` : respect des conventions PEP8
- `black --check` : formatage automatique du code
- `pylint` : qualité du code (score minimum 7/10)

---

## 11. Qualité du code

Trois outils complémentaires assurent la qualité du code Python :

| Outil | Rôle | Seuil |
|---|---|---|
| **flake8** | Détecte les violations PEP8 (longueur de ligne, espaces, imports inutilisés) | 0 violation, max-line-length=100 |
| **black** | Formatage automatique uniforme du code | Aucun fichier mal formaté |
| **pylint** | Analyse statique approfondie (variables non utilisées, logique, complexité) | Score ≥ 7.0 / 10 |

**19 tests automatisés** répartis en 4 fichiers couvrent la qualité du code, le preprocessing, le modèle et l'API.

---

## 12. Limites et perspectives

### Limites identifiées

1. **Surapprentissage résiduel** : le Random Forest reste un peu surajusté (F1 train = 0.77 vs test = 0.62). Des hyperparamètres plus conservateurs (`min_samples_leaf`, `max_depth` de 6 à 8) réduiraient encore cet écart.

2. **Taille du dataset** : 7 043 exemples est relativement faible pour un modèle complexe. Davantage de données améliorerait la généralisation.

3. **Features limitées** : le dataset ne contient pas d'informations comportementales (fréquence d'appel au support, historique des paiements en retard) qui seraient très prédictives du churn.

4. **Seuil fixe** : le seuil de classification est fixé à 0.5 par défaut. Un ajustement métier (privilégier le rappel pour ne rater aucun client à risque) pourrait améliorer l'utilité opérationnelle.

### Pistes d'amélioration

- Tester **XGBoost** ou **LightGBM** (meilleure régularisation native)
- Ajuster le **seuil de décision** selon le coût métier (faux négatif vs faux positif)
- Ajouter de la **validation temporelle** (les clients plus récents dans le jeu de test)
- Intégrer une **explication des prédictions** avec SHAP (quels facteurs ont conduit à la prédiction pour un client donné)

---

## 13. Conclusion

Ce projet a permis de construire un pipeline complet de Machine Learning, de la donnée brute jusqu'au déploiement :

1. **Exploration** — identification des variables les plus liées au churn (contrat, ancienneté, service internet)
2. **Preprocessing** — nettoyage, encodage OHE, normalisation, séparation stratifiée
3. **Modélisation** — 4 algorithmes comparés, optimisation par GridSearchCV
4. **Validation** — courbes d'apprentissage, détection du surapprentissage, monitoring Evidently
5. **Déploiement** — API FastAPI + dashboard Dash interactif
6. **CI/CD** — pipeline GitHub Actions automatisant les tests de qualité

Le modèle retenu (Random Forest optimisé, AUC = 0.84, F1 = 0.62, rappel = 0.72) est opérationnel et exposé via une API REST. Un surapprentissage a été identifié et documenté honnêtement — réduire la profondeur des arbres et augmenter le nombre minimum d'exemples par feuille sont les premières actions correctives à mener.
