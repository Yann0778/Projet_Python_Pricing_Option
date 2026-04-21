# Option Pricing with Machine Learning
> Comparaison de trois approches pour prédire le **mid_price** d'options AAPL: Black-Scholes, Random Forest et réseau de neurones MLP.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Table des matières

1. [Objectif du projet](#objectif-du-projet)
2. [Pourquoi le mid_price ?](#pourquoi-le-mid_price-)
3. [Structure du projet](#structure-du-projet)
4. [Pipeline](#pipeline)
5. [Description des scripts](#description-des-scripts)
6. [Features utilisées](#features-utilisées-11-variables)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Exécution](#exécution)
10. [Outputs générés](#outputs-générés)
11. [Métriques comparatives](#métriques-comparatives-test-set-20)
12. [Paramètres clés](#paramètres-clés)

---

## Objectif du projet

Le pricing d'options est un problème central en finance de marché : déterminer la juste valeur d'un contrat d'option avant qu'il ne soit échangé. Le modèle de référence historique, Black-Scholes (1973), repose sur des hypothèses fortes: volatilité constante, marchés continus, absence de friction, qui sont rarement vérifiées en pratique. Il en résulte des biais systématiques, notamment sur les options très en dehors ou très en dedans de la monnaie (OTM/ITM), et sur les maturités courtes.

Ce projet cherche à répondre à une question concrète : **les modèles de machine learning peuvent-ils faire mieux que Black-Scholes pour prédire le prix de marché réel d'une option ?**

Pour y répondre, trois approches sont comparées sur des données réelles d'options sur l'action Apple (AAPL) :

- **Black-Scholes** sert de baseline analytique et permet de quantifier les biais du modèle théorique face aux prix observés.
- **Random Forest** est un modèle d'ensemble qui apprend directement depuis les données sans hypothèse sur la dynamique du sous-jacent. Il capture les non-linéarités que BS ne peut pas modéliser.
- **MLP (réseau de neurones multicouche)** est une approche plus flexible encore, capable d'apprendre des représentations profondes des features d'options.

Les trois modèles sont évalués sur le même test set (20% des données, split stratifié) avec des métriques communes: RMSE, MAE, R², biais et MAPE pour permettre une comparaison équitable et reproductible.

---

## Pourquoi le mid_price ?

La cible de tous les modèles est `mid_price = (bid + ask) / 2`.

| Métrique | Description |
|----------|-------------|
| `lastPrice` | Dernier trade — potentiellement *stale* (vieux) |
| `mid_price` | Centre du spread actuel → reflète l'état du carnet en temps réel ✓ |

---

## Structure du projet

```
option-pricing/
├── RandomForest.py          # Script 1/3 — Random Forest
├── BlackScholes.py          # Script 2/3 — Black-Scholes baseline
├── ReseauDeNeuronne.py      # Script 3/3 — MLP + comparaison finale
├── outputs/                 # Généré automatiquement à l'exécution
│   ├── rf_midprice.pkl
│   ├── mlp_midprice.pkl
│   ├── split_indices.json
│   └── options_with_bs_midprice.csv
└── README.md
```

---

## Pipeline

```
options_dataset.csv
       │
       ▼
  mid_price = (bid + ask) / 2
       │
       ▼
  Feature Engineering (11 variables)
       │
       ▼
  Train / Test split 80/20 (stratifié sur is_call)
       │
       ├──▶ Script 1 : Random Forest   →  rf_midprice.pkl
       ├──▶ Script 2 : Black-Scholes   →  options_with_bs_midprice.csv
       └──▶ Script 3 : MLP             →  mlp_midprice.pkl + comparaison finale
```

---

## Description des scripts

### `RandomForest.py` — Script 1/3

Modèle ML principal. Entraîne un Random Forest avec optimisation par GridSearchCV.

**Étapes :**
1. Chargement du CSV et construction du `mid_price`
2. Feature engineering (11 variables)
3. Split 80/20 stratifié, sauvegarde des indices dans `split_indices.json`
4. Baseline RF (100 estimateurs)
5. Cross-validation 5-fold
6. GridSearchCV (32 combinaisons)
7. Évaluation finale sur le test set
8. Analyse de l'importance des features (MDI + Permutation)
9. Visualisations → `rf_midprice_results.png`
10. Sauvegarde du modèle → `outputs/rf_midprice.pkl`

---

### `BlackScholes.py` — Script 2/3

Modèle analytique de référence. Implémente la formule de Black-Scholes et calcule les Grecques.

**Étapes :**
1. Chargement et feature engineering (identique Script 1)
2. Formule BS vectorisée + Grecques (δ, γ, ν, θ, ρ)
3. Application sur le dataset complet
4. Métriques BS vs `mid_price` (dataset complet)
5. Analyse des biais structurels par moneyness et maturité
6. BS évalué sur le même test set 20% que le RF (via `split_indices.json`)
7. Visualisations → `bs_midprice_results.png`
8. Export dataset enrichi → `outputs/options_with_bs_midprice.csv`

**Paramètre financier :**
```python
R_FREE = 0.043  # Taux Fed 2026 à adapter si nécessaire
```

---

### `ReseauDeNeuronne.py` — Script 3/3

MLP sklearn avec normalisation, GridSearch sur l'architecture, puis tableau comparatif final des 3 modèles.

**Étapes :**
1. Chargement et feature engineering (identique Scripts 1 & 2)
2. StandardScaler (fit sur X_train uniquement — pas de data leakage)
3. MLP baseline `(128, 64)` ReLU, Adam, early stopping
4. Cross-validation 5-fold (Pipeline scaler + MLP)
5. GridSearchCV (30 combinaisons d'architecture et régularisation)
6. Évaluation MLP optimisé
7. Recalcul BS et chargement RF sur le même test set
8. Visualisations complètes + tableau de synthèse → `mlp_midprice_comparison.png`
9. Tableau terminal final avec classement et vainqueur 
10. Sauvegarde → `outputs/mlp_midprice.pkl`

---

## Features utilisées (11 variables)

| Catégorie | Features |
|-----------|----------|
| Marché | `strike`, `spot`, `impliedVolatility`, `time_to_maturity` |
| Engineered | `log_moneyness`, `intrinsic_value`, `vol_sqrt_t` |
| Flags | `is_call` (0/1), `itm` (0/1) |
| Liquidité | `spread` (ask − bid), `log_volume` |

---

## Installation

### Prérequis

- Python ≥ 3.10

### Dépendances

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib
```

---

## Configuration

> ⚠️ **À faire avant de lancer** : adapter le chemin du fichier CSV dans chacun des 3 scripts.

```python
# Ligne à modifier dans RandomForest.py, BlackScholes.py, ReseauDeNeuronne.py
df = pd.read_csv(r"C:\Users\DELL\Downloads\options_dataset.csv")

# Remplacer par votre chemin, exemples :
df = pd.read_csv(r"C:\Users\VotreNom\Downloads\options_dataset.csv")  # Windows
df = pd.read_csv("/home/user/data/options_dataset.csv")                # macOS / Linux
```

---

## Exécution

> ⚠️ **Les scripts doivent être lancés dans l'ordre.** Le Script 1 génère `split_indices.json` et `rf_midprice.pkl` dont dépendent les suivants.

```bash
# Étape 1 — Random Forest (génère split_indices.json + rf_midprice.pkl)
python RandomForest.py

# Étape 2 — Black-Scholes (lit split_indices.json)
python BlackScholes.py

# Étape 3 — MLP + comparaison finale (lit rf_midprice.pkl)
python ReseauDeNeuronne.py
```

---

## Outputs générés

| Fichier | Type | Généré par |
|---------|------|------------|
| `rf_midprice_results.png` | Visualisations RF | Script 1 |
| `outputs/rf_midprice.pkl` | Modèle RF sérialisé | Script 1 |
| `outputs/split_indices.json` | Indices train/test | Script 1 |
| `bs_midprice_results.png` | Visualisations BS | Script 2 |
| `outputs/options_with_bs_midprice.csv` | Dataset enrichi avec prix BS et Grecques | Script 2 |
| `mlp_midprice_comparison.png` | Comparaison complète 3 modèles | Script 3 |
| `outputs/mlp_midprice.pkl` | Modèle MLP sérialisé | Script 3 |

---

## Métriques comparatives (test set 20%)

Les valeurs exactes s'affichent dans le terminal à la fin de `ReseauDeNeuronne.py`. Le classement final est calculé automatiquement :

```
TABLEAU DE SYNTHÈSE FINAL — TARGET : MID_PRICE — Test Set (20%)

  Métrique           Black-Scholes    Random Forest              MLP
  ──────────────────────────────────────────────────────────────
  RMSE ($)            X.XXXX           X.XXXX ✓          X.XXXX
  MAE  ($)            X.XXXX           X.XXXX            X.XXXX ✓
  R²                  X.XXXXXX         X.XXXXXX ✓        X.XXXXXX
  Biais ($)          +X.XXXX          +X.XXXX            X.XXXX ✓
  MAPE (%)            XX.XX            X.XX ✓            X.XX

  🏆 MEILLEUR MODÈLE (mid_price) : Random Forest  (ou MLP selon vos données)
```

---

## Paramètres clés

| Paramètre | Valeur | Fichier |
|-----------|--------|---------|
| `SEED` | 42 | Tous les scripts |
| `R_FREE` | 0.043 (4.3%) | `BlackScholes.py`, `ReseauDeNeuronne.py` |
| Test size | 20% | `RandomForest.py` |
| CV folds | 5 | Tous les scripts ML |
| MLP architecture baseline | `(128, 64)` ReLU | `ReseauDeNeuronne.py` |
| Solver MLP | Adam | `ReseauDeNeuronne.py` |
