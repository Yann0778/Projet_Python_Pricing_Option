# Option Pricing with Machine Learning
> Comparaison de trois approches ML pour prédire le **mid_price** d'options AAPL: Black-Scholes, Random Forest et XGBoost précédée d'une analyse exploratoire complète.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006400?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Table des matières

1. [Objectif du projet](#1-objectif-du-projet)
2. Construction de la base de données des options
3. [Pourquoi le mid_price ?](#2-pourquoi-le-mid_price-)
4. [Structure du projet](#3-structure-du-projet)
5. [Pipeline](#4-pipeline)
6. [Description des scripts](#5-description-des-scripts)
7. [Features utilisées](#6-features-utilisées)
8. [Installation](#7-installation)
9. [Configuration](#8-configuration)
10. [Exécution](#9-exécution)
11. [Outputs générés](#10-outputs-générés)
12. [Métriques comparatives](#11-métriques-comparatives)
13. [Paramètres clés](#12-paramètres-clés)
Bibliographie 

---

## 1. Objectif du projet

Le **pricing d'options** est un problème central en finance de marché : il s'agit de déterminer la juste valeur d'un contrat d'option avant qu'il ne soit échangé. Le modèle de référence historique, **Black-Scholes (1973)**, repose sur des hypothèses fortes: volatilité constante, marchés continus, absence de friction qui sont rarement vérifiées en pratique. Il en résulte des **biais systématiques**, notamment sur les options très en dehors ou très en dedans de la monnaie (OTM/ITM) et sur les maturités courtes.

Ce projet cherche à répondre à une question concrète :

> **Les modèles de machine learning peuvent-ils faire mieux que Black-Scholes pour prédire le prix de marché réel d'une option ?**

Pour y répondre, le projet suit une progression en quatre étapes sur des données réelles d'options sur l'action **Apple (AAPL)** :

- **Statistiques descriptives** : explorer la structure des données, les distributions de volatilité implicite, le smile de volatilité, les corrélations entre variables.
- **Black-Scholes** : établir un benchmark analytique rigoureux et quantifier ses biais structurels face aux prix observés.
- **Random Forest** : exploiter un modèle d'ensemble qui apprend directement depuis les données, sans hypothèse sur la dynamique du sous-jacent, et qui capture les non-linéarités que BS ne peut pas modéliser.
- **XGBoost** : aller plus loin avec le gradient boosting, réputé pour sa précision sur des données tabulaires financières, avec early stopping et optimisation poussée des hyperparamètres.

Les trois modèles de pricing sont évalués sur le **même test set (20% des données, split stratifié)** avec des métriques communes: RMSE, MAE, R², biais et MAPE pour garantir une comparaison équitable et **entièrement reproductible**.

---

## 2. Construction de la base de données des options


Afin de constituer une base de données exploitable pour l’analyse et la modélisation des prix d’options, nous avons développé une fonction Python permettant d’extraire automatiquement les données de marché à partir de la plateforme Yahoo Finance.

***Bibliothèques utilisées***

La mise en œuvre de ce processus repose sur plusieurs bibliothèques Python :

- **yfinance** : permet d’accéder aux données financières (actions, options, historiques de prix) depuis Yahoo Finance
- **pandas** : utilisée pour manipuler et structurer les données sous forme de DataFrame
- **datetime** : permet de gérer les dates, notamment pour le calcul du temps jusqu’à maturité
- **os** : utilisé pour gérer les chemins de fichiers et sauvegarder les données

Ces bibliothèques doivent être installées au préalable :

```python
pip install yfinance pandas datetime os
```

***Description de la fonction de construction des données***

La fonction **build_options_dataset** a pour objectif de construire automatiquement un jeu de données complet contenant les caractéristiques principales des options pour un actif donné (dans notre cas Apple, ticker "AAPL").

**Récupération des données de base**

La fonction commence par :

- créer un objet Ticker via yfinance
- extraire le prix spot de l’actif (dernier prix de clôture disponible)
- récupérer la liste des dates d’expiration des options disponibles

**Extraction des chaînes d’options**

Pour chaque date d’expiration :

- les options call et put sont récupérées via option_chain
- les deux types sont fusionnés dans un seul DataFrame
- une variable "type" est ajoutée pour distinguer calls et puts

**Sélection des variables pertinentes**

Seules les colonnes utiles sont conservées :

- strike : prix d’exercice
- lastPrice, bid, ask : informations de prix
- volume : liquidité
- impliedVolatility : volatilité implicite
- type : call ou put

Les données de toutes les maturités sont concaténées en un seul DataFrame, constituant ainsi la base de données finale
Les données sont ensuite exportées au format CSV dans un dossier local

code : 
***Création de la base de données***
```python
def build_options_dataset(ticker_symbol="AAPL", max_expirations=None):
    ticker = yf.Ticker(ticker_symbol)

    # spot price (reproductible à la date d'exécution)
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]

    # expirations disponibles (liste déterministe à date donnée)
    expirations = ticker.options

    if max_expirations is not None:
        expirations = expirations[:max_expirations]

    all_data = []

    # date "today" fixée pour reproductibilité (IMPORTANT)
    today = datetime(2026, 4, 16)

    for exp in expirations:
        try:
            opt = ticker.option_chain(exp)
        except Exception:
            continue

        calls = opt.calls.copy()
        puts = opt.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"

        df = pd.concat([calls, puts], ignore_index=True)

        # filtrage colonnes utiles
        cols = [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "volume",
            "impliedVolatility",
            "type"
        ]

        df = df[cols]

        # features dérivées
        df["spot"] = spot_price
        df["moneyness"] = df["spot"] / df["strike"]

        expiration_date = datetime.strptime(exp, "%Y-%m-%d")
        df["time_to_maturity"] = (expiration_date - today).days / 365

        df["expiration"] = exp
        df["ticker"] = ticker_symbol

        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    return full_df

data = build_options_dataset("AAPL")
```

***Sauvegarde de la base***

```python
import os

path = r"C:\Users\abeli\OneDrive\Documents\Deuxieme_Semestre\Projet_Pricing"

# sécurité : créer le dossier si besoin
os.makedirs(path, exist_ok=True)

file_path = os.path.join(path, "options_dataset.csv")

data.to_csv(file_path, index=False)

print("Dataset sauvegardé ici :", file_path)

```


## 3. Pourquoi le mid_price ?

La cible de tous les modèles est `mid_price = (bid + ask) / 2`.

| Variable | Description |
|----------|-------------|
| `lastPrice` | Dernier trade exécuté — potentiellement *stale* (obsolète) |
| `mid_price` | Centre du spread bid/ask — reflète l'état du carnet en temps réel ✓ |

Le `lastPrice` peut dater de plusieurs heures sur des options peu liquides, là où le `mid_price` est toujours calculable à partir des cotations en cours.

---

## 4. Structure du projet

```
option-pricing/
│
├── Statiqtiques_descriptives.py   # Script 1/4 — Analyse exploratoire (EDA)
├── BlackScholes.py                # Script 2/4 — Baseline Black-Scholes + Grecques
├── RandomForest.py                # Script 3/4 — Random Forest
├── XGBoost.py                     # Script 4/4 — XGBoost + comparaison finale
│
├── outputs/                       # Généré automatiquement à l'exécution
│   ├── split_indices.json             # Indices train/test partagés entre scripts
│   ├── options_with_bs_midprice.csv   # Dataset enrichi avec prix BS et Grecques
│   ├── bs_midprice_results.png
│   ├── rf_midprice_results.png
│   └── xgb_midprice_results.png
│
├── requirements.txt
└── README.md
```

---

## 5. Pipeline

```
options_dataset.csv
        │
        ▼
┌──────────────────────────────────────┐
│  Script 1 — Statistiques descriptives│
│  EDA, smile de vol, corrélations      │
└──────────────────────────────────────┘
        │
        ▼
  mid_price = (bid + ask) / 2
  Feature Engineering (7 variables)
  Train / Test split 80/20 stratifié
        │
        ├──▶ Script 2 : Black-Scholes  →  bs_midprice_results.png
        │                                  options_with_bs_midprice.csv
        │                                  split_indices.json
        │
        ├──▶ Script 3 : Random Forest  →  rf_midprice_results.png
        │                                  (lit split_indices.json)
        │
        └──▶ Script 4 : XGBoost        →  xgb_midprice_results.png
                                           comparaison finale BS / RF / XGB
```

---

## 6. Description des scripts

### `Statiqtiques_descriptives.py` — Script 1/4

Analyse exploratoire complète des données avant toute modélisation. Ce script ne produit pas de modèle, il sert à comprendre la structure des données.

**Étapes :**
1. Chargement du CSV et aperçu des dimensions
2. Nettoyage : imputation des valeurs manquantes (`bid`, `volume`) par la médiane
3. Feature engineering exploratoire (`mid_price`, `spread`, `log_moneyness`, `intrinsic_value`)
4. Statistiques descriptives globales
5. Matrice de corrélation des variables numériques
6. Smile de volatilité implicite par échéance (calls vs puts)
7. Boxplots de la volatilité implicite par échéance
8. Distribution du moneyness et prime vs strike
9. Heatmap de corrélation IV × maturité par niveau de moneyness

---

### `BlackScholes.py` — Script 2/4

Baseline analytique. Implémente la formule de Black-Scholes vectorisée et calcule les Grecques sur l'ensemble du dataset. **Génère également `split_indices.json`** — le fichier de référence qui garantit que RF et XGBoost sont évalués sur exactement le même test set.

**Étapes :**
1. Chargement et construction du `mid_price` (bid NaN → ask/2)
2. Formule BS vectorisée (calls et puts)
3. Calcul des Grecques : δ (delta), γ (gamma), ν (vega), θ (theta), ρ (rho)
4. Application sur le dataset complet
5. Métriques BS vs `mid_price` — dataset complet, calls séparés, puts séparés
6. Analyse des biais structurels par bucket de moneyness (Deep OTM → Deep ITM) et maturité
7. Sauvegarde du split 80/20 → `outputs/split_indices.json`
8. Visualisations → `bs_midprice_results.png`
9. Export du dataset enrichi → `outputs/options_with_bs_midprice.csv`

**Paramètre financier :**
```python
R_FREE = 0.043  # Taux Fed 2026 — à adapter si nécessaire
```

---

### `RandomForest.py` — Script 3/4

Premier modèle ML. Random Forest entraîné et optimisé par `RandomizedSearchCV` sur 100 combinaisons. Utilise le split défini par `BlackScholes.py` pour garantir la comparabilité des résultats.

**Étapes :**
1. Chargement, nettoyage et feature engineering (7 variables)
2. Chargement du split 80/20 depuis `outputs/split_indices.json`
3. `RandomizedSearchCV` (100 combinaisons, CV 5-fold)
4. Évaluation finale sur le test set (MAE, MSE, RMSE, R²)
5. Importance des features : MDI + Permutation Importance
6. Visualisations → `rf_midprice_results.png`

---

### `XGBoost.py` — Script 4/4

Modèle ML final. XGBoost avec triple split (train / validation / test), `RandomizedSearchCV` et early stopping.

**Étapes :**
1. Chargement, nettoyage et feature engineering (7 variables)
2. Triple split : 64% train / 16% validation / 20% test (stratifié)
3. `RandomizedSearchCV` (100 combinaisons, CV 5-fold)
4. Entraînement final avec early stopping (patience = 50 rounds) sur le set de validation
5. Évaluation finale (MAE, MSE, RMSE, R²)
6. Importance des features : MDI + Permutation Importance
7. Visualisations → `xgb_midprice_results.png`

---

## 7. Features utilisées

Les 7 variables suivantes sont communes aux scripts RF et XGBoost :

| Catégorie | Feature | Description |
|-----------|---------|-------------|
| Marché | `spot` | Prix du sous-jacent |
| Marché | `strike` | Prix d'exercice |
| Marché | `time_to_maturity` | Maturité en années |
| Engineered | `log_moneyness` | log(spot / strike) |
| Engineered | `intrinsic_value` | max(S−K, 0) pour call ; max(K−S, 0) pour put |
| Engineered | `log_volume` | log(volume) — proxy de liquidité |
| Flag | `is_call` | 1 = call, 0 = put |

> `BlackScholes.py` utilise en plus `impliedVolatility` pour calibrer le modèle directement aux conditions de marché.

---

## 8. Installation

**Prérequis** : Python ≥ 3.10

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn joblib
```

---

## 9. Configuration

> ⚠️ **À faire avant de lancer** — le chemin du fichier CSV est hardcodé dans chaque script. Il faut l'adapter à votre machine.

```python
# Ligne à modifier dans chacun des 4 scripts
data = pd.read_csv(r"C:\Users\ERAZER\Desktop\options_dataset.csv")

# Exemples selon votre OS :
data = pd.read_csv(r"C:\Users\VotreNom\Downloads\options_dataset.csv")   # Windows
data = pd.read_csv("/home/user/data/options_dataset.csv")                 # macOS / Linux
```

---

## 10. Exécution

> ⚠️ **Ordre obligatoire.** `BlackScholes.py` génère `split_indices.json` dont dépendent RF et XGBoost pour leur évaluation hors échantillon.

```bash
# Étape 1 — Analyse exploratoire
python Statiqtiques_descriptives.py

# Étape 2 — Black-Scholes (génère outputs/split_indices.json)
python BlackScholes.py

# Étape 3 — Random Forest (lit split_indices.json)
python RandomForest.py

# Étape 4 — XGBoost (lit split_indices.json)
python XGBoost.py
```

---

## 11. Outputs générés

| Fichier | Description | Généré par |
|---------|-------------|------------|
| `outputs/split_indices.json` | Indices train/test partagés entre les scripts | Script 2 (BS) |
| `outputs/options_with_bs_midprice.csv` | Dataset enrichi — prix BS, erreurs, Grecques | Script 2 (BS) |
| `bs_midprice_results.png` | 9 graphiques Black-Scholes | Script 2 (BS) |
| `rf_midprice_results.png` | 8 graphiques Random Forest | Script 3 (RF) |
| `xgb_midprice_results.png` | 8 graphiques XGBoost | Script 4 (XGB) |

---

## 12. Métriques comparatives

Les trois modèles sont évalués sur le **même test set 20%**. Les valeurs exactes dépendent du dataset et des hyperparamètres retenus par la recherche aléatoire.

```
--- Performance sur le test set (20%) ---

  Modèle            MAE ($)     RMSE ($)       R²
  ──────────────────────────────────────────────────
  Black-Scholes     X.XXXX      X.XXXX       X.XXXX
  Random Forest     X.XXXX      X.XXXX       X.XXXX
  XGBoost           X.XXXX      X.XXXX       X.XXXX
```

---

## 13. Paramètres clés

| Paramètre | Valeur | Fichier(s) |
|-----------|--------|-----------|
| `RANDOM_STATE` / `SEED` | 42 | Tous les scripts |
| `R_FREE` | 0.043 (4.3% — taux Fed 2026) | `BlackScholes.py` |
| Test size | 20% | `BlackScholes.py`, `RandomForest.py`, `XGBoost.py` |
| CV folds | 5 | `RandomForest.py`, `XGBoost.py` |
| `n_iter` RandomizedSearchCV | 100 | `RandomForest.py`, `XGBoost.py` |
| Early stopping rounds | 50 | `XGBoost.py` |
| Eval metric XGBoost | RMSE | `XGBoost.py` |


## Bibliographie 

Zeyuan Li &  Qingdao Huang (2025). Option Pricing Using Ensemble Learning
SERENADELLACORTE,LAURENSVANMIEGHEM,ANTONISPAPAPANTOLEON & ANDJONASPAPAZOGLOU-HENNIG (2026). MACHINE LEARNING FOR OPTION PRICING:
AN EMPIRICAL INVESTIGATION OF NETWORK ARCHITECTURES
