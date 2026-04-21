"""
=============================================================================
OPTION PRICING — TARGET : MID_PRICE
SCRIPT 1/3 — BLACK-SCHOLES BASELINE
=============================================================================
"""

"""

Description
-----------
Ce script implémente le modèle de Black-Scholes comme baseline théorique
pour le pricing d’options, avec pour objectif la prédiction du prix de marché
approximé par le mid_price (moyenne bid/ask).

Le modèle utilise la volatilité implicite observée sur le marché
(impliedVolatility), ce qui correspond à une calibration directe du modèle
aux conditions de marché.

Les résultats sont comparés aux prix réels afin d’analyser les biais
structurels du modèle de Black-Scholes.


Données utilisées
----------------
Le dataset contient des options avec les variables suivantes :
- spot : prix du sous-jacent
- strike : prix d’exercice
- time_to_maturity : maturité (en années)
- impliedVolatility : volatilité implicite
- bid / ask : prix de marché
- type : call / put

Le mid_price est défini comme :
    mid_price = (bid + ask) / 2


Pipeline
--------

1. Chargement et préparation des données
   - Lecture du dataset
   - Construction du mid_price
   - Suppression des valeurs aberrantes (prix <= 0)

2. Feature engineering
   - is_call (encodage binaire)
   - volume (remplissage des valeurs manquantes)
   - valeur intrinsèque
   - log-moneyness = log(S / K)
   - bid-ask spread

3. Implémentation du modèle Black-Scholes
   - Calcul des prix théoriques (call / put)
   - Calcul des grecques : delta, gamma, vega, theta, rho

4. Application du modèle
   - Pricing sur l’ensemble du dataset
   - Ajout des grecques au dataset

5. Évaluation des performances
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (coefficient de détermination)
   - Biais moyen (Mean Error)
   - MAPE (Mean Absolute Percentage Error)

   Évaluation effectuée :
   - sur l’ensemble du dataset
   - séparément pour calls et puts

6. Analyse des biais structurels
   - Biais par bucket de moneyness :
       Deep OTM → Deep ITM
   - Biais par maturité :
       court terme → long terme

   Objectif :
   identifier les zones où Black-Scholes est systématiquement biaisé

7. Évaluation hors échantillon
   - Utilisation du même split que le modèle Random Forest
   - Chargement des indices de test depuis un fichier JSON
   - Comparaison BS vs mid_price sur le test set

8. Visualisations
   - Prédit vs Réel
   - Résidus vs moneyness
   - Résidus vs maturité
   - Biais moyen par bucket
   - Distribution des erreurs
   - Distribution des grecques
   - Erreur vs volatilité × √T

9. Export des résultats
   - Graphiques (PNG)
   - Dataset enrichi avec :
       prix BS, erreurs, grecques, features


Objectif
--------
Fournir un benchmark théorique pour le pricing d’options et mettre en
évidence les limites structurelles du modèle de Black-Scholes face aux
prix de marché.

Ce script sert de référence pour comparaison avec des modèles
de machine learning (Random Forest, XGboost).


Remarques importantes
--------------------
- Le modèle suppose :
    * volatilité constante
    * absence de sauts
    * marchés frictionless

- Les écarts observés (biais) reflètent :
    * smile de volatilité
    * microstructure du marché (spread)
    * effets non capturés par BS

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, json
import requests
import pandas as pd
from io import StringIO
warnings.filterwarnings("ignore")

from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED  = 42
R_FREE = 0.043 #Taux de la Fed en 2026

# ─────────────────────────────────────────────
# ÉTAPE 1 — Chargement & mid_price
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement et construction du mid_price")
print("=" * 60)

def load_data_from_gdrive(url):
    """
    Télécharge un fichier CSV depuis un lien de partage Google Drive.
    """
    # Extraire l'ID du fichier depuis l'URL
    if "file/d/" in url:
        file_id = url.split("file/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1]
    else:
        raise ValueError("Impossible d'extraire l'ID du fichier depuis l'URL fournie.")
    
    # Construire l'URL de téléchargement direct
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print("Téléchargement du fichier depuis Google Drive...")
    try:
        response = requests.get(download_url)
        response.raise_for_status()  # Vérifie si la requête a réussi
        data = pd.read_csv(StringIO(response.text))
        print("Téléchargement réussi !")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement : {e}")
        return None

file_url = "https://drive.google.com/file/d/12KymJi6lZMmPRmaLB33klfMXRDTrIYJ1/view?usp=sharing"
df = load_data_from_gdrive(file_url)

df["bid"] = df["bid"].fillna(df["bid"].median())
df["mid_price"] = (df["bid"] + df["ask"]) / 2
df = df[df["mid_price"] > 0].reset_index(drop=True)

# Feature engineering (identique script 1)
df["is_call"]         = (df["type"] == "call").astype(int)
df["volume"]          = df["volume"].fillna(0)
df["intrinsic_value"] = np.where(
    df["is_call"]==1,
    np.maximum(df["spot"]-df["strike"],0),
    np.maximum(df["strike"]-df["spot"],0),
)
df["log_moneyness"] = np.log(df["spot"]/df["strike"])
df["spread"]        = df["ask"] - df["bid"].fillna(df["ask"])

print(f"Shape : {df.shape}  |  Calls : {(df['type']=='call').sum()}  Puts : {(df['type']=='put').sum()}")

# ─────────────────────────────────────────────
# ÉTAPE 2 — Formule Black-Scholes & Grecques
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Implémentation Black-Scholes")
print("=" * 60)
print(f"\nr = {R_FREE:.3f}  |  σ = impliedVolatility (vol de marché calibrée)")

def black_scholes(S, K, T, r, sigma, option_type="call"):
    T     = np.where(T <= 0, 1e-10, T)
    sigma = np.where(sigma <= 0, 1e-10, sigma)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    put  = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return np.where(option_type=="call", call, put)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    T     = np.where(T <= 0, 1e-10, T)
    sigma = np.where(sigma <= 0, 1e-10, sigma)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = np.where(option_type=="call", norm.cdf(d1), norm.cdf(d1)-1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega  = S*norm.pdf(d1)*np.sqrt(T)/100
    theta = np.where(
        option_type=="call",
        (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365,
        (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365,
    )
    rho = np.where(
        option_type=="call",
        K*T*np.exp(-r*T)*norm.cdf(d2)/100,
        -K*T*np.exp(-r*T)*norm.cdf(-d2)/100,
    )
    return pd.DataFrame({"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho})

# ─────────────────────────────────────────────
# ÉTAPE 3 — Application sur tout le dataset
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Application BS sur le dataset complet")
print("=" * 60)

df["bs_price"] = black_scholes(
    df["spot"].values, df["strike"].values,
    df["time_to_maturity"].values, R_FREE,
    df["impliedVolatility"].values, df["type"].values,
)
greeks = bs_greeks(
    df["spot"].values, df["strike"].values,
    df["time_to_maturity"].values, R_FREE,
    df["impliedVolatility"].values, df["type"].values,
)
df = pd.concat([df.reset_index(drop=True), greeks], axis=1)

print("\nAperçu prix BS vs mid_price :")
print(df[["type","strike","spot","time_to_maturity","mid_price","bs_price"]].head(8).to_string())

# ─────────────────────────────────────────────
# ÉTAPE 4 — Métriques BS (dataset complet)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Métriques BS vs mid_price (dataset complet)")
print("=" * 60)

def metrics_report(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    me   = float((y_pred - y_true).mean())
    mape = float(np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100)
    print(f"\n  [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.6f}  Biais={me:+.4f}  MAPE={mape:.2f}%")
    return dict(RMSE=rmse, MAE=mae, R2=r2, ME=me, MAPE=mape)

m_all   = metrics_report(df["mid_price"], df["bs_price"], "Tous — BS vs mid_price")
m_calls = metrics_report(df.loc[df["type"]=="call","mid_price"],
                         df.loc[df["type"]=="call","bs_price"], "Calls")
m_puts  = metrics_report(df.loc[df["type"]=="put","mid_price"],
                         df.loc[df["type"]=="put","bs_price"],  "Puts")

# ─────────────────────────────────────────────
# ÉTAPE 5 — Analyse des biais structurels
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Biais structurels BS vs mid_price")
print("=" * 60)

df["bs_error"]   = df["bs_price"] - df["mid_price"]
df["bs_abs_err"] = df["bs_error"].abs()

df["moneyness_bucket"] = pd.cut(
    df["log_moneyness"],
    bins=[-np.inf,-0.3,-0.1,-0.02,0.02,0.1,0.3,np.inf],
    labels=["Deep OTM","OTM","Slight OTM","ATM","Slight ITM","ITM","Deep ITM"],
)
df["ttm_bucket"] = pd.cut(
    df["time_to_maturity"],
    bins=[0,0.1,0.25,0.5,1.0,np.inf],
    labels=["<1M","1-3M","3-6M","6-12M",">1Y"],
)

print("\nBiais BS par moneyness :")
print(df.groupby("moneyness_bucket",observed=True)["bs_error"].agg(["mean","std","count"]).round(4).to_string())
print("\nBiais BS par maturité :")
print(df.groupby("ttm_bucket",observed=True)["bs_error"].agg(["mean","std","count"]).round(4).to_string())

# ─────────────────────────────────────────────
# ÉTAPE 6 — BS sur le même test set que RF
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — BS sur le test set (20%) — même split que RF")
print("=" * 60)

import os
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

with open(os.path.join(OUTPUT_DIR, "split_indices.json"), "r") as f:
    split_indices = json.load(f)

test_idx = split_indices["X_test_idx"]
df_test  = df.loc[df.index.isin(test_idx)]

y_test_mid = df_test["mid_price"]
y_pred_bs  = df_test["bs_price"]

m_bs_test = metrics_report(y_test_mid, y_pred_bs, "BS test set")

# ─────────────────────────────────────────────
# ÉTAPE 7 — Visualisations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Visualisations")
print("=" * 60)

fig = plt.figure(figsize=(20, 18))
fig.suptitle("Black-Scholes Baseline — target : mid_price (AAPL)", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
colors = df["type"].map({"call":"#4C72B0","put":"#DD8452"})
ax1.scatter(df["mid_price"], df["bs_price"], alpha=0.25, s=10, c=colors)
lim=[0, df["mid_price"].max()*1.05]
ax1.plot(lim,lim,"k--",lw=1.5)
ax1.set_title(f"BS vs mid_price (R²={r2_score(df['mid_price'],df['bs_price']):.4f})")
ax1.set_xlabel("mid_price réel ($)"); ax1.set_ylabel("BS prédit ($)")
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color="#4C72B0",label="Calls"),Patch(color="#DD8452",label="Puts")],fontsize=8)

ax2 = fig.add_subplot(gs[0,1])
for t,c in [("call","#4C72B0"),("put","#DD8452")]:
    mask=df["type"]==t
    ax2.scatter(df.loc[mask,"log_moneyness"],df.loc[mask,"bs_error"],alpha=0.25,s=10,color=c,label=t)
ax2.axhline(0,color="red",lw=1.5,linestyle="--")
ax2.set_title("Biais BS vs log-moneyness"); ax2.set_xlabel("log(S/K)"); ax2.set_ylabel("BS - mid ($)")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(df["time_to_maturity"],df["bs_error"],alpha=0.25,s=10,color="#55A868")
ax3.axhline(0,color="red",lw=1.5,linestyle="--")
ax3.set_title("Biais BS vs maturité"); ax3.set_xlabel("T (années)"); ax3.set_ylabel("BS - mid ($)")

ax4 = fig.add_subplot(gs[1,0])
bm = df.groupby("moneyness_bucket",observed=True)["bs_error"].mean()
colors_bar=["#C44E52" if v<0 else "#4C72B0" for v in bm]
ax4.bar(range(len(bm)),bm,color=colors_bar,alpha=0.85)
ax4.axhline(0,color="black",lw=1,linestyle="--")
ax4.set_xticks(range(len(bm))); ax4.set_xticklabels(bm.index,rotation=35,ha="right",fontsize=8)
ax4.set_title("Biais moyen BS par moneyness"); ax4.set_ylabel("Biais ($)")

ax5 = fig.add_subplot(gs[1,1])
bt = df.groupby("ttm_bucket",observed=True)["bs_error"].mean()
colors_bar2=["#C44E52" if v<0 else "#4C72B0" for v in bt]
ax5.bar(range(len(bt)),bt,color=colors_bar2,alpha=0.85)
ax5.axhline(0,color="black",lw=1,linestyle="--")
ax5.set_xticks(range(len(bt))); ax5.set_xticklabels(bt.index,fontsize=9)
ax5.set_title("Biais moyen BS par maturité"); ax5.set_ylabel("Biais ($)")

ax6 = fig.add_subplot(gs[1,2])
ax6.hist(df["bs_error"],bins=60,color="#9467BD",edgecolor="white",alpha=0.85)
ax6.axvline(0,color="black",lw=1.5,linestyle="--")
ax6.set_title("Distribution biais BS vs mid_price"); ax6.set_xlabel("BS - mid ($)")

ax7 = fig.add_subplot(gs[2,0])
ax7.hist(df["delta"][df["type"]=="call"],bins=40,alpha=0.65,color="#4C72B0",label="calls",density=True)
ax7.hist(df["delta"][df["type"]=="put"],bins=40,alpha=0.65,color="#DD8452",label="puts",density=True)
ax7.set_title("Distribution Delta"); ax7.set_xlabel("Delta"); ax7.legend(fontsize=8)

ax8 = fig.add_subplot(gs[2,1])
ax8.hist(df["vega"],bins=60,color="#2CA02C",edgecolor="white",alpha=0.85)
ax8.set_title("Distribution Vega (pour 1% σ)"); ax8.set_xlabel("Vega ($/%)")

ax9 = fig.add_subplot(gs[2,2])
ax9.scatter(df["vol_sqrt_t"] if "vol_sqrt_t" in df.columns
            else df["impliedVolatility"]*np.sqrt(df["time_to_maturity"]),
            df["bs_abs_err"], alpha=0.25, s=10, color="#8172B2")
ax9.set_title("Erreur abs. BS vs vol×√T"); ax9.set_xlabel("σ√T"); ax9.set_ylabel("|BS - mid| ($)")

plt.savefig("bs_midprice_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphiques → bs_midprice_results.png")

# Export dataset enrichi
cols_export = ["ticker","type","strike","spot","time_to_maturity","impliedVolatility",
               "mid_price","lastPrice","bs_price","bs_error","bs_abs_err",
               "log_moneyness","moneyness_bucket","ttm_bucket","delta","gamma","vega","theta","rho"]
df[cols_export].to_csv(
    os.path.join(OUTPUT_DIR, "options_with_bs_midprice.csv"),
    index=False
)
print("Dataset enrichi → options_with_bs_midprice.csv")

print(f"\nRésultats BS vs mid_price (dataset complet) :")
print(f"  RMSE={m_all['RMSE']:.4f}  MAE={m_all['MAE']:.4f}  R²={m_all['R2']:.6f}  Biais={m_all['ME']:+.4f}  MAPE={m_all['MAPE']:.2f}%")
print(f"Résultats BS vs mid_price (test set) :")
print(f"  RMSE={m_bs_test['RMSE']:.4f}  MAE={m_bs_test['MAE']:.4f}  R²={m_bs_test['R2']:.6f}  Biais={m_bs_test['ME']:+.4f}  MAPE={m_bs_test['MAPE']:.2f}%")
