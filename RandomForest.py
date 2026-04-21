"""
=============================================================================
OPTION PRICING — TARGET : MID_PRICE  (bid+ask)/2
SCRIPT 1/3 — RANDOM FOREST
=============================================================================
mid_price est une meilleure approximation du "vrai" prix de marché :
  - lastPrice = dernier trade, potentiellement stale (vieux)
  - mid_price = centre du spread actuel → reflète l'état du carnet en temps réel
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# ÉTAPE 1 — Chargement
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement et construction du mid_price")
print("=" * 60)

df = pd.read_csv(r"C:\Users\DELL\Downloads\options_dataset.csv")
print(f"Shape brut : {df.shape}")

# Construction du mid_price
# bid=NaN → options illiquides deep OTM, bid implicitement ~0 → mid = ask/2
df["mid_price"] = np.where(
    df["bid"].isna(),
    df["ask"] / 2,
    (df["bid"] + df["ask"]) / 2,
)

print(f"\nNaN mid_price après imputation : {df['mid_price'].isna().sum()}")
print(f"Comparaison mid vs lastPrice :")
print(f"  mid_price  — mean={df['mid_price'].mean():.3f}  std={df['mid_price'].std():.3f}")
print(f"  lastPrice  — mean={df['lastPrice'].mean():.3f}  std={df['lastPrice'].std():.3f}")
print(f"  Diff moy (mid - last) : {(df['mid_price'] - df['lastPrice']).mean():.4f} $")

# Supprimer les lignes où mid_price = 0 (options théoriquement sans valeur)
df = df[df["mid_price"] > 0].reset_index(drop=True)
print(f"\nAprès suppression mid_price=0 : {df.shape[0]} lignes")

# ─────────────────────────────────────────────
# ÉTAPE 2 — Feature engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Feature engineering")
print("=" * 60)

data = df.copy()
data["is_call"]         = (data["type"] == "call").astype(int)
data["volume"]          = data["volume"].fillna(0)

data["intrinsic_value"] = np.where(
    data["is_call"] == 1,
    np.maximum(data["spot"] - data["strike"], 0),
    np.maximum(data["strike"] - data["spot"], 0),
)
data["itm"]             = (data["intrinsic_value"] > 0).astype(int)
data["log_moneyness"]   = np.log(data["spot"] / data["strike"])
data["spread"]          = data["ask"] - data["bid"].fillna(data["ask"])  # spread ≈ ask si bid NaN
data["vol_sqrt_t"]      = data["impliedVolatility"] * np.sqrt(data["time_to_maturity"])
data["log_volume"]      = np.log1p(data["volume"])
data["time_value"]      = data["mid_price"] - data["intrinsic_value"]   # valeur temps

print("\nFeatures créées : intrinsic_value, itm, log_moneyness, spread,")
print("                  vol_sqrt_t, log_volume, time_value")

# ─────────────────────────────────────────────
# ÉTAPE 3 — Features & target
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Sélection features / target")
print("=" * 60)

FEATURES = [
    "strike", "spot", "time_to_maturity", "impliedVolatility",
    "is_call", "log_moneyness", "intrinsic_value", "itm",
    "vol_sqrt_t", "log_volume", "spread",
]
TARGET = "mid_price"

X = data[FEATURES]
y = data[TARGET]

print(f"Features ({len(FEATURES)}) : {FEATURES}")
print(f"Target   : {TARGET}")
print(f"X : {X.shape}  |  y : {y.shape}")
print(f"\nDistribution de mid_price :")
print(y.describe().round(4))

# ─────────────────────────────────────────────
# ÉTAPE 4 — Train / Test split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Split 80/20 stratifié")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
    stratify=data.loc[X.index, "is_call"],
)
print(f"Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")
print(f"Calls — Train : {X_train['is_call'].mean():.2%}  | Test : {X_test['is_call'].mean():.2%}")

# ─────────────────────────────────────────────
# ÉTAPE 5 — Baseline RF
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Baseline Random Forest")
print("=" * 60)

rf_base = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_b  = mean_absolute_error(y_test, y_pred_base)
r2_b   = r2_score(y_test, y_pred_base)
print(f"  RMSE : {rmse_b:.4f} $  |  MAE : {mae_b:.4f} $  |  R² : {r2_b:.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 6 — Cross-validation 5-Fold
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Cross-validation 5-Fold")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rmse = cross_val_score(rf_base, X_train, y_train,
                          cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_r2   = cross_val_score(rf_base, X_train, y_train,
                          cv=kf, scoring="r2", n_jobs=-1)
print(f"  CV RMSE : {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"  CV R²   : {cv_r2.mean():.6f} ± {cv_r2.std():.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 7 — GridSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — GridSearchCV")
print("=" * 60)

param_grid = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
    "max_features"     : ["sqrt", 0.5],
}
grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, verbose=1, refit=True,
)
grid_rf.fit(X_train, y_train)
print(f"\nMeilleurs paramètres : {grid_rf.best_params_}")
print(f"Meilleur CV RMSE    : {-grid_rf.best_score_:.4f}")

# ─────────────────────────────────────────────
# ÉTAPE 8 — Évaluation finale RF
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 8 — Évaluation finale RF (test set)")
print("=" * 60)

rf_best   = grid_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)
me_rf   = float((y_pred_rf - y_test).mean())
mape_rf = float(np.mean(np.abs((y_test - y_pred_rf) / (y_test + 1e-8))) * 100)

print(f"  RMSE  : {rmse_rf:.4f} $  (baseline : {rmse_b:.4f})")
print(f"  MAE   : {mae_rf:.4f} $")
print(f"  R²    : {r2_rf:.6f}")
print(f"  Biais : {me_rf:+.4f} $")
print(f"  MAPE  : {mape_rf:.2f}%")

for opt_type, label in [(1,"Calls"), (0,"Puts")]:
    mask = X_test["is_call"] == opt_type
    r = np.sqrt(mean_squared_error(y_test[mask], y_pred_rf[mask]))
    r2 = r2_score(y_test[mask], y_pred_rf[mask])
    print(f"  {label} — RMSE : {r:.4f}  |  R² : {r2:.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 9 — Feature importance
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 9 — Feature Importance")
print("=" * 60)

mdi = pd.Series(rf_best.feature_importances_, index=FEATURES).sort_values(ascending=False)
perm = permutation_importance(rf_best, X_test, y_test, n_repeats=20, random_state=SEED, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)

print("\nMDI :")
print(mdi.round(4))
print("\nPermutation :")
print(perm_s.round(4))

# ─────────────────────────────────────────────
# ÉTAPE 10 — Visualisations
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 10 — Visualisations")
print("=" * 60)

residuals = y_pred_rf - y_test
rel_err   = np.abs(residuals) / (y_test + 1e-8) * 100

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Random Forest — Option Pricing (target : mid_price, AAPL)", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
ax1.hist(y, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
ax1.set_title("Distribution de mid_price")
ax1.set_xlabel("mid_price ($)"); ax1.set_ylabel("Fréquence")

ax2 = fig.add_subplot(gs[0,1])
cm  = X_test["is_call"] == 1
ax2.scatter(y_test[cm],  y_pred_rf[cm],  alpha=0.4, s=14, color="#4C72B0", label="Calls")
ax2.scatter(y_test[~cm], y_pred_rf[~cm], alpha=0.4, s=14, color="#DD8452", label="Puts")
lim = [0, y_test.max()*1.05]
ax2.plot(lim, lim, "k--", lw=1.2)
ax2.set_title(f"Prédit vs Réel  (R²={r2_rf:.4f})")
ax2.set_xlabel("mid_price réel ($)"); ax2.set_ylabel("mid_price prédit ($)"); ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(y_pred_rf, residuals, alpha=0.3, s=12, color="#55A868")
ax3.axhline(0, color="red", lw=1.5, linestyle="--")
ax3.set_title("Résidus vs Prédit")
ax3.set_xlabel("Prédit ($)"); ax3.set_ylabel("Résidu ($)")

ax4 = fig.add_subplot(gs[1,0])
ax4.hist(residuals, bins=60, color="#C44E52", edgecolor="white", alpha=0.85)
ax4.axvline(0, color="black", lw=1.5, linestyle="--")
ax4.set_title("Distribution des résidus")
ax4.set_xlabel("Résidu ($)")

ax5 = fig.add_subplot(gs[1,1])
mdi.sort_values().plot.barh(ax=ax5, color="#4C72B0", alpha=0.85)
ax5.set_title("MDI Feature Importance"); ax5.set_xlabel("Importance")

ax6 = fig.add_subplot(gs[1,2])
perm_s.sort_values().plot.barh(ax=ax6, color="#DD8452", alpha=0.85)
ax6.set_title("Permutation Importance"); ax6.set_xlabel("Importance moyenne")

ax7 = fig.add_subplot(gs[2,0])
log_m = X_test["log_moneyness"]
ax7.scatter(log_m, rel_err, alpha=0.3, s=12, color="#8172B2")
ax7.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--",
            label=f"Médiane {rel_err.median():.1f}%")
ax7.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax7.set_title("Erreur relative vs Log-Moneyness")
ax7.set_xlabel("log(S/K)"); ax7.set_ylabel("Err. rel. (%)"); ax7.legend(fontsize=8)

ax8 = fig.add_subplot(gs[2,1])
ax8.scatter(X_test["time_to_maturity"], rel_err, alpha=0.3, s=12, color="#64B5CD")
ax8.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--")
ax8.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax8.set_title("Erreur relative vs Maturité")
ax8.set_xlabel("T (années)"); ax8.set_ylabel("Err. rel. (%)")

ax9 = fig.add_subplot(gs[2,2])
bars = ax9.bar(["Baseline RF","RF Optimisé"], [rmse_b, rmse_rf],
               color=["#4C72B0","#DD8452"], alpha=0.85, width=0.4)
ax9.set_title("RMSE Baseline vs Optimisé"); ax9.set_ylabel("RMSE ($)")
for bar, val in zip(bars, [rmse_b, rmse_rf]):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.3f}", ha="center", fontsize=9)

plt.savefig("rf_midprice_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphiques → rf_midprice_results.png")

import os

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(rf_best, os.path.join(OUTPUT_DIR, "rf_midprice.pkl"))
print("Modèle → rf_midprice.pkl")

# Sauvegarder split pour réutilisation dans les scripts suivants
split_info = {
    "X_train_idx": X_train.index.tolist(),
    "X_test_idx" : X_test.index.tolist(),
}
import json
with open(os.path.join(OUTPUT_DIR, "split_indices.json"), "w") as f:
    json.dump(split_info, f)

print(f"\nRésultats RF (mid_price) :")
print(f"  RMSE={rmse_rf:.4f}  MAE={mae_rf:.4f}  R²={r2_rf:.6f}  Biais={me_rf:+.4f}  MAPE={mape_rf:.2f}%")
