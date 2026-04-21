"""
Option Pricing with Random Forest
---------------------------------
Ce script implémente un modèle Random Forest pour prédire le prix moyen (mid_price)
des options à partir de caractéristiques telles que la monnaie (moneyness), le temps jusqu'à l'échéance,
le type d'option (call/put) et d'autres variables de marché.

Le pipeline comprend :
- Chargement et nettoyage des données
- Ingénierie de caractéristiques (log-moneyness, spread, valeur intrinsèque, etc.)
- Division entraînement / test avec stratification par type d'option
- Optimisation des hyperparamètres par RandomizedSearchCV
- Évaluation (MAE, MSE, RMSE, R²)
- Analyse d'importance des caractéristiques (MDI et permutation)
- Visualisations (résidus, prédit vs réel, distributions d'erreur)

"""

# =============================================================================
# 1. IMPORTATIONS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Supprimer les avertissements superflus
warnings.filterwarnings('ignore')

# Fixer les seeds pour la reproductibilité
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 2. CHARGEMENT DES DONNÉES (ADAPTER LE CHEMIN)
# =============================================================================
data = pd.read_csv(r"C:\Users\ERAZER\Desktop\options_dataset.csv")

# =============================================================================
# 3. NETTOYAGE ET TRAITEMENT DES VALEURS MANQUANTES
# =============================================================================
print("Valeurs manquantes par colonne :")
print(data.isnull().sum())

data['bid'] = data['bid'].fillna(data['bid'].median())
data['volume'] = data['volume'].fillna(data['volume'].median())

print("\nValeurs manquantes après imputation :")
print(data.isnull().sum())

# =============================================================================
# 4. CRÉATION DE CARACTÉRISTIQUES (FEATURE ENGINEERING)
# =============================================================================
data["mid_price"] = (data["bid"] + data["ask"]) / 2
data["spread"] = data["ask"] - data["bid"]
data["log_moneyness"] = np.log(data["spot"] / data["strike"])
data["log_volume"] = np.log(data["volume"])
data["is_call"] = (data["type"] == "call").astype(int)
data["intrinsic_value"] = np.where(
    data["type"] == "call",
    np.maximum(data["spot"] - data["strike"], 0),
    np.maximum(data["strike"] - data["spot"], 0)
)

# Supprimer les lignes où mid_price = 0
data = data[data["mid_price"] > 0].reset_index(drop=True)
print(f"\nAprès suppression mid_price=0 : {data.shape[0]} lignes")

# =============================================================================
# 5. SÉLECTION DES CARACTÉRISTIQUES ET DE LA CIBLE
# =============================================================================
feature_cols = [
    'log_moneyness',
    'time_to_maturity',
    'is_call',
    'spot',
    'strike',
    'log_volume',
    'intrinsic_value'
]
target_col = 'mid_price'

X = data[feature_cols].copy()
y = data[target_col].copy()

# =============================================================================
# 6. DIVISION ENTRAÎNEMENT / TEST
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE,
    stratify=data.loc[X.index, "is_call"]
)

print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape}")
print(f"Taille de l'ensemble de test : {X_test.shape}")

# =============================================================================
# 7. OPTIMISATION DES HYPERPARAMÈTRES
# =============================================================================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None]
}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nOptimisation des hyperparamètres Random Forest en cours...")
random_search.fit(X_train, y_train)

print("\n--- Meilleurs hyperparamètres ---")
print(random_search.best_params_)

best_rf = random_search.best_estimator_

# =============================================================================
# 8. ÉVALUATION SUR L'ENSEMBLE DE TEST
# =============================================================================
y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Performance du modèle sur l'ensemble de test ---")
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.4f}")

# =============================================================================
# 9. IMPORTANCE DES CARACTÉRISTIQUES
# =============================================================================
print("\n" + "="*60)
print("IMPORTANCE DES CARACTÉRISTIQUES")
print("="*60)

mdi = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
perm = permutation_importance(best_rf, X_test, y_test, n_repeats=20,
                              random_state=RANDOM_STATE, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

print("\nImportance MDI :")
print(mdi.round(4))
print("\nImportance par permutation :")
print(perm_s.round(4))

# =============================================================================
# 10. VISUALISATIONS
# =============================================================================
print("\n" + "="*60)
print("VISUALISATIONS")
print("="*60)

residuals = y_pred - y_test
rel_err = np.abs(residuals) / (y_test + 1e-8) * 100

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Random Forest — Pricing d'options (cible : mid_price)", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
ax1.hist(y, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
ax1.set_title("Distribution de mid_price")
ax1.set_xlabel("mid_price ($)"); ax1.set_ylabel("Fréquence")

ax2 = fig.add_subplot(gs[0,1])
cm = X_test["is_call"] == 1
ax2.scatter(y_test[cm], y_pred[cm], alpha=0.4, s=14, color="#4C72B0", label="Calls")
ax2.scatter(y_test[~cm], y_pred[~cm], alpha=0.4, s=14, color="#DD8452", label="Puts")
lim = [0, y_test.max()*1.05]
ax2.plot(lim, lim, "k--", lw=1.2)
ax2.set_title(f"Prédit vs Réel (R²={r2:.4f})")
ax2.set_xlabel("mid_price réel ($)"); ax2.set_ylabel("mid_price prédit ($)"); ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(y_pred, residuals, alpha=0.3, s=12, color="#55A868")
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
ax5.set_title("Importance MDI"); ax5.set_xlabel("Importance")

ax6 = fig.add_subplot(gs[1,2])
perm_s.sort_values().plot.barh(ax=ax6, color="#DD8452", alpha=0.85)
ax6.set_title("Importance par permutation"); ax6.set_xlabel("Importance moyenne")

ax7 = fig.add_subplot(gs[2,0])
log_m = X_test["log_moneyness"]
ax7.scatter(log_m, rel_err, alpha=0.3, s=12, color="#8172B2")
ax7.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--", label=f"Médiane {rel_err.median():.1f}%")
ax7.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax7.set_title("Erreur relative vs Log-Moneyness")
ax7.set_xlabel("log(S/K)"); ax7.set_ylabel("Erreur relative (%)"); ax7.legend(fontsize=8)

ax8 = fig.add_subplot(gs[2,1])
ax8.scatter(X_test["time_to_maturity"], rel_err, alpha=0.3, s=12, color="#64B5CD")
ax8.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--")
ax8.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax8.set_title("Erreur relative vs Maturité")
ax8.set_xlabel("Temps jusqu'à échéance (années)"); ax8.set_ylabel("Erreur relative (%)")

plt.savefig("rf_midprice_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphique sauvegardé sous 'rf_midprice_results.png'")

print("\nScript terminé avec succès.")
