"""
=============================================================================
OPTION PRICING — TARGET : MID_PRICE
SCRIPT 3/3 — XGBoost
=============================================================================
"""


"""
Pricing d'options avec XGBoost
-------------------------------
Ce script implémente un modèle de régression XGBoost pour prédire le prix moyen (mid_price)
des options à partir de caractéristiques telles que la monnaie (moneyness), le temps jusqu'à l'échéance,
le type d'option (call/put) et d'autres variables de marché.

Le pipeline comprend :
- Chargement et nettoyage des données
- Ingénierie de caractéristiques (log-moneyness, spread, valeur intrinsèque, etc.)
- Division entraînement / validation / test avec stratification par type d'option
- Optimisation des hyperparamètres par RandomizedSearchCV
- Entraînement final du modèle avec early stopping
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb

# Supprimer les avertissements superflus pour une sortie plus propre
warnings.filterwarnings('ignore')

# =============================================================================
# 2. CHARGEMENT DES DONNÉES
# =============================================================================
# Charger le jeu de données des options (adapter le chemin si nécessaire)
data = pd.read_csv(r"C:\Users\ERAZER\Desktop\options_dataset.csv")

# =============================================================================
# 3. NETTOYAGE ET TRAITEMENT DES VALEURS MANQUANTES
# =============================================================================
print("Valeurs manquantes par colonne :")
print(data.isnull().sum())

# Remplacer les valeurs manquantes de 'bid' et 'volume' par leur médiane
data['bid'] = data['bid'].fillna(data['bid'].median())
data['volume'] = data['volume'].fillna(data['volume'].median())

print("\nValeurs manquantes après imputation :")
print(data.isnull().sum())

# =============================================================================
# 4. CRÉATION DE CARACTÉRISTIQUES (FEATURE ENGINEERING)
# =============================================================================
# Prix moyen (cible)
data["mid_price"] = (data["bid"] + data["ask"]) / 2

# Spread bid-ask
data["spread"] = data["ask"] - data["bid"]

# Log-monnaie : ln(spot / strike)
data["log_moneyness"] = np.log(data["spot"] / data["strike"])

# Log-volume pour gérer l'échelle
data["log_volume"] = np.log(data["volume"])

# Indicatrice binaire pour les calls (1 pour call, 0 pour put)
data["is_call"] = (data["type"] == "call").astype(int)

# Valeur intrinsèque (max(spot - strike, 0) pour les calls ; max(strike - spot, 0) pour les puts)
data["intrinsic_value"] = np.where(
    data["type"] == "call",
    np.maximum(data["spot"] - data["strike"], 0),
    np.maximum(data["strike"] - data["spot"], 0)
)

# Supprimer les lignes où mid_price = 0 (options théoriquement sans valeur)
data = data[data["mid_price"] > 0].reset_index(drop=True)
print(f"\nAprès suppression de mid_price=0 : {data.shape[0]} lignes")

# =============================================================================
# 5. SÉLECTION DES CARACTÉRISTIQUES ET DE LA CIBLE
# =============================================================================
feature_cols = [
    'log_moneyness',      # Log du ratio spot / strike
    'time_to_maturity',   # Temps jusqu'à l'échéance (en années)
    'is_call',            # 1 pour call, 0 pour put
    'spot',               # Prix spot du sous-jacent
    'strike',             # Prix d'exercice
    'log_volume',         # Log du volume
    'intrinsic_value'     # Valeur intrinsèque
]

target_col = 'mid_price'

X = data[feature_cols].copy()
y = data[target_col].copy()

# =============================================================================
# 6. DIVISION ENTRAÎNEMENT / VALIDATION / TEST
# =============================================================================
# Première division : 80% entraînement, 20% test (stratification par type d'option)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=data.loc[X.index, "is_call"]
)

# Deuxième division : à partir des 80% d'entraînement, on prend 80% pour l'entraînement
# et 20% pour la validation (soit 64% du total pour l'entraînement final, 16% pour la validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42,
    stratify=data.loc[X_train.index, "is_call"]
)

print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape}")
print(f"Taille de l'ensemble de validation : {X_val.shape}")
print(f"Taille de l'ensemble de test : {X_test.shape}")

# =============================================================================
# 7. OPTIMISATION DES HYPERPARAMÈTRES AVEC RANDOMIZED SEARCH
# =============================================================================
# Modèle XGBoost de base
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Grille des hyperparamètres
param_grid = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]   # Réduction minimale de la loss pour une division supplémentaire
}

# Recherche aléatoire avec validation croisée 5-folds
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=100,          # Nombre de combinaisons à tester
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\nOptimisation des hyperparamètres XGBoost en cours...")
random_search.fit(X_train, y_train)

print("\n--- Meilleurs hyperparamètres XGBoost ---")
print(random_search.best_params_)

best_xgb = random_search.best_estimator_

# =============================================================================
# 8. ENTRAÎNEMENT FINAL AVEC EARLY STOPPING
# =============================================================================
# Combiner les ensembles d'entraînement et de validation pour l'entraînement final
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

# Récupérer les meilleurs paramètres et ajouter ceux nécessaires à l'early stopping
best_params = random_search.best_params_.copy()
best_params['early_stopping_rounds'] = 50
best_params['eval_metric'] = 'rmse'
best_params['verbose'] = False

final_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)

print("\nEntraînement du modèle final avec early stopping...")
final_xgb.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# =============================================================================
# 9. ÉVALUATION SUR L'ENSEMBLE DE TEST
# =============================================================================
y_pred = final_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Performance du modèle sur l'ensemble de test ---")
print(f"MAE  (Mean Absolute Error)      : {mae:.6f}")
print(f"MSE  (Mean Squared Error)       : {mse:.6f}")
print(f"RMSE (Root Mean Squared Error)  : {rmse:.6f}")
print(f"R²   (Coefficient of Determination) : {r2:.4f}")

# =============================================================================
# 10. IMPORTANCE DES CARACTÉRISTIQUES
# =============================================================================
print("\n" + "="*60)
print("IMPORTANCE DES CARACTÉRISTIQUES")
print("="*60)

# Importance MDI (Mean Decrease in Impurity)
mdi = pd.Series(final_xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Importance par permutation (plus robuste, indépendante du modèle)
perm = permutation_importance(final_xgb, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

print("\nImportance MDI :")
print(mdi.round(4))
print("\nImportance par permutation :")
print(perm_s.round(4))

# =============================================================================
# 11. VISUALISATIONS
# =============================================================================
print("\n" + "="*60)
print("VISUALISATIONS")
print("="*60)

# Préparer les résidus et l'erreur relative
residuals = y_pred - y_test
rel_err = np.abs(residuals) / (y_test + 1e-8) * 100

fig = plt.figure(figsize=(20, 16))
fig.suptitle("XGBoost — Pricing d'options (cible : mid_price)", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# (1) Distribution de mid_price
ax1 = fig.add_subplot(gs[0,0])
ax1.hist(y, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
ax1.set_title("Distribution de mid_price")
ax1.set_xlabel("mid_price ($)"); ax1.set_ylabel("Fréquence")

# (2) Prédit vs Réel
ax2 = fig.add_subplot(gs[0,1])
cm = X_test["is_call"] == 1
ax2.scatter(y_test[cm], y_pred[cm], alpha=0.4, s=14, color="#4C72B0", label="Calls")
ax2.scatter(y_test[~cm], y_pred[~cm], alpha=0.4, s=14, color="#DD8452", label="Puts")
lim = [0, y_test.max()*1.05]
ax2.plot(lim, lim, "k--", lw=1.2)
ax2.set_title(f"Prédit vs Réel (R²={r2:.4f})")
ax2.set_xlabel("mid_price réel ($)"); ax2.set_ylabel("mid_price prédit ($)"); ax2.legend(fontsize=8)

# (3) Résidus vs Prédit
ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(y_pred, residuals, alpha=0.3, s=12, color="#55A868")
ax3.axhline(0, color="red", lw=1.5, linestyle="--")
ax3.set_title("Résidus vs Prédit")
ax3.set_xlabel("Prédit ($)"); ax3.set_ylabel("Résidu ($)")

# (4) Distribution des résidus
ax4 = fig.add_subplot(gs[1,0])
ax4.hist(residuals, bins=60, color="#C44E52", edgecolor="white", alpha=0.85)
ax4.axvline(0, color="black", lw=1.5, linestyle="--")
ax4.set_title("Distribution des résidus")
ax4.set_xlabel("Résidu ($)")

# (5) Importance MDI
ax5 = fig.add_subplot(gs[1,1])
mdi.sort_values().plot.barh(ax=ax5, color="#4C72B0", alpha=0.85)
ax5.set_title("Importance MDI"); ax5.set_xlabel("Importance")

# (6) Importance par permutation
ax6 = fig.add_subplot(gs[1,2])
perm_s.sort_values().plot.barh(ax=ax6, color="#DD8452", alpha=0.85)
ax6.set_title("Importance par permutation"); ax6.set_xlabel("Importance moyenne")

# (7) Erreur relative vs Log-Moneyness
ax7 = fig.add_subplot(gs[2,0])
log_m = X_test["log_moneyness"]
ax7.scatter(log_m, rel_err, alpha=0.3, s=12, color="#8172B2")
ax7.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--", label=f"Médiane {rel_err.median():.1f}%")
ax7.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax7.set_title("Erreur relative vs Log-Moneyness")
ax7.set_xlabel("log(S/K)"); ax7.set_ylabel("Erreur relative (%)"); ax7.legend(fontsize=8)

# (8) Erreur relative vs Maturité
ax8 = fig.add_subplot(gs[2,1])
ax8.scatter(X_test["time_to_maturity"], rel_err, alpha=0.3, s=12, color="#64B5CD")
ax8.axhline(rel_err.median(), color="red", lw=1.5, linestyle="--")
ax8.set_ylim(0, min(rel_err.quantile(0.99), 200))
ax8.set_title("Erreur relative vs Maturité")
ax8.set_xlabel("Temps jusqu'à échéance (années)"); ax8.set_ylabel("Erreur relative (%)")

# Sauvegarder la figure
plt.savefig("xgb_midprice_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphique sauvegardé sous 'xgb_midprice_results.png'")

print("\nScript terminé avec succès.")
