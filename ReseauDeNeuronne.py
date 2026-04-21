"""
=============================================================================
OPTION PRICING — TARGET : MID_PRICE
SCRIPT 3/3 — RÉSEAU DE NEURONES (MLP) + TABLEAU DE SYNTHÈSE FINAL
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, json, joblib
warnings.filterwarnings("ignore")

from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED   = 42
R_FREE = 0.043
np.random.seed(SEED)

# ─────────────────────────────────────────────
# ÉTAPE 0 — Chargement & feature engineering
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 0 — Chargement & feature engineering (mid_price)")
print("=" * 60)

df = pd.read_csv(r"C:\Users\DELL\Downloads\options_dataset.csv")
df["mid_price"] = np.where(df["bid"].isna(), df["ask"]/2, (df["bid"]+df["ask"])/2)
df = df[df["mid_price"] > 0].reset_index(drop=True)

data = df.copy()
data["is_call"]         = (data["type"]=="call").astype(int)
data["volume"]          = data["volume"].fillna(0)
data["intrinsic_value"] = np.where(
    data["is_call"]==1,
    np.maximum(data["spot"]-data["strike"],0),
    np.maximum(data["strike"]-data["spot"],0),
)
data["itm"]             = (data["intrinsic_value"]>0).astype(int)
data["log_moneyness"]   = np.log(data["spot"]/data["strike"])
data["spread"]          = data["ask"] - data["bid"].fillna(data["ask"])
data["vol_sqrt_t"]      = data["impliedVolatility"]*np.sqrt(data["time_to_maturity"])
data["log_volume"]      = np.log1p(data["volume"])

FEATURES = [
    "strike","spot","time_to_maturity","impliedVolatility",
    "is_call","log_moneyness","intrinsic_value","itm",
    "vol_sqrt_t","log_volume","spread",
]
TARGET = "mid_price"

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
    stratify=data.loc[X.index,"is_call"],
)
print(f"Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# ÉTAPE 1 — Normalisation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 1 — Normalisation StandardScaler")
print("=" * 60)
print("""
StandardScaler obligatoire pour le MLP :
  • Adam optimise par descente de gradient — sensible aux échelles
  • Poids initialisés pour des données ≈ N(0,1)
  • Fit sur X_train uniquement → pas de data leakage
  • Intégré dans un Pipeline sklearn pour la CV
""")

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ─────────────────────────────────────────────
# ÉTAPE 2 — MLP Baseline
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — MLP Baseline  (128, 64)  ReLU  Adam  early_stopping")
print("=" * 60)

mlp_base = MLPRegressor(
    hidden_layer_sizes=(128,64), activation="relu", solver="adam",
    alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=500,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
    random_state=SEED,
)
mlp_base.fit(X_train_sc, y_train)
y_pred_base_mlp = mlp_base.predict(X_test_sc)

rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_base_mlp))
mae_b  = mean_absolute_error(y_test, y_pred_base_mlp)
r2_b   = r2_score(y_test, y_pred_base_mlp)
print(f"  Itérations : {mlp_base.n_iter_}  |  RMSE={rmse_b:.4f}  MAE={mae_b:.4f}  R²={r2_b:.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 3 — Cross-validation 5-Fold (Pipeline)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Cross-validation 5-Fold (Pipeline scaler + MLP)")
print("=" * 60)

pipe_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(128,64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=500,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=SEED,
    )),
])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rmse = cross_val_score(pipe_mlp, X_train, y_train, cv=kf,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
cv_r2   = cross_val_score(pipe_mlp, X_train, y_train, cv=kf,
                          scoring="r2", n_jobs=-1)
print(f"\n  CV RMSE : {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
print(f"  CV R²   : {cv_r2.mean():.6f} ± {cv_r2.std():.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 4 — GridSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — GridSearchCV (30 combinaisons)")
print("=" * 60)

param_grid_mlp = {
    "mlp__hidden_layer_sizes": [(64,32),(128,64),(256,128),(128,64,32),(256,128,64)],
    "mlp__alpha"             : [1e-5, 1e-4, 1e-3],
    "mlp__learning_rate_init": [1e-3, 5e-4],
}
grid_mlp = GridSearchCV(
    pipe_mlp, param_grid=param_grid_mlp,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, verbose=1, refit=True,
)
grid_mlp.fit(X_train, y_train)
print(f"\nMeilleurs paramètres : {grid_mlp.best_params_}")
print(f"Meilleur CV RMSE    : {-grid_mlp.best_score_:.4f}")

# ─────────────────────────────────────────────
# ÉTAPE 5 — Évaluation MLP optimisé
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Évaluation MLP optimisé (test set)")
print("=" * 60)

mlp_best   = grid_mlp.best_estimator_
y_pred_mlp = mlp_best.predict(X_test)

rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_mlp  = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp   = r2_score(y_test, y_pred_mlp)
me_mlp   = float((y_pred_mlp - y_test).mean())
mape_mlp = float(np.mean(np.abs((y_test-y_pred_mlp)/(y_test+1e-8)))*100)

print(f"  RMSE  : {rmse_mlp:.4f} $  (baseline MLP : {rmse_b:.4f})")
print(f"  MAE   : {mae_mlp:.4f} $")
print(f"  R²    : {r2_mlp:.6f}")
print(f"  Biais : {me_mlp:+.4f} $")
print(f"  MAPE  : {mape_mlp:.2f}%")
for t,l in [(1,"Calls"),(0,"Puts")]:
    mask = X_test["is_call"]==t
    r=np.sqrt(mean_squared_error(y_test[mask],y_pred_mlp[mask]))
    r2=r2_score(y_test[mask],y_pred_mlp[mask])
    print(f"  {l} — RMSE={r:.4f}  R²={r2:.6f}")

# ─────────────────────────────────────────────
# ÉTAPE 6 — Black-Scholes & RF sur même test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Recalcul BS et RF sur le même test set")
print("=" * 60)

def black_scholes(S,K,T,r,sigma,option_type="call"):
    T=np.where(T<=0,1e-10,T); sigma=np.where(sigma<=0,1e-10,sigma)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    call=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    put =K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return np.where(option_type=="call",call,put)

y_pred_bs = black_scholes(
    X_test["spot"].values, X_test["strike"].values,
    X_test["time_to_maturity"].values, R_FREE,
    X_test["impliedVolatility"].values,
    data.loc[X_test.index,"type"].values,
)

import os
import joblib

BASE_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

rf_best = joblib.load(os.path.join(OUTPUT_DIR, "rf_midprice.pkl"))
y_pred_rf = rf_best.predict(X_test)

def metrics(y_true,y_pred):
    return dict(
        RMSE=np.sqrt(mean_squared_error(y_true,y_pred)),
        MAE =mean_absolute_error(y_true,y_pred),
        R2  =r2_score(y_true,y_pred),
        ME  =float((y_pred-y_true).mean()),
        MAPE=float(np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100),
    )

m_bs  = metrics(y_test, y_pred_bs)
m_rf  = metrics(y_test, y_pred_rf)
m_mlp = metrics(y_test, y_pred_mlp)

for name,m in [("BS",m_bs),("RF",m_rf),("MLP",m_mlp)]:
    print(f"  {name} | RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"R²={m['R2']:.6f}  Biais={m['ME']:+.4f}  MAPE={m['MAPE']:.2f}%")

# ─────────────────────────────────────────────
# ÉTAPE 7 — Courbe de convergence
# ─────────────────────────────────────────────
best_mlp_raw = grid_mlp.best_estimator_.named_steps["mlp"]
loss_curve   = best_mlp_raw.loss_curve_
val_curve    = best_mlp_raw.validation_scores_
print(f"\nMLP convergence : {len(loss_curve)} itérations  |  Best val R²={max(val_curve):.4f}")

# ─────────────────────────────────────────────
# ÉTAPE 8 — Visualisations complètes
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 8 — Visualisations complètes")
print("=" * 60)

res_bs  = y_pred_bs  - y_test.values
res_rf  = y_pred_rf  - y_test.values
res_mlp = y_pred_mlp - y_test.values
COLORS  = {"BS":"#9467BD","RF":"#2CA02C","MLP":"#D62728"}

fig = plt.figure(figsize=(22, 24))
fig.suptitle("MLP + Comparaison Complète — target : mid_price (AAPL)",
             fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# 1. Loss MLP
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(loss_curve, color=COLORS["MLP"], lw=2, label="Train loss")
ax1.set_title("Convergence MLP (loss)"); ax1.set_xlabel("Itération"); ax1.set_ylabel("MSE"); ax1.set_yscale("log"); ax1.legend(fontsize=9)

# 2. R² validation interne
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(val_curve, color="#FF7F0E", lw=2, label="Val R²")
ax2.axhline(max(val_curve), color="red", lw=1, linestyle="--", label=f"Best={max(val_curve):.4f}")
ax2.set_title("R² validation interne"); ax2.set_xlabel("Itération"); ax2.legend(fontsize=9)

# 3. MLP prédit vs réel
ax3 = fig.add_subplot(gs[0,2])
cm = X_test["is_call"]==1
ax3.scatter(y_test[cm],  y_pred_mlp[cm],  alpha=0.4, s=12, color="#4C72B0", label="Calls")
ax3.scatter(y_test[~cm], y_pred_mlp[~cm], alpha=0.4, s=12, color="#DD8452", label="Puts")
lims=[0, y_test.max()*1.05]; ax3.plot(lims,lims,"k--",lw=1.2)
ax3.set_title(f"MLP vs mid_price\nR²={m_mlp['R2']:.4f}  RMSE={m_mlp['RMSE']:.2f}$")
ax3.set_xlabel("mid_price réel ($)"); ax3.set_ylabel("Prédit ($)"); ax3.legend(fontsize=8)

# 4-6. Prédit vs réel pour les 3 modèles
for idx,(label,y_pred_m,col) in enumerate([
    ("Black-Scholes",y_pred_bs, COLORS["BS"]),
    ("Random Forest",y_pred_rf, COLORS["RF"]),
    ("MLP",          y_pred_mlp,COLORS["MLP"]),
]):
    ax=fig.add_subplot(gs[1,idx])
    ax.scatter(y_test, y_pred_m, alpha=0.3, s=10, color=col)
    lims=[0,y_test.max()*1.05]; ax.plot(lims,lims,"k--",lw=1.2)
    m=metrics(y_test,y_pred_m)
    ax.set_title(f"{label}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}$")
    ax.set_xlabel("mid_price réel ($)"); ax.set_ylabel("Prédit ($)")

# 7. Distribution des résidus
ax7=fig.add_subplot(gs[2,0])
for res,label,col in [(res_bs,"BS",COLORS["BS"]),(res_rf,"RF",COLORS["RF"]),(res_mlp,"MLP",COLORS["MLP"])]:
    ax7.hist(res,bins=60,alpha=0.5,color=col,label=label,density=True)
ax7.axvline(0,color="black",lw=1.5,linestyle="--")
ax7.set_title("Distribution résidus vs mid_price"); ax7.set_xlabel("Résidu ($)"); ax7.legend(fontsize=8)

# 8. RMSE par bucket de moneyness
ax8=fig.add_subplot(gs[2,1])
df_test=data.loc[X_test.index].copy()
df_test["pred_bs"]=y_pred_bs; df_test["pred_rf"]=y_pred_rf; df_test["pred_mlp"]=y_pred_mlp
df_test["moneyness_bucket"]=pd.cut(
    np.log(df_test["spot"]/df_test["strike"]),
    bins=[-np.inf,-0.3,-0.1,0.0,0.1,0.3,np.inf],
    labels=["Deep OTM","OTM","ATM-","ATM+","ITM","Deep ITM"],
)
rmse_bucket={}
for label,col_p in [("BS","pred_bs"),("RF","pred_rf"),("MLP","pred_mlp")]:
    rmse_bucket[label]=df_test.groupby("moneyness_bucket",observed=True).apply(
        lambda g: np.sqrt(mean_squared_error(g["mid_price"],g[col_p]))
    )
x=np.arange(6); w=0.25
for i,(label,col) in enumerate([("BS",COLORS["BS"]),("RF",COLORS["RF"]),("MLP",COLORS["MLP"])]):
    ax8.bar(x+i*w,rmse_bucket[label].values,width=w,label=label,color=col,alpha=0.85)
ax8.set_xticks(x+w); ax8.set_xticklabels(rmse_bucket["BS"].index,rotation=30,ha="right",fontsize=8)
ax8.set_title("RMSE par moneyness (mid_price)"); ax8.set_ylabel("RMSE ($)"); ax8.legend(fontsize=8)

# 9. RMSE par maturité
ax9=fig.add_subplot(gs[2,2])
df_test["ttm_bucket"]=pd.cut(df_test["time_to_maturity"],
    bins=[0,0.1,0.25,0.5,1.0,np.inf],labels=["<1M","1-3M","3-6M","6-12M",">1Y"])
rmse_ttm={}
for label,col_p in [("BS","pred_bs"),("RF","pred_rf"),("MLP","pred_mlp")]:
    rmse_ttm[label]=df_test.groupby("ttm_bucket",observed=True).apply(
        lambda g: np.sqrt(mean_squared_error(g["mid_price"],g[col_p]))
    )
x2=np.arange(5)
for i,(label,col) in enumerate([("BS",COLORS["BS"]),("RF",COLORS["RF"]),("MLP",COLORS["MLP"])]):
    ax9.bar(x2+i*w,rmse_ttm[label].values,width=w,label=label,color=col,alpha=0.85)
ax9.set_xticks(x2+w); ax9.set_xticklabels(rmse_ttm["BS"].index,fontsize=9)
ax9.set_title("RMSE par maturité (mid_price)"); ax9.set_ylabel("RMSE ($)"); ax9.legend(fontsize=8)

# 10-11. Résidus vs log-moneyness pour RF et MLP
for idx,(label,y_pred_m,col) in enumerate([
    ("Random Forest",y_pred_rf,COLORS["RF"]),
    ("MLP",          y_pred_mlp,COLORS["MLP"]),
]):
    ax=fig.add_subplot(gs[3,idx])
    res=y_pred_m - y_test.values
    lm=np.log(X_test["spot"]/X_test["strike"])
    ax.scatter(lm,res,alpha=0.3,s=10,color=col)
    ax.axhline(0,color="red",lw=1.5,linestyle="--")
    ax.set_title(f"Résidus vs Log-Moneyness\n{label}")
    ax.set_xlabel("log(S/K)"); ax.set_ylabel("Résidu ($)")

# 12. TABLEAU DE SYNTHÈSE VISUEL
ax12=fig.add_subplot(gs[3,2])
ax12.axis("off")

col_labels=["Métrique","Black-Scholes","Random Forest","MLP"]
row_data=[
    ["RMSE ($)",  f"{m_bs['RMSE']:.3f}",  f"{m_rf['RMSE']:.3f}",  f"{m_mlp['RMSE']:.3f}"],
    ["MAE  ($)",  f"{m_bs['MAE']:.3f}",   f"{m_rf['MAE']:.3f}",   f"{m_mlp['MAE']:.3f}"],
    ["R²",        f"{m_bs['R2']:.5f}",    f"{m_rf['R2']:.5f}",    f"{m_mlp['R2']:.5f}"],
    ["Biais ($)", f"{m_bs['ME']:+.3f}",   f"{m_rf['ME']:+.3f}",   f"{m_mlp['ME']:+.3f}"],
    ["MAPE (%)",  f"{m_bs['MAPE']:.1f}",  f"{m_rf['MAPE']:.1f}",  f"{m_mlp['MAPE']:.1f}"],
]

table=ax12.table(cellText=row_data,colLabels=col_labels,cellLoc="center",loc="center",bbox=[0,0,1,1])
table.auto_set_font_size(False); table.set_fontsize(9)
for j in range(4):
    table[0,j].set_facecolor("#2C3E50"); table[0,j].set_text_props(color="white",fontweight="bold")

best_col={
    0: 1+[m_bs["RMSE"],m_rf["RMSE"],m_mlp["RMSE"]].index(min(m_bs["RMSE"],m_rf["RMSE"],m_mlp["RMSE"])),
    1: 1+[m_bs["MAE"], m_rf["MAE"], m_mlp["MAE"]].index(min(m_bs["MAE"], m_rf["MAE"], m_mlp["MAE"])),
    2: 1+[m_bs["R2"],  m_rf["R2"],  m_mlp["R2"]].index(max(m_bs["R2"],  m_rf["R2"],  m_mlp["R2"])),
    3: 1+[abs(m_bs["ME"]),abs(m_rf["ME"]),abs(m_mlp["ME"])].index(min(abs(m_bs["ME"]),abs(m_rf["ME"]),abs(m_mlp["ME"]))),
    4: 1+[m_bs["MAPE"],m_rf["MAPE"],m_mlp["MAPE"]].index(min(m_bs["MAPE"],m_rf["MAPE"],m_mlp["MAPE"])),
}
for ri,bc in best_col.items():
    table[ri+1,bc].set_facecolor("#2ECC71"); table[ri+1,bc].set_text_props(fontweight="bold")

ax12.set_title("Tableau de synthèse — mid_price (test set 20%)",fontsize=10,fontweight="bold",pad=8)

plt.savefig("mlp_midprice_comparison.png",dpi=150,bbox_inches="tight")
plt.show()
print("Graphiques → mlp_midprice_comparison.png")

# ─────────────────────────────────────────────
# TABLEAU TERMINAL FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("TABLEAU DE SYNTHÈSE FINAL — TARGET : MID_PRICE — Test Set (20%)")
print("=" * 65)

models={"Black-Scholes":m_bs,"Random Forest":m_rf,"MLP":m_mlp}
print(f"\n  {'Métrique':<18} {'Black-Scholes':>14} {'Random Forest':>14} {'MLP':>14}")
print("  "+"─"*62)
for metric in [("RMSE ($)","RMSE",False),("MAE  ($)","MAE",False),("R²","R2",True),
               ("Biais ($)","ME",None),("MAPE (%)","MAPE",False)]:
    label,key,higher_better=metric
    vals={m:models[m][key] for m in models}
    if key=="ME": best=min(models,key=lambda m:abs(models[m][key]))
    elif higher_better: best=max(models,key=lambda m:models[m][key])
    else: best=min(models,key=lambda m:models[m][key])
    fmt="{:+.4f}" if key=="ME" else ("{:.6f}" if key=="R2" else ("{:.2f}" if key=="MAPE" else "{:.4f}"))
    row=f"  {label:<18}"
    for m in models:
        mark=" ✓" if m==best else "  "
        row+=f" {fmt.format(vals[m]):>12}{mark}"
    print(row)
print("  "+"─"*62)

# Rang global
ranks={m:0 for m in models}
for key,hb in [("RMSE",False),("MAE",False),("R2",True),("MAPE",False)]:
    for rank,m in enumerate(sorted(models,key=lambda m:models[m][key],reverse=hb),1):
        ranks[m]+=rank
for rank,m in enumerate(sorted(models,key=lambda m:abs(models[m]["ME"])),1):
    ranks[m]+=rank

print(f"\n  Rang global (somme des rangs, moins = mieux) :")
for m,total in sorted(ranks.items(),key=lambda x:x[1]):
    print(f"    {m:<20} : rang total = {total}  (moy. {total/5:.1f})")

winner=min(ranks,key=lambda m:ranks[m])
print(f"\n  🏆 MEILLEUR MODÈLE (mid_price) : {winner}")
print(f"""
  Gains par rapport à Black-Scholes :
    RF  : RMSE  −{(1-m_rf['RMSE']/m_bs['RMSE'])*100:.0f}%   MAE −{(1-m_rf['MAE']/m_bs['MAE'])*100:.0f}%
    MLP : RMSE  −{(1-m_mlp['RMSE']/m_bs['RMSE'])*100:.0f}%   MAE −{(1-m_mlp['MAE']/m_bs['MAE'])*100:.0f}%

  Résumé des RMSE (test set, target = mid_price) :
    Black-Scholes  : {m_bs['RMSE']:.4f} $
    Random Forest  : {m_rf['RMSE']:.4f} $
    MLP            : {m_mlp['RMSE']:.4f} $
""")

joblib.dump(
    mlp_best,
    os.path.join(OUTPUT_DIR, "mlp_midprice.pkl")
)
print("Modèle MLP → mlp_midprice.pkl")
