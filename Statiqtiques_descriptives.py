# =========================== IMPORTS ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================== FONCTIONS ===========================
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



def clean_data(df):
    """Gère les valeurs manquantes et ajoute des features dérivées."""
    # Remplacer les valeurs manquantes par la médiane
    df['bid'] = df['bid'].fillna(df['bid'].median())
    df['volume'] = df['volume'].fillna(df['volume'].median())
    
    # Nouvelles variables
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["log_moneyness"] = np.log(df["spot"] / df["strike"])
    df["log_volume"] = np.log(df["volume"])
    df["intrinsic_value"] = np.where(
        df["type"] == "call",
        np.maximum(df["spot"] - df["strike"], 0),
        np.maximum(df["strike"] - df["spot"], 0)
    )
    return df

def plot_correlation_matrix(df):
    """Affiche la matrice de corrélation des variables numériques."""
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Matrice de corrélation", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_smile_by_expiration(df):
    """Trace le smile de volatilité implicite pour chaque échéance."""
    expirations = sorted(df["expiration"].unique())
    cols = 3
    rows = math.ceil(len(expirations) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    for i, exp in enumerate(expirations):
        ax = axes[i]
        subset = df[df["expiration"] == exp]
        calls = subset[subset["type"] == "call"]
        puts = subset[subset["type"] == "put"]
        ax.scatter(calls["strike"], calls["impliedVolatility"],
                   label="Calls", alpha=0.6, s=10)
        ax.scatter(puts["strike"], puts["impliedVolatility"],
                   label="Puts", alpha=0.6, s=10)
        ax.set_title(exp)
        ax.set_xlabel("Strike")
        ax.set_ylabel("Volatilité implicite")
        ax.grid(True)
        ax.legend()
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_iv_boxplot_by_expiration(df):
    """Boxplot de la volatilité implicite par échéance (Calls vs Puts)."""
    expirations = sorted(df["expiration"].unique())
    # Limiter à 20 pour lisibilité
    if len(expirations) > 20:
        expirations = expirations[:20]
    
    call_data, put_data, exp_labels = [], [], []
    for exp in expirations:
        subset = df[df["expiration"] == exp]
        call_iv = subset[subset["type"] == "call"]["impliedVolatility"].values * 100
        put_iv = subset[subset["type"] == "put"]["impliedVolatility"].values * 100
        if len(call_iv) > 0:
            call_data.append(call_iv)
            put_data.append(put_iv)
            exp_labels.append(exp[5:])  # format MM/DD
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Boxplots Calls
    bp1 = ax1.boxplot(call_data, labels=exp_labels, patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax1.set_title("CALLS - Volatilité implicite par échéance", fontweight='bold')
    ax1.set_xlabel("Échéance (MM/JJ)")
    ax1.set_ylabel("Volatilité implicite (%)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=df['impliedVolatility'].mean()*100, color='red', linestyle='--',
                label=f'Moyenne globale: {df["impliedVolatility"].mean()*100:.1f}%')
    ax1.legend()
    
    # Boxplots Puts
    bp2 = ax2.boxplot(put_data, labels=exp_labels, patch_artist=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_facecolor('salmon')
        patch.set_alpha(0.7)
    ax2.set_title("PUTS - Volatilité implicite par échéance", fontweight='bold')
    ax2.set_xlabel("Échéance (MM/JJ)")
    ax2.set_ylabel("Volatilité implicite (%)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=df['impliedVolatility'].mean()*100, color='red', linestyle='--',
                label=f'Moyenne globale: {df["impliedVolatility"].mean()*100:.1f}%')
    ax2.legend()
    
    plt.suptitle("Distribution de la volatilité implicite par échéance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_iv_moneyness_distribution(df):
    """Histogrammes de moneyness pour calls et puts, et volatilité par moneyness."""
    fig = plt.figure(figsize=(20, 16))
    
    # Distribution moneyness - Calls
    ax1 = plt.subplot(3, 3, 1)
    subset = df[df['type'] == 'call']
    ax1.hist(subset['moneyness'], bins=30, alpha=0.5, color='blue', label='CALLS')
    ax1.axvline(x=1, color='black', linestyle='-', label='ATM (moneyness=1)')
    ax1.set_xlabel('Moneyness (Spot/Strike)')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution de moneyness - CALLS', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution moneyness - Puts
    ax2 = plt.subplot(3, 3, 2)
    subset = df[df['type'] == 'put']
    ax2.hist(subset['moneyness'], bins=30, alpha=0.5, color='red', label='PUTS')
    ax2.axvline(x=1, color='black', linestyle='-', label='ATM (moneyness=1)')
    ax2.set_xlabel('Moneyness (Spot/Strike)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution de moneyness - PUTS', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Volatilité moyenne par intervalle de moneyness
    ax3 = plt.subplot(3, 3, 3)
    iv_by_moneyness = df.groupby(pd.cut(df['moneyness'], bins=20))['impliedVolatility'].mean()
    x_labels = [f"{round(interval.left,2)}-{round(interval.right,2)}" for interval in iv_by_moneyness.index]
    ax3.plot(range(len(iv_by_moneyness)), iv_by_moneyness.values*100, 'o-', color='darkgreen', linewidth=2)
    ax3.set_xticks(range(len(iv_by_moneyness)))
    ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_xlabel("Intervalle de moneyness")
    ax3.set_ylabel("Volatilité implicite moyenne (%)")
    ax3.set_title("Volatilité par niveau de moneyness", fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Premium vs Strike
    ax4 = plt.subplot(3, 3, 4)
    for opt_type, color in [('call', 'blue'), ('put', 'red')]:
        subset = df[df['type'] == opt_type].sort_values('strike')
        ax4.scatter(subset['strike'], subset['lastPrice'], alpha=0.6, color=color, label=f'{opt_type.upper()}s', s=30)
    ax4.set_xlabel("Strike ($)")
    ax4.set_ylabel("Prime de l'option ($)")
    ax4.set_title("Prime vs Strike", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_iv_maturity_correlation(df):
    """Heatmap de corrélation IV vs Time to Maturity par niveau de moneyness."""
    df_clean = df.copy()
    df_clean['iv_pct'] = df_clean['impliedVolatility'] * 100
    df_clean['maturity_days'] = df_clean['time_to_maturity'] * 365
    df_clean['moneyness_group'] = pd.cut(df_clean['moneyness'], bins=8)
    
    groups, correlations, counts = [], [], []
    for name, group in df_clean.groupby('moneyness_group'):
        if len(group) > 10:
            corr = group['maturity_days'].corr(group['iv_pct'])
            groups.append(f"{name.left:.2f}-{name.right:.2f}")
            correlations.append(corr)
            counts.append(len(group))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap 1D
    corr_array = np.array(correlations).reshape(-1, 1)
    im = ax1.imshow(corr_array, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_yticks(range(len(groups)))
    ax1.set_yticklabels(groups, fontsize=10)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Corrélation\nIV ↔ Maturité'])
    ax1.set_title("Corrélation IV vs Time to Maturity\npar niveau de moneyness", fontweight='bold')
    for i, corr in enumerate(correlations):
        color = 'white' if abs(corr) > 0.5 else 'black'
        ax1.text(0, i, f'{corr:.2f}', ha='center', va='center', color=color, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Coefficient de corrélation')
    
    # Scatter global
    scatter = ax2.scatter(df_clean['maturity_days'], df_clean['iv_pct'],
                          c=df_clean['moneyness'], cmap='viridis', alpha=0.4, s=15)
    ax2.set_xlabel("Time to Maturity (jours)")
    ax2.set_ylabel("Volatilité implicite (%)")
    ax2.set_title("IV vs Maturité (couleur = moneyness)", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Tendance linéaire
    z = np.polyfit(df_clean['time_to_maturity'], df_clean['iv_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, df_clean['time_to_maturity'].max(), 100)
    ax2.plot(x_line*365, p(x_line), 'r--', linewidth=2, label=f'Tendance: {z[0]:.1f} pts/an')
    ax2.legend()
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Moneyness (Spot/Strike)')
    plt.tight_layout()
    plt.show()


# =========================== EXÉCUTION DIRECTE ===========================
# 1. Chargement des données
file_url = "https://drive.google.com/file/d/12KymJi6lZMmPRmaLB33klfMXRDTrIYJ1/view?usp=sharing"
data = load_data_from_gdrive(file_url)file_url = "https://drive.google.com/file/d/12KymJi6lZMmPRmaLB33klfMXRDTrIYJ1/view?usp=sharing"
data = load_data_from_gdrive(file_url)

# 2. Nettoyage et feature engineering
data = clean_data(data)

# 3. Aperçu des données
print("\n=== Valeurs manquantes ===")
print(data.isnull().sum())
print("\n=== Statistiques descriptives ===")
print(data.describe())

# 4. Visualisations
print("\n=== Génération des graphiques ===")
plot_correlation_matrix(data)
plot_smile_by_expiration(data)
plot_iv_boxplot_by_expiration(data)
plot_iv_moneyness_distribution(data)
plot_iv_maturity_correlation(data)

print("\nAnalyse terminée. Les graphiques ont été affichés.")
