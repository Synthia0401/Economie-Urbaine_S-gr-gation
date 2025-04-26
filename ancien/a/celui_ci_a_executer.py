import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configuration des styles de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("talk")

# === CONFIGURATION À ADAPTER ===
# Remplacez par la colonne de consommation ENAF par habitant si disponible
COL_CONSO = 'pop20'       
# Coefficient d'artificialisation (taux 2009-2023)
COL_COEF  = 'artcom0923'


def load_data(chemin_au, chemin_csp):
    """
    Charge les données depuis les fichiers CSV avec gestion d'erreurs
    """
    try:
        df_au = pd.read_csv(chemin_au, sep=';', low_memory=False)
        df_csp = pd.read_csv(chemin_csp, sep=';', low_memory=False)
        print(f"Données chargées avec succès : {df_au.shape[0]} aires urbaines et {df_csp.shape[0]} lignes CSP")
        return df_au, df_csp
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None, None


def nettoyer_donnees_au(df_au):
    """
    Nettoie et structure les données des aires urbaines
    """
    df = df_au.copy()
    df['AAV'] = df['AAV2020'].astype(str).str.zfill(3)

    # Identifier le nom de l'aire
    libelle_candidates = [c for c in df.columns if c.upper() in ['LIBELLE', 'NOM', 'LIBAAV2020', 'LIBELLEAAV']]
    if libelle_candidates:
        name_col = libelle_candidates[0]
        df = df.rename(columns={name_col: 'nom_aire_urbaine'})
    else:
        df['nom_aire_urbaine'] = "Aire urbaine " + df['AAV']

    # Sélection et renommage
    df = df[['AAV', 'AAV2020', 'TAAV2017', 'NB_COM', 'nom_aire_urbaine']]
    df = df.rename(columns={
        'TAAV2017': 'population_totale',
        'NB_COM': 'nombre_communes'
    })
    df['population_totale'] = pd.to_numeric(df['population_totale'], errors='coerce')
    df['nombre_communes']   = pd.to_numeric(df['nombre_communes'],   errors='coerce')
    df = df.dropna(subset=['population_totale', 'AAV'])
    return df


def preparer_donnees_conso(df_csp):
    """
    Prépare les données de consommation et de coefficient pour l'analyse
    """
    df = df_csp.copy()
    df['AAV'] = df['aav2020'].astype(str).str.extract(r'^(\d{3})').fillna('000')
    df['conso_par_hab'] = pd.to_numeric(df[COL_CONSO], errors='coerce')
    df['coef_artif']    = pd.to_numeric(df[COL_COEF],  errors='coerce')
    df = df.dropna(subset=['conso_par_hab', 'coef_artif'])
    return df


def nettoyer_donnees_complete(df_csp):
    """
    Effectue un nettoyage complet des données avant l'analyse
    - Traite les valeurs manquantes et aberrantes
    - Filtre les données non pertinentes
    - Convertit et vérifie les types de données
    """
    print("Début du nettoyage approfondi des données...")
    df = df_csp.copy()
    
    # 1. Conversion des types de données
    colonnes_numeriques = [COL_CONSO, COL_COEF, 'pop20']
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Filtrage des lignes sans données pertinentes
    nb_avant = df.shape[0]
    df = df.dropna(subset=[COL_CONSO, COL_COEF])
    print(f"Lignes filtrées pour valeurs manquantes: {nb_avant - df.shape[0]}")
    
    # 3. Traitement des valeurs aberrantes (méthode IQR)
    for col in [COL_CONSO, COL_COEF]:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            filtre_outliers = ~((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)))
            nb_avant = df.shape[0]
            df = df[filtre_outliers]
            print(f"Outliers filtrés dans {col}: {nb_avant - df.shape[0]}")
    
    # 4. Filtrage des communes sans aire urbaine assignée
    nb_avant = df.shape[0]
    df = df[df['aav2020'].notna() & (df['aav2020'] != '') & (df['aav2020'] != '00000')]
    print(f"Communes sans aire urbaine filtrées: {nb_avant - df.shape[0]}")
    
    # 5. Vérification de la cohérence des données
    if 'pop20' in df.columns:
        nb_avant = df.shape[0]
        df = df[df['pop20'] > 0]  # Filtrage des communes sans population
        print(f"Communes sans population filtrées: {nb_avant - df.shape[0]}")
    
    print(f"Nettoyage terminé: {df.shape[0]} lignes conservées")
    return df


def agreger_conso(df):
    """
    Agrège Conso et Coef au niveau des AAV et calcule la corrélation
    """
    df_agg = (
        df.groupby('AAV')[['conso_par_hab', 'coef_artif']]
          .mean()
          .rename(columns={
              'conso_par_hab': 'avg_conso_par_hab',
              'coef_artif':    'avg_coef_artif'
          })
          .reset_index()
    )
    corr = df_agg['avg_conso_par_hab'].corr(df_agg['avg_coef_artif'])
    print(f"Corrélation Conso vs Coef Artif : {corr:.3f}")
    return df_agg


def fusionner_donnees(df_au, df_agg):
    """
    Fusionne les données AU et Conso/Coef, ajoute catégorie de taille
    """
    df = pd.merge(df_au, df_agg, on='AAV', how='inner')
    df['categorie_taille'] = pd.cut(
        df['population_totale'],
        bins=[0, 50000, 200000, 500000, float('inf')],
        labels=['Petite', 'Moyenne', 'Grande', 'Très grande']
    )
    print(f"Fusion réalisée : {df.shape[0]} aires urbaines")
    return df


def creer_visualisation(df):
    """
    Trace le nuage de points Conso ENAF par hab vs Coef Artif
    avec ligne de tendance et annotation du R²
    """
    plt.figure(figsize=(10, 6))
    # Scatter plot
    sns.scatterplot(
        data=df,
        x='avg_conso_par_hab',
        y='avg_coef_artif',
        size='population_totale',
        hue='categorie_taille',
        alpha=0.7,
        legend='brief'
    )
    # Ligne de tendance
    sns.regplot(
        data=df,
        x='avg_conso_par_hab',
        y='avg_coef_artif',
        scatter=False,
        line_kws={'linewidth':2, 'linestyle':'--', 'color':'red'}
    )
    # Calcul du R²
    x = df['avg_conso_par_hab']
    y = df['avg_coef_artif']
    # Ajustement linéaire
    coef = np.polyfit(x, y, 1)
    y_pred = np.polyval(coef, x)
    r2 = np.corrcoef(y, y_pred)[0,1]**2
    # Annotation R²
    plt.text(
        0.05, 0.95,
        f'$R^2 = {r2:.2f}$',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8)
    )
    # Échelle log si souhaitée
    plt.xscale('log')
    plt.title("Conso ENAF par hab vs Coef Artif par aire urbaine")
    plt.xlabel("Conso ENAF par hab (moyenne AU)")
    plt.ylabel("Coef Artif (moyenne AU)")
    plt.legend(title="Catégorie de taille", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    # Chemins des fichiers (adapter si besoin)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chemin_au  = os.path.join(base_dir, 'inputs', 'csv', 'AAV2020_au_01-01-2024.csv')
    chemin_csp = os.path.join(base_dir, 'inputs', 'csv', 'conso2009_2023_resultats_com.csv')

    df_au, df_csp = load_data(chemin_au, chemin_csp)
    if df_au is None or df_csp is None:
        return
    
    # Étape de nettoyage complet des données
    df_csp_nettoye = nettoyer_donnees_complete(df_csp)
    
    # Suite du traitement avec les données nettoyées
    df_au_clean     = nettoyer_donnees_au(df_au)
    df_conso_clean  = preparer_donnees_conso(df_csp_nettoye)
    df_conso_agg    = agreger_conso(df_conso_clean)
    df_final        = fusionner_donnees(df_au_clean, df_conso_agg)
    
    # Ajouter une analyse statistique descriptive avant la visualisation
    print("\nStatistiques descriptives des variables d'intérêt:")
    print(df_final[['avg_conso_par_hab', 'avg_coef_artif']].describe())
    
    creer_visualisation(df_final)


if __name__ == "__main__":

    main()
