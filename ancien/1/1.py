import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Étape 1 : Chargement des données
df_au = pd.read_csv('../inputs/csv/AAV2020_au_01-01-2024.csv', sep=';', low_memory=False)
df_csp = pd.read_csv('../inputs/csv/conso2009_2023_resultats_com.csv', sep=';', low_memory=False)

print(f"Chargement réussi - AU: {df_au.shape[0]} lignes, CSP: {df_csp.shape[0]} lignes")

# Étape 2 : Extraction du code AAV à partir de la colonne aav2020
# La colonne aav2020 contient des valeurs comme '524 - Châtillon-sur-Chalaronne'
# Nous devons extraire uniquement le code numérique
df_csp['AAV'] = df_csp['aav2020'].astype(str).str.extract(r'^(\d+)')[0].fillna('000')
df_csp['AAV'] = df_csp['AAV'].str.zfill(3)  # Assurer un format à 3 chiffres

# Standardiser aussi la colonne AAV dans df_au
df_au['AAV'] = df_au['AAV2020'].astype(str).str.zfill(3)

# Étape 3 : Identification des colonnes pour cadres et ouvriers
# Pour améliorer l'analyse, on va essayer d'utiliser plusieurs colonnes
# Colonnes potentielles pour les cadres (activités tertiaires supérieures)
print("\nRecherche des colonnes pertinentes pour l'analyse CSP...")

# Recherche de colonnes pertinentes pour l'analyse des cadres et ouvriers
# Les fichiers INSEE peuvent contenir différentes nomenclatures selon les années
cadres_keywords = ['cadres', 'professions intellectuelles supérieures', 'act']
ouvriers_keywords = ['ouvriers', 'agriculteurs', 'hab']

# Dictionnaire pour stocker les colonnes trouvées
colonnes_csp = {
    'cadres': [],
    'ouvriers': []
}

# Parcourir toutes les colonnes pour trouver celles qui correspondent à nos critères
for col in df_csp.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in cadres_keywords):
        colonnes_csp['cadres'].append(col)
    elif any(keyword in col_lower for keyword in ouvriers_keywords):
        colonnes_csp['ouvriers'].append(col)

# Afficher les colonnes trouvées
print(f"Colonnes potentielles pour cadres: {colonnes_csp['cadres'][:5]}...")
print(f"Colonnes potentielles pour ouvriers: {colonnes_csp['ouvriers'][:5]}...")

# Étape 4 : Sélection des dernières colonnes disponibles (les plus à jour)
col_cadres = 'art22act23'  # On choisit la dernière colonne d'activité
col_ouvriers = 'art22hab23'  # On choisit la dernière colonne d'habitat

print(f"Colonne choisie pour cadres: {col_cadres}")
print(f"Colonne choisie pour ouvriers: {col_ouvriers}")

# Inspection des valeurs pour s'assurer qu'elles sont numériques
df_csp[col_cadres] = pd.to_numeric(df_csp[col_cadres], errors='coerce')
df_csp[col_ouvriers] = pd.to_numeric(df_csp[col_ouvriers], errors='coerce')

print(f"\nStatistiques pour la colonne {col_cadres}:")
print(f"Min: {df_csp[col_cadres].min()}, Max: {df_csp[col_cadres].max()}, Moyenne: {df_csp[col_cadres].mean()}")
print(f"Nombre de valeurs nulles: {df_csp[col_cadres].isna().sum()}")

print(f"\nStatistiques pour la colonne {col_ouvriers}:")
print(f"Min: {df_csp[col_ouvriers].min()}, Max: {df_csp[col_ouvriers].max()}, Moyenne: {df_csp[col_ouvriers].mean()}")
print(f"Nombre de valeurs nulles: {df_csp[col_ouvriers].isna().sum()}")

# Étape 5 : Calcul des ratios CSP avec gestion des valeurs nulles et division par zéro
# Pour éviter la division par zéro, on ajoute une petite valeur
epsilon = 1e-6  # Une très petite valeur pour éviter la division par zéro

# On calcule le ratio cadres/ouvriers
df_csp['ratio_cadres_ouvriers'] = df_csp[col_cadres] / (df_csp[col_ouvriers] + epsilon)

# On calcule aussi le pourcentage de cadres sur la population totale
df_csp['pct_cadres'] = 100 * df_csp[col_cadres] / df_csp['pop20']

# Et le pourcentage d'ouvriers sur la population totale
df_csp['pct_ouvriers'] = 100 * df_csp[col_ouvriers] / df_csp['pop20']

# On affiche quelques statistiques sur ces ratios
print("\nStatistiques pour le ratio cadres/ouvriers:")
print(f"Min: {df_csp['ratio_cadres_ouvriers'].min()}, Max: {df_csp['ratio_cadres_ouvriers'].max()}")
print(f"Médiane: {df_csp['ratio_cadres_ouvriers'].median()}, Moyenne: {df_csp['ratio_cadres_ouvriers'].mean()}")

# Suppression des valeurs aberrantes (ratios trop élevés)
q95 = df_csp['ratio_cadres_ouvriers'].quantile(0.95)
df_csp = df_csp[df_csp['ratio_cadres_ouvriers'] <= q95]

print(f"Après suppression des valeurs aberrantes (> 95ème percentile):")
print(f"Min: {df_csp['ratio_cadres_ouvriers'].min()}, Max: {df_csp['ratio_cadres_ouvriers'].max()}")
print(f"Médiane: {df_csp['ratio_cadres_ouvriers'].median()}, Moyenne: {df_csp['ratio_cadres_ouvriers'].mean()}")

# Étape 6 : Agrégation par aire urbaine
df_agg = df_csp.groupby('AAV').agg({
    'ratio_cadres_ouvriers': 'mean',
    'pct_cadres': 'mean',
    'pct_ouvriers': 'mean',
    'pop20': 'sum'
}).reset_index()

print(f"\nAgrégation réussie: {len(df_agg)} aires urbaines analysées")
print("Top 5 des aires urbaines par ratio cadres/ouvriers:")
print(df_agg.sort_values('ratio_cadres_ouvriers', ascending=False).head())

# Étape 7 : Fusion avec les données des aires urbaines
df_merged = pd.merge(df_agg, df_au[['AAV', 'LIBAAV2020', 'TAAV2017', 'NB_COM']], on='AAV', how='inner')
print(f"\nFusion réussie: {len(df_merged)} aires urbaines avec données complètes")

# Étape 8 : Création des graphiques
# 8.1 : Graphique ratio cadres/ouvriers vs population
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=df_merged, 
    x='TAAV2017', 
    y='ratio_cadres_ouvriers',
    size='NB_COM',
    sizes=(20, 200),
    alpha=0.7,
    palette='viridis'
)

# Ajouter une ligne de tendance
sns.regplot(
    data=df_merged, 
    x='TAAV2017', 
    y='ratio_cadres_ouvriers',
    scatter=False, 
    color='red',
    line_kws={"linewidth": 2}
)

# Étiquettes pour les grandes aires urbaines et les cas extrêmes
for idx, row in df_merged.nlargest(5, 'TAAV2017').iterrows():
    plt.annotate(
        row['LIBAAV2020'], 
        (row['TAAV2017'], row['ratio_cadres_ouvriers']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold'
    )

# Étiquettes pour les 3 aires urbaines avec le plus fort ratio
for idx, row in df_merged.nlargest(3, 'ratio_cadres_ouvriers').iterrows():
    plt.annotate(
        row['LIBAAV2020'], 
        (row['TAAV2017'], row['ratio_cadres_ouvriers']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        color='green'
    )

plt.title("Relation entre taille des aires urbaines et ratio cadres/ouvriers", fontsize=16)
plt.xlabel("Population totale de l'aire urbaine (millions)", fontsize=14)
plt.ylabel("Ratio cadres/ouvriers", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('relation_aires_urbaines_ratio_co.png', dpi=300)
plt.show()

# 8.2 : Graphique pourcentage de cadres vs population
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=df_merged, 
    x='TAAV2017', 
    y='pct_cadres',
    size='NB_COM',
    sizes=(20, 200),
    alpha=0.7,
    palette='Blues'
)

# Ajouter une ligne de tendance
sns.regplot(
    data=df_merged, 
    x='TAAV2017', 
    y='pct_cadres',
    scatter=False, 
    color='blue',
    line_kws={"linewidth": 2}
)

# Étiquettes pour les grandes aires urbaines
for idx, row in df_merged.nlargest(5, 'TAAV2017').iterrows():
    plt.annotate(
        row['LIBAAV2020'], 
        (row['TAAV2017'], row['pct_cadres']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Relation entre taille des aires urbaines et pourcentage de cadres", fontsize=16)
plt.xlabel("Population totale de l'aire urbaine (millions)", fontsize=14)
plt.ylabel("Pourcentage de cadres", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('relation_aires_urbaines_pct_cadres.png', dpi=300)
plt.show()

# 8.3 : Graphique pourcentage d'ouvriers vs population
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=df_merged, 
    x='TAAV2017', 
    y='pct_ouvriers',
    size='NB_COM',
    sizes=(20, 200),
    alpha=0.7,
    palette='Oranges'
)

# Ajouter une ligne de tendance
sns.regplot(
    data=df_merged, 
    x='TAAV2017', 
    y='pct_ouvriers',
    scatter=False, 
    color='orange',
    line_kws={"linewidth": 2}
)

# Étiquettes pour les grandes aires urbaines
for idx, row in df_merged.nlargest(5, 'TAAV2017').iterrows():
    plt.annotate(
        row['LIBAAV2020'], 
        (row['TAAV2017'], row['pct_ouvriers']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Relation entre taille des aires urbaines et pourcentage d'ouvriers", fontsize=16)
plt.xlabel("Population totale de l'aire urbaine (millions)", fontsize=14)
plt.ylabel("Pourcentage d'ouvriers", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('relation_aires_urbaines_pct_ouvriers.png', dpi=300)
plt.show()

# Analyses statistiques
print("\n--- Analyses statistiques ---")
print(f"Corrélation entre population et ratio cadres/ouvriers: {df_merged['TAAV2017'].corr(df_merged['ratio_cadres_ouvriers']):.4f}")
print(f"Corrélation entre population et % cadres: {df_merged['TAAV2017'].corr(df_merged['pct_cadres']):.4f}")
print(f"Corrélation entre population et % ouvriers: {df_merged['TAAV2017'].corr(df_merged['pct_ouvriers']):.4f}")

print("\nTOP 10 des aires urbaines par ratio cadres/ouvriers:")
print(df_merged.sort_values('ratio_cadres_ouvriers', ascending=False).head(10)[['LIBAAV2020', 'TAAV2017', 'ratio_cadres_ouvriers', 'pct_cadres', 'pct_ouvriers']])