import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Chargement des données
df_au = pd.read_csv('data/AAV2020_au_01-01-2024.csv', sep=';')
df_pop = pd.read_csv('data/base-pop-historiques-1876-2022.csv', sep=';')

# Calcul de la croissance démographique récente (2020 - 2017)
df_pop['croissance_demo'] = df_pop['PMUN2020'] - df_pop['PMUN2017']

# Agrégation par aire urbaine (fusion sur nom des villes)
df_merge = pd.merge(df_au, df_pop[['LIBGEO', 'croissance_demo']], 
                    left_on='LIBAAV2020', right_on='LIBGEO', how='inner')

# Ajout de la population totale actuelle
df_merge['population_totale'] = df_merge['TAAV2017']

# Vérifier et nettoyer les données manquantes ou aberrantes
df_merge = df_merge.dropna(subset=['population_totale', 'croissance_demo'])

# Graphique pertinent : Scatter plot avec régression linéaire
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_merge, x='population_totale', y='croissance_demo', size='NB_COM', alpha=0.7, legend=False)
sns.regplot(data=df_merge, x='population_totale', y='croissance_demo', scatter=False, color='red')

plt.title("Relation entre taille des AU et croissance démographique récente (2017–2020)", fontsize=16)
plt.xlabel("Population totale de l'AU (2017)", fontsize=14)
plt.ylabel("Croissance démographique (2020-2017)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()