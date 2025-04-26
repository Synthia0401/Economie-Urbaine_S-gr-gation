import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynib import OLS

# 1. Charger les données avec les chemins corrects pour l'environnement local
conso = pd.read_csv('../inputs/csv/conso2009_2023_resultats_com.csv', sep=';', dtype={'idcom': str})
aav = pd.read_csv('../inputs/csv/AAV2020_au_01-01-2024.csv', sep=';', dtype={'AAV2020': str})
pop = pd.read_csv('../inputs/csv/base-pop-historiques-1876-2022.csv', sep=';', dtype={'CODGEO': str})

# 2. Préparer les indicateurs
# 2.1 Stocker la consommation totale NAF -> artificialisé par commune
conso_comm = conso[['idcom', 'naf09art23']].copy()

# 2.2 Calculer la croissance démographique 2014-2020 par commune
pop20 = pop[['CODGEO', 'PMUN2020']].rename(columns={'PMUN2020': 'pop2020'})
pop14 = pop[['CODGEO', 'PMUN2014']].rename(columns={'PMUN2014': 'pop2014'})
pop_growth = pop20.merge(pop14, on='CODGEO', how='inner')
pop_growth['pop_growth_pct'] = (pop_growth['pop2020'] - pop_growth['pop2014']) / pop_growth['pop2014'] * 100

# 3. Fusionner les tables
df = conso_comm.merge(pop_growth[['CODGEO', 'pop_growth_pct']],
                      left_on='idcom', right_on='CODGEO', how='inner') \
               .merge(aav[['AAV2020', 'LIBAAV2020']], left_on='idcom', right_on='AAV2020', how='inner')

# 4. Supprimer les valeurs manquantes ou infinies
df = df.dropna(subset=['naf09art23', 'pop_growth_pct'])
df = df[~df['pop_growth_pct'].isin([float('inf'), float('-inf')])]  # Supprimer les valeurs infinies

# 5. Agréger par aire urbaine
grouped = df.groupby('LIBAAV2020').agg({
    'naf09art23': 'sum',
    'pop_growth_pct': 'mean'
}).reset_index()

# 6. Créer la colonne 'total_naf_consumed'
grouped = grouped.rename(columns={'naf09art23': 'total_naf_consumed'})

# 7. Préparer les données pour la régression
X = grouped['pop_growth_pct'].values.reshape(-1, 1)  # Variable indépendante (croissance démographique)
y = grouped['total_naf_consumed'].values  # Variable dépendante (consommation de terrain NAF)

# 8. Appliquer le modèle de régression linéaire avec PyNiB
model = OLS(X, y)  # Initialiser le modèle de régression
results = model.fit()  # Ajuster le modèle aux données

# 9. Résumé du modèle
print("\nRésumé de la régression linéaire:")
print(results.summary())

# 10. Visualisation
plt.figure(figsize=(8,6))
plt.scatter(grouped['pop_growth_pct'], grouped['total_naf_consumed'], label='Données')
plt.plot(grouped['pop_growth_pct'], results.fittedvalues, color='red', label='Régression linéaire')
plt.xlabel('Croissance démographique (%) 2014-2020')
plt.ylabel('Consommation NAF (2009-2023, m²)')
plt.title('Consommation de surface NAF vs Croissance démographique')
plt.legend()
plt.grid(True)
plt.show()
