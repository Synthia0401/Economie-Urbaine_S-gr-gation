import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Chargement des données
df_au = pd.read_csv('../inputs/csv/AAV2020_au_01-01-2024.csv', sep=';')
df_csp = pd.read_csv('../inputs/csv/conso2009_2023_resultats_com.csv', sep=';')

# Trouver les colonnes disponibles
col_actif = sorted([col for col in df_csp.columns if re.match(r'art\d{2}act\d{2}', col)])[-1]
col_pop = 'pop20'  # ajuste si tu utilises une autre année

# Préparation
df_csp = df_csp[['idcom', 'aav2020', col_actif, col_pop]].dropna()
df_csp['AAV'] = df_csp['aav2020'].astype(str).str.extract(r'^(\d{3})')[0]
df_csp = df_csp.dropna(subset=['AAV'])

# Calcul de la part réelle d'actifs
df_csp['part_actifs'] = df_csp[col_actif] / df_csp[col_pop]

# Agrégation par aire urbaine
df_agg = df_csp.groupby('AAV').agg({'part_actifs': 'mean'}).reset_index()

# Préparation AU
df_au['AAV'] = df_au['AAV2020'].astype(str).str.zfill(3)
df_au['population_totale'] = df_au['TAAV2017']

# Fusion
df = pd.merge(df_au, df_agg, on='AAV', how='inner')

# Garder uniquement les 8 plus hautes et 8 plus basses (augmenté pour plus de données)
top_bottom = pd.concat([
    df.nlargest(8, 'part_actifs'),
    df.nsmallest(8, 'part_actifs')
])

# Définir un style plus moderne
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# Créer le barplot avec une palette plus contrastée
bars = sns.barplot(
    data=top_bottom.sort_values('part_actifs', ascending=False), 
    x='part_actifs', 
    y='LIBAAV2020', 
    palette='RdBu_r',  # Palette rouge-bleu inversée
    edgecolor='black',
    linewidth=0.5
)

# Ajouter les valeurs à la fin des barres
for i, v in enumerate(top_bottom.sort_values('part_actifs', ascending=False)['part_actifs']):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=9)

# Améliorer le titre et les étiquettes
plt.title("Aires urbaines avec les parts d'actifs les plus extrêmes", fontsize=14, fontweight='bold')
plt.xlabel("Part d'actifs (actifs / population)", fontsize=12)
plt.ylabel("Aire urbaine", fontsize=12)

# Ajouter une ligne médiane pour référence
median_value = df['part_actifs'].median()
plt.axvline(x=median_value, color='gray', linestyle=':', linewidth=1)
plt.text(median_value + 0.01, -0.8, f"Médiane: {median_value:.2f}", fontsize=9, style='italic')

# Ajouter des informations contextuelles
plt.text(0.01, -1.5, 
         f"Nombre total d'aires urbaines: {len(df)}\nColonne utilisée pour le calcul: {col_actif}", 
         fontsize=8, style='italic')

# Ajuster les limites de l'axe x pour une meilleure visualisation
plt.xlim(top_bottom['part_actifs'].min() - 0.05, top_bottom['part_actifs'].max() + 0.1)

# Affiner la grille
plt.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.grid(False, axis='y')

# Ajouter une bordure au graphique
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.tight_layout()
plt.savefig('part_actifs_extremes.png', dpi=300)
plt.show()

# Afficher quelques statistiques sur les données
print(f"Moyenne des parts d'actifs: {df['part_actifs'].mean():.3f}")
print(f"Médiane des parts d'actifs: {df['part_actifs'].median():.3f}")
print(f"Écart-type: {df['part_actifs'].std():.3f}")