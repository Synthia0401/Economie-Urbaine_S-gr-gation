import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Chargement des données
df_au = pd.read_csv('data/AAV2020_au_01-01-2024.csv', sep=';')
df_csp = pd.read_csv('data/conso2009_2023_resultats_com.csv', sep=';', low_memory=False)

# Préparation des données aires urbaines
df_au['AAV'] = df_au['AAV2020'].astype(str).str.zfill(3)
df_au['population_totale'] = df_au['TAAV2017'].astype(float)  # Assurer que c'est bien un float
df_au = df_au[df_au['population_totale'] > 0]

# Préparation des données communes par CSP
df_csp['AAV'] = df_csp['aav2020'].astype(str).str.extract(r'^(\d{3})')[0]
df_csp = df_csp.dropna(subset=['AAV'])

# Colonnes CSP
csp_cols = [col for col in df_csp.columns if 'act' in col and col.startswith('art')][:5]

# Si pas de colonnes trouvées, utiliser un exemple
if len(csp_cols) == 0:
    print("Aucune colonne CSP trouvée, utilisation de colonnes d'exemple")
    csp_cols = ['art22act15', 'art22act16', 'art22act17']

# Calcul de l'indice de ségrégation
segregation_data = []

for aav in df_au['AAV'].unique():
    communes = df_csp[df_csp['AAV'] == aav]
    
    if len(communes) > 1:
        variations = []
        
        for col in csp_cols:
            if col in communes.columns:
                pop_col = 'pop20' if 'pop20' in communes.columns else 'population'
                if pop_col in communes.columns:
                    communes_temp = communes[(communes[col].notna()) & (communes[pop_col] > 0)].copy()
                    if not communes_temp.empty:
                        communes_temp.loc[:, 'prop'] = communes_temp[col] / communes_temp[pop_col]
                        if communes_temp['prop'].mean() > 0:
                            cv = communes_temp['prop'].std() / communes_temp['prop'].mean()
                            variations.append(cv)
        
        if variations:
            # Ajouter un petit bruit aléatoire pour éviter les alignements verticaux
            pop_jittered = df_au[df_au['AAV'] == aav]['population_totale'].values[0] * (1 + np.random.normal(0, 0.03))
            
            segregation_data.append({
                'AAV': aav,
                'segregation_index': np.mean(variations),
                'nb_communes': len(communes),
                'population_totale_jittered': pop_jittered
            })

# Créer DataFrame et fusionner
df_segregation = pd.DataFrame(segregation_data)
df_final = pd.merge(df_au, df_segregation, on='AAV', how='inner')

# Transformer la population en log pour mieux visualiser (comme dans l'image 2)
df_final['population_log'] = np.log10(df_final['population_totale'])

# Création du graphique
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid", {'grid.linestyle': ':'})

# Utiliser les valeurs avec jitter pour éviter les alignements verticaux
plt.scatter(
    df_final['population_totale_jittered'], 
    df_final['segregation_index'],
    alpha=0.6,
    s=25,
    color='steelblue',
    edgecolor='none'
)

# Courbe de tendance
model = np.polyfit(df_final['population_totale'], df_final['segregation_index'], 1)
x_range = np.linspace(df_final['population_totale'].min(), df_final['population_totale'].max(), 100)
plt.plot(x_range, np.polyval(model, x_range), color='red', alpha=0.7, linewidth=1.5, linestyle='-')

# Ajout du R² de façon discrète
r_squared = stats.pearsonr(df_final['population_totale'], df_final['segregation_index'])[0]**2
plt.text(
    0.05, 0.95, 
    f'R² = {r_squared:.2f}', 
    transform=plt.gca().transAxes,
    fontsize=9,
    color='black',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.5')
)

# Étiquettes uniquement pour les grandes villes
for _, row in df_final.nlargest(5, 'population_totale').iterrows():
    plt.annotate(
        row['LIBAAV2020'],
        (row['population_totale_jittered'], row['segregation_index']),
        fontsize=8,
        xytext=(5, 0),
        textcoords='offset points',
        color='navy',
        alpha=0.8
    )

# Personnalisation du graphique
plt.title("Relation entre taille des aires urbaines et niveau de ségrégation", fontsize=12)
plt.xlabel("Population totale de l'aire urbaine (2017)", fontsize=10)
plt.ylabel("Indice de ségrégation (variation des CSP)", fontsize=10)

plt.tight_layout()
plt.savefig('segregation_taille_AU_scatter.png', dpi=300)
plt.show()

print(f"R² = {r_squared:.3f}")