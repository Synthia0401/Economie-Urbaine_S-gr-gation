import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# chargements sans géométrie pour éviter l’erreur shapely2
df_conso = pd.read_csv('carte/conso2009_2023_resultats_com.csv', sep=';')
df_conso["codgeo"] = df_conso["idcom"].astype(str).str.zfill(5)
gdf_com = gpd.read_file('carte/com_aav2020_2024/com_aav2020_2024.shp', ignore_geometry=True)
gdf_au  = gpd.read_file('carte/aav2020_2024/aav2020_2024.shp',       ignore_geometry=True)

# fusion + agrégation
gdf_com = gdf_com.merge(df_conso[['codgeo','surfcom2023']], on='codgeo', how='left')
gdf_com['surfcom2023'] = gdf_com['surfcom2023'].fillna(0)
df_agg = gdf_com.groupby('aav2020', as_index=False)['surfcom2023'].sum().rename(columns={'surfcom2023':'ENAF_tot'})
gdf_au = gdf_au.merge(df_agg, on='aav2020', how='left').fillna(0)

# Top 10 et tracé
nom_col = next(c for c in gdf_au.columns if 'lib' in c.lower())
top10 = (gdf_au[[nom_col,'ENAF_tot']]
         .sort_values('ENAF_tot',ascending=False)
         .head(10))

plt.figure(figsize=(10,6))
plt.plot(top10[nom_col], top10['ENAF_tot'], marker='o')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Surface ENAF consommée (ha)')
plt.title('Top 10 des Aires Urbaines par ENAF consommée (2023)')
plt.tight_layout()
plt.show()
