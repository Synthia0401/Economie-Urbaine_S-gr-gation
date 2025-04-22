#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script : folium_enaf_par_au.py
Objectif :
  • Charger les shapefiles AU et communes + CSV de conso.
  • Agréger la consommation ENAF par aire urbaine.
  • Générer une carte interactive Folium avec un dégradé OrRd
    et une légende claire.
"""

import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
import numpy as np
from pathlib import Path

# 1. Chemins vers vos données (à adapter si nécessaire)
base       = Path(__file__).resolve().parent
shp_au     = base / "carte" / "aav2020_2024"   / "aav2020_2024.shp"
shp_com    = base / "carte" / "com_aav2020_2024" / "com_aav2020_2024.shp"
csv_conso  = base / "carte" / "conso2009_2023_resultats_com.csv"

# 2. Charger AU, communes et conso
gdf_au   = gpd.read_file(shp_au)
gdf_com  = gpd.read_file(shp_com)
df_conso = pd.read_csv(csv_conso, sep=";", low_memory=False, encoding="utf-8")

# 3. Préparer la jointure
df_conso["codgeo"] = df_conso["idcom"].astype(str).str.zfill(5)
conso_col = "surfcom2023"    # ou le nom exact dans votre CSV

gdf_com = gdf_com.merge(
    df_conso[["codgeo", conso_col]],
    on="codgeo", how="left"
)
gdf_com[conso_col] = gdf_com[conso_col].fillna(0)

# 4. Agréger par aire urbaine (champ "aav2020")
df_agg = (
    gdf_com
    .groupby("aav2020", as_index=False)[conso_col]
    .sum()
    .rename(columns={conso_col: "ENAF_tot"})
)

gdf_au = gdf_au.merge(df_agg, on="aav2020", how="left")
gdf_au["ENAF_tot"] = gdf_au["ENAF_tot"].fillna(0)
# 5. Construire une StepColormap à 7 classes
vmin, vmax = gdf_au["ENAF_tot"].min(), gdf_au["ENAF_tot"].max()
# on découpe en intervalles égaux
steps = list(np.linspace(vmin, vmax, num=8))  # 7 intervalles
colormap = cm.StepColormap(
    colors=cm.linear.OrRd_09.colors,  # palette OrRd
    index=steps,
    vmin=vmin, vmax=vmax,
)

# 6. Créer la carte puis y ajouter le colormap
# 6. Créer la carte puis y ajouter le colormap
m = folium.Map(location=[46.2276, 2.2137], zoom_start=6)
colormap.add_to(m)

# 7. Ajouter la couche choroplèthe
folium.GeoJson(
    gdf_au.to_crs(epsg=4326),
    style_function=lambda feature: {
        "fillColor": colormap(feature["properties"]["ENAF_tot"]),
        "color": "gray",
        "weight": 0.3,
        "dashArray": "3",
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[next(c for c in gdf_au.columns if "lib" in c.lower()), "ENAF_tot"],
        aliases=["Aire urbaine", "ENAF 2023 (ha)"],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: white;
            border: 1px solid gray;
            border-radius: 3px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        """
    )
).add_to(m)

# 8. Ajouter la légende
colormap.add_to(m)

# 9. Sauvegarder la carte
out_html = base / "folium_enaf_par_au_2023.html"
m.save(out_html)
print(f"Carte Folium enregistrée sous : {out_html}")
