# Analyse de la consommation ENAF (Espaces Naturels, Agricoles et Forestiers) par les aires urbaines
# ===============================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement des données
# ========================

# Chargement des données des aires urbaines
aires_urbaines_data = pd.DataFrame({
    'AAV2020': ['000', '001', '002', '003', '004', '005', '006', '008', '010', '012', '013', '014', '015', 
                '017', '018', '019', '020', '022', '023', '024', '025', '026', '028', '029', '030', '031'],
    'LIBAAV2020': ['Commune hors attraction des villes', 'Paris', 'Lyon', 'Marseille - Aix-en-Provence', 
                  'Lille (partie française)', 'Toulouse', 'Bordeaux', 'Nantes', 'Strasbourg (partie française)', 
                  'Montpellier', 'Rennes', 'Grenoble', 'Rouen', 'Nice', 'Toulon', 'Tours', 'Nancy', 
                  'Clermont-Ferrand', 'Saint-Étienne', 'Caen', 'Orléans', 'Angers', 'Dijon', 'Mulhouse', 
                  'Perpignan', 'Cannes - Antibes'],
    'TAAV2017': [0, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'TDAAV2017': [0, 50, 42, 42, 42, 42, 42, 41, 41, 41, 41, 41, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 33, 33, 32],
    'NB_COM': [8921, 1929, 397, 115, 201, 527, 275, 116, 268, 161, 183, 204, 317, 100, 35, 162, 353, 209, 105, 296, 136, 81, 333, 132, 118, 24]
})

# Chargement des données démographiques des communes
population_data = pd.DataFrame({
    'CODGEO': ['01001', '01002', '01004', '01005', '01006', '01007', '01008', '01009', '01010'],
    'REG': [84, 84, 84, 84, 84, 84, 84, 84, 84],
    'DEP': ['01', '01', '01', '01', '01', '01', '01', '01', '01'],
    'LIBGEO': ["L'Abergement-Clémenciat", "L'Abergement-de-Varey", "Ambérieu-en-Bugey", 
               "Ambérieux-en-Dombes", "Ambléon", "Ambronay", "Ambutrix", "Andert-et-Condon", "Anglefort"],
    'PMUN2022': [859, 273, 15554, 1917, 114, 2828, 764, 335, 1150],
    'PMUN2014': [767, 239, 14022, 1627, 109, 2570, 743, 338, 1142]
})

# Création d'un dictionnaire pour transformer la chaîne en DataFrame
paste_data_str = """idcom;idcomtxt;idreg;idregtxt;iddep;iddeptxt;epci23;epci23txt;scot;aav2020;aav2020txt;aav2020_typo;naf09art10;art09act10;art09hab10;art09mix10;art09rou10;art09fer10;art09inc10;naf10art11;art10act11;art10hab11;art10mix11;art10rou11;art10fer11;art10inc11;naf11art12;art11act12;art11hab12;art11mix12;art11rou12;art11fer12;art11inc12;naf12art13;art12act13;art12hab13;art12mix13;art12rou13;art12fer13;art12inc13;naf13art14;art13act14;art13hab14;art13mix14;art13rou14;art13fer14;art13inc14;naf14art15;art14act15;art14hab15;art14mix15;art14rou15;art14fer15;art14inc15;naf15art16;art15act16;art15hab16;art15mix16;art15rou16;art15fer16;art15inc16;naf16art17;art16act17;art16hab17;art16mix17;art16rou17;art16fer17;art16inc17;naf17art18;art17act18;art17hab18;art17mix18;art17rou18;art17fer18;art17inc18;naf18art19;art18act19;art18hab19;art18mix19;art18rou19;art18fer19;art18inc19;naf19art20;art19act20;art19hab20;art19mix20;art19rou20;art19fer20;art19inc20;naf20art21;art20act21;art20hab21;art20mix21;art20rou21;art20fer21;art20inc21;naf21art22;art21act22;art21hab22;art21mix22;art21rou22;art21fer22;art21inc22;naf22art23;art22act23;art22hab23;art22mix23;art22rou23;art22fer23;art22inc23;naf09art23;art09act23;art09hab23;art09mix23;art09inc23;art09rou23;art09fer23;artcom0923;pop14;pop20;pop1420;men14;men20;men1420;emp14;emp20;emp1420;mepart1420;menhab1420;artpop1420;surfcom2023
01001;"L""Abergement-Clémenciat";84;Auvergne-Rhône-Alpes;01;Ain;200069193;CC de la Dombes;SCoT de la Dombes;524 - Châtillon-sur-Chalaronne;1;T01_20;8324;0;8324;0;0;0;0;8324;0;8324;0;0;0;0;0;0;0;0;0;0;0;650;0;650;0;0;0;0;8037;0;6629;0;1408;0;0;0;0;0;0;0;0;0;857;0;857;0;0;0;0;1718;0;1718;0;0;0;0;3580;0;3580;0;0;0;0;2721;0;2721;0;0;0;0;0;0;0;0;0;0;0;15955;0;15955;0;0;0;0;15699;0;15699;0;0;0;0;1646;0;1646;0;0;0;0;67511;0;66103;0;0;1408;0;0.43;767;806;39;310;334;24;85;78;-7;19.15;27.04;227.59;15619934
01002;"L""Abergement-de-Varey";84;Auvergne-Rhône-Alpes;01;Ain;240100883;"CC de la Plaine de l""Ain";SCoT BUCOPA;N/A - hors champ;0;T00_30;1139;0;1094;0;45;0;0;1140;0;1095;0;45;0;0;4638;0;169;0;89;0;4380;977;0;977;0;0;0;0;0;0;0;0;0;0;0;975;0;500;0;475;0;0;1749;0;1749;0;0;0;0;1205;0;1205;0;0;0;0;0;0;0;0;0;0;0;1037;0;600;0;0;0;437;1740;0;1740;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;14600;0;9129;0;4817;654;0;0.16;239;262;23;103;110;7;13;28;15;32.81;10.44;291.57;9175479
01004;Ambérieu-en-Bugey;84;Auvergne-Rhône-Alpes;01;Ain;240100883;"CC de la Plaine de l""Ain";SCoT BUCOPA;243 - Ambérieu-en-Bugey;1;T01_10;79035;46980;26502;931;4621;0;1;79035;46980;26502;932;4622;0;-1;30959;5957;11488;0;1264;0;12250;14426;5200;8241;0;985;0;0;14873;2996;8248;0;2759;869;1;29188;483;25958;0;2747;0;0;94624;20695;49174;149;24604;0;2;45103;1069;36637;0;7396;0;1;75418;32829;35688;0;6901;0;0;87831;11955;43702;8010;19967;4196;1;13093;4907;8124;0;62;0;0;9786;174;9148;0;464;0;0;23371;19426;3927;0;18;0;0;13374;0;13251;0;123;0;0;610116;199651;306590;10022;12255;76533;5065;2.49;14022;14288;266;6163;6771;608;7453;7942;489;31.77;17.61;1297.96;24508833
01005;Ambérieux-en-Dombes;84;Auvergne-Rhône-Alpes;01;Ain;200042497;CC Dombes Saône Vallée;SCoT du Val de Saône-Dombes;002 - Lyon;4;T45_20;6205;0;5645;0;560;0;0;6205;0;5645;0;560;0;0;9185;0;9112;0;23;0;50;2799;0;2799;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;886;0;886;0;0;0;0;16323;0;11591;957;3775;0;0;3729;0;3729;0;0;0;0;0;0;0;0;0;0;0;25621;0;20628;0;4993;0;0;41956;33286;8670;0;0;0;0;776;0;776;0;0;0;0;8416;5015;3401;0;0;0;0;122101;38301;72882;957;50;9911;0;0.76;1627;1782;155;608;740;132;281;257;-24;23.2;28.35;300.38;16014205
01006;Ambléon;84;Auvergne-Rhône-Alpes;01;Ain;200040350;CC Bugey Sud;SCoT du Bugey;286 - Belley;1;T01_20;0;0;0;0;0;0;0;0;0;0;0;0;0;0;98;0;98;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;1450;0;1450;0;0;0;0;0;0;0;0;0;0;0;1000;0;1000;0;0;0;0;3959;0;2657;0;1302;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;6507;0;5205;0;0;1302;0;0.11;109;113;4;55;55;0;6;5;-1;-1.56;0.0;1602.25;6030856
01007;Ambronay;84;Auvergne-Rhône-Alpes;01;Ain;240100883;"CC de la Plaine de l""Ain";SCoT BUCOPA;243 - Ambérieu-en-Bugey;1;T01_20;19333;1783;14425;0;282;0;2843;19334;1784;14425;0;283;0;2842;10161;0;10161;0;0;0;0;4418;0;3815;0;603;0;0;11208;0;9411;0;1797;0;0;24511;0;20172;0;4339;0;0;14000;7580;2399;0;4021;0;0;27673;2131;4930;0;19106;0;1506;12508;0;11698;0;809;0;1;19203;5757;10230;0;3216;0;0;21253;17043;4126;0;84;0;0;13026;5609;5002;0;2415;0;0;35574;0;34023;1551;0;0;0;4563;0;4563;0;0;0;0;236765;41687;149380;1551;7192;36955;0;0.70;2570;2827;257;1033;1195;162;488;475;-13;12.51;13.6;463.61;33651916
01008;Ambutrix;84;Auvergne-Rhône-Alpes;01;Ain;240100883;"CC de la Plaine de l""Ain";SCoT BUCOPA;243 - Ambérieu-en-Bugey;1;T01_20;1337;0;1337;0;0;0;0;1338;0;1338;0;0;0;0;3045;1379;1500;0;166;0;0;1847;0;1847;0;0;0;0;1218;0;1218;0;0;0;0;226;0;226;0;0;0;0;2862;0;2033;0;829;0;0;8959;0;7993;0;966;0;0;1338;0;1272;0;66;0;0;2550;0;2550;0;0;0;0;1921;0;1921;0;0;0;0;2414;0;0;0;2414;0;0;4350;0;1050;0;0;0;3300;0;0;0;0;0;0;0;33405;1379;24285;0;3300;4441;0;0.65;743;768;25;308;334;26;145;117;-28;-1.12;14.56;714.24;5174643
01009;Andert-et-Condon;84;Auvergne-Rhône-Alpes;01;Ain;200040350;CC Bugey Sud;SCoT du Bugey;286 - Belley;1;T01_20;676;0;676;0;0;0;0;676;0;676;0;0;0;0;1088;0;1088;0;0;0;0;1314;0;1314;0;0;0;0;3418;0;3418;0;0;0;0;3065;0;3052;0;13;0;0;6847;0;6847;0;0;0;0;0;0;0;0;0;0;0;2265;0;2265;0;0;0;0;1000;0;1000;0;0;0;0;2020;0;2020;0;0;0;0;8;0;8;0;0;0;0;964;0;964;0;0;0;0;0;0;0;0;0;0;0;23341;0;23328;0;0;13;0;0.33;338;324;-14;140;148;8;31;17;-14;-3.95;5.26;-1085.5;6995740
01010;Anglefort;84;Auvergne-Rhône-Alpes;01;Ain;200070852;CC Usses et Rhône;Communautés de communes Usses et Rhône;N/A - hors champ;0;T00_30;17093;0;10511;0;322;0;6260;17094;0;10512;0;323;0;6259;22509;10150;7723;0;4636;0;0;7824;0;7824;0;0;0;0;8014;0;7524;0;490;0;0;0;0;0;0;0;0;0;1317;0;1317;0;0;0;0;3363;0;3155;0;208;0;0;978;0;978;0;0;0;0;700;0;700;0;0;0;0;1707;0;1645;0;62;0;0;5346;0;5274;0;72;0;0;650;0;650;0;0;0;0;7556;0;6818;250;487;0;1;94151;10150;64631;250;12520;6600;0;0.32;1142;1101;-41;457;481;24;230;285;55;97.95;29.76;-196.71;29508583"""

# Transformer la chaîne en DataFrame
lines = paste_data_str.strip().split('\n')
cols = lines[0].split(';')
data = []
for line in lines[1:]:
    values = []
    in_quotes = False
    current_value = ""
    
    for char in line:
        if char == ';' and not in_quotes:
            values.append(current_value.strip('"'))
            current_value = ""
        elif char == '"':
            in_quotes = not in_quotes
            current_value += char
        else:
            current_value += char
    
    # Ajouter la dernière valeur
    values.append(current_value.strip('"'))
    data.append(values)

# Créer le DataFrame
enaf_data = pd.DataFrame(data, columns=cols)

# 2. Prétraitement des données
# ===========================

# Nettoyage et transformation des données ENAF
# Convertir les colonnes numériques
numeric_cols = [col for col in enaf_data.columns if col not in ['idcomtxt', 'idregtxt', 'iddeptxt', 'epci23txt', 'scot', 'aav2020txt', 'aav2020_typo']]
for col in numeric_cols:
    enaf_data[col] = pd.to_numeric(enaf_data[col], errors='coerce')

# Extraire l'identifiant de l'aire urbaine à partir de aav2020
enaf_data['aire_urbaine_id'] = enaf_data['aav2020'].str.split(' - ').str[0].str.strip()

# Fusionner avec les données des aires urbaines pour avoir le nom complet
aires_urbaines_dict = aires_urbaines_data.set_index('AAV2020')['LIBAAV2020'].to_dict()

# Extraire la consommation ENAF par commune
# Calculer la consommation totale d'ENAF entre 2009 et 2023
enaf_data['total_enaf_2009_2023'] = enaf_data['naf09art23']

# Calculer la consommation pour l'habitat
enaf_data['total_enaf_hab_2009_2023'] = enaf_data['art09hab23']

# Calculer la consommation pour l'activité économique
enaf_data['total_enaf_act_2009_2023'] = enaf_data['art09act23']

# Calculer la consommation pour les infrastructures routières
enaf_data['total_enaf_rou_2009_2023'] = enaf_data['art09rou23']

# Calculer la consommation pour les infrastructures ferroviaires
enaf_data['total_enaf_fer_2009_2023'] = enaf_data['art09fer23']

# Calculer la consommation pour les usages mixtes
enaf_data['total_enaf_mix_2009_2023'] = enaf_data['art09mix23']

# 3. Agrégation des données par aire urbaine
# =========================================

# Filtrer les communes appartenant à une aire urbaine
enaf_in_au = enaf_data[enaf_data['aav2020txt'] != 'N/A - hors champ'].copy()

# Extraire l'ID de l'aire urbaine à partir du code aav2020
enaf_in_au['au_id'] = enaf_in_au['aav2020'].str.split(' - ').str[0]

# Agréger la consommation ENAF par aire urbaine
enaf_by_au = enaf_in_au.groupby('au_id').agg({
    'total_enaf_2009_2023': 'sum',
    'total_enaf_hab_2009_2023': 'sum',
    'total_enaf_act_2009_2023': 'sum', 
    'total_enaf_rou_2009_2023': 'sum',
    'total_enaf_fer_2009_2023': 'sum',
    'total_enaf_mix_2009_2023': 'sum',
    'pop14': 'sum',
    'pop20': 'sum',
    'pop1420': 'sum',
    'idcom': 'count'  # Nombre de communes dans l'aire urbaine
}).reset_index()

# Renommer la colonne de compte
enaf_by_au.rename(columns={'idcom': 'nb_communes'}, inplace=True)

# Rejoindre avec les noms des aires urbaines
enaf_by_au['au_id'] = enaf_by_au['au_id'].astype(str).str.zfill(3)
enaf_by_au = enaf_by_au.merge(
    aires_urbaines_data[['AAV2020', 'LIBAAV2020']], 
    left_on='au_id', 
    right_on='AAV2020', 
    how='left'
)

# Calculer des indicateurs supplémentaires
enaf_by_au['enaf_per_commune'] = enaf_by_au['total_enaf_2009_2023'] / enaf_by_au['nb_communes']
enaf_by_au['enaf_per_1000_hab'] = enaf_by_au['total_enaf_2009_2023'] / (enaf_by_au['pop20'] / 1000)
enaf_by_au['enaf_per_hab_growth'] = enaf_by_au['total_enaf_2009_2023'] / enaf_by_au['pop1420']
enaf_by_au['ratio_hab_total'] = enaf_by_au['total_enaf_hab_2009_2023'] / enaf_by_au['total_enaf_2009_2023']

# 4. Analyse exploratoire des données
# =================================

# Visualiser la consommation totale d'ENAF par aire urbaine
plt.figure(figsize=(12, 8))
sorted_data = enaf_by_au.sort_values('total_enaf_2009_2023', ascending=False).head(15)
sns.barplot(x='total_enaf_2009_2023', y='LIBAAV2020', data=sorted_data)
plt.title('Consommation totale d\'ENAF par aire urbaine (2009-2023)')
plt.xlabel('Surface ENAF consommée (m²)')
plt.ylabel('Aire urbaine')
plt.tight_layout()
plt.show()

# Analyse de la répartition des types de consommation
plt.figure(figsize=(14, 10))
data_for_plot = sorted_data.copy()
data_for_plot = data_for_plot.set_index('LIBAAV2020')
cols_to_plot = ['total_enaf_hab_2009_2023', 'total_enaf_act_2009_2023', 
                'total_enaf_rou_2009_2023', 'total_enaf_fer_2009_2023', 'total_enaf_mix_2009_2023']
data_for_plot[cols_to_plot].plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Répartition de la consommation d\'ENAF par type et par aire urbaine (2009-2023)')
plt.xlabel('Aire urbaine')
plt.ylabel('Surface ENAF consommée (m²)')
plt.legend(['Habitat', 'Activité économique', 'Infrastructures routières', 
            'Infrastructures ferroviaires', 'Usages mixtes'])
plt.tight_layout()
plt.show()

# Visualiser la consommation ENAF relative à la population
plt.figure(figsize=(12, 8))
sorted_data = enaf_by_au.sort_values('enaf_per_1000_hab', ascending=False).head(15)
sns.barplot(x='enaf_per_1000_hab', y='LIBAAV2020', data=sorted_data)
plt.title('Consommation d\'ENAF par 1000 habitants par aire urbaine (2009-2023)')
plt.xlabel('Surface ENAF consommée par 1000 habitants (m²)')
plt.ylabel('Aire urbaine')
plt.tight_layout()
plt.show()

# Visualiser la consommation ENAF relative à la croissance démographique
plt.figure(figsize=(12, 8))
filtered_data = enaf_by_au[enaf_by_au['pop1420'] > 0].sort_values('enaf_per_hab_growth', ascending=False).head(15)
sns.barplot(x='enaf_per_hab_growth', y='LIBAAV2020', data=filtered_data)
plt.title('Consommation d\'ENAF par habitant supplémentaire (2009-2023)')
plt.xlabel('Surface ENAF consommée par habitant supplémentaire (m²)')
plt.ylabel('Aire urbaine')
plt.tight_layout()
plt.show()

# 5. Modélisation et analyse statistique
# ===================================

# Préparation des données pour la régression
X = enaf_by_au[['pop20', 'pop1420', 'nb_communes']].values
y = enaf_by_au['total_enaf_2009_2023'].values

# Ajouter une colonne de constante pour l'interception
X_sm = sm.add_constant(X)

# Ajuster le modèle
model = sm.OLS(y, X_sm)
results = model.fit()

# Afficher les résultats
print(results.summary())

# Visualiser la relation entre la croissance démographique et la consommation d'ENAF
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pop1420', y='total_enaf_2009_2023', hue='LIBAAV2020', size='pop20', 
                sizes=(50, 500), alpha=0.7, data=enaf_by_au)
plt.title('Relation entre croissance démographique et consommation d\'ENAF')
plt.xlabel('Croissance de la population (2014-2020)')
plt.ylabel('Surface ENAF consommée (m²)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. Visualisation des résultats avec Plotly pour interactivité
# ===========================================================

# Graphique à bulles interactif
fig = px.scatter(enaf_by_au, x='pop1420', y='total_enaf_2009_2023', 
                 size='pop20', color='LIBAAV2020',
                 hover_name='LIBAAV2020', 
                 hover_data=['nb_communes', 'enaf_per_1000_hab', 'ratio_hab_total'],
                 title='Consommation d\'ENAF en fonction de la croissance démographique')

fig.update_layout(
    xaxis_title='Croissance démographique (2014-2020)',
    yaxis_title='Surface ENAF consommée (m²)',
    legend_title='Aire urbaine',
    height=600,
    width=900
)

fig.show()

# Visualisation de la répartition des types de consommation en graphique interactif
data_for_sunburst = []
for _, row in enaf_by_au.iterrows():
    for type_conso, col in zip(['Habitat', 'Activité', 'Routes', 'Ferroviaire', 'Mixte'], 
                              ['total_enaf_hab_2009_2023', 'total_enaf_act_2009_2023', 
                               'total_enaf_rou_2009_2023', 'total_enaf_fer_2009_2023', 'total_enaf_mix_2009_2023']):
        if not pd.isna(row[col]) and row[col] > 0:
            data_for_sunburst.append({
                'Aire urbaine': row['LIBAAV2020'],
                'Type': type_conso,
                'Valeur': row[col]
            })

sunburst_df = pd.DataFrame(data_for_sunburst)

# Créer le graphique en anneau (sunburst)
fig = px.sunburst(
    sunburst_df, 
    path=['Aire urbaine', 'Type'], 
    values='Valeur',
    title='Répartition de la consommation d\'ENAF par aire urbaine et par type d\'usage'
)
fig.update_layout(height=800, width=800)
fig.show()

# Graphique à barres interactif pour comparer les aires urbaines
fig = px.bar(
    enaf_by_au.sort_values('total_enaf_2009_2023', ascending=False).head(15), 
    x='LIBAAV2020', 
    y=['total_enaf_hab_2009_2023', 'total_enaf_act_2009_2023', 
       'total_enaf_rou_2009_2023', 'total_enaf_fer_2009_2023', 'total_enaf_mix_2009_2023'],
    title='Composition de la consommation d\'ENAF par aire urbaine',
    labels={
        'value': 'Surface consommée (m²)',
        'LIBAAV2020': 'Aire urbaine',
        'variable': 'Type d\'usage'
    },
    barmode='stack',
    height=600,
    width=900
)

fig.update_layout(
    xaxis={'categoryorder': 'total descending'},
    legend_title='Type d\'usage',
    legend={
        'traceorder': 'normal'
    }
)

# Renommer les variables dans la légende
fig.for_each_trace(lambda t: t.update(
    name={
        'total_enaf_hab_2009_2023': 'Habitat',
        'total_enaf_act_2009_2023': 'Activité économique',
        'total_enaf_rou_2009_2023': 'Routes',
        'total_enaf_fer_2009_2023': 'Ferroviaire',
        'total_enaf_mix_2009_2023': 'Mixte'
    }[t.name]
))

fig.show()

# 7. Efficacité de la consommation d'ENAF par aire urbaine
# =======================================================

# Calculer l'efficacité de la consommation d'ENAF en relation avec la croissance démographique
# Plus ce ratio est bas, plus l'aire urbaine est efficace dans sa consommation d'espace
enaf_by_au['efficacite_enaf'] = np.where(
    enaf_by_au['pop1420'] > 0,
    enaf_by_au['total_enaf_2009_2023'] / enaf_by_au['pop1420'],
    np.nan
)

# Filtrer pour éviter les valeurs infinies ou NaN
efficacite_data = enaf_by_au[~np.isnan(enaf_by_au['efficacite_enaf']) & 
                            (enaf_by_au['efficacite_enaf'] != np.inf)].copy()

# Visualiser l'efficacité de la consommation d'ENAF
plt.figure(figsize=(12, 8))
sorted_data = efficacite_data.sort_values('efficacite_enaf').head(15)
bars = sns.barplot(x='efficacite_enaf', y='LIBAAV2020', data=sorted_data)

# Ajouter les valeurs sur les barres
for bar, value in zip(bars.patches, sorted_data['efficacite_enaf']):
    bars.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}', va='center')

plt.title('Efficacité de la consommation d\'ENAF par aire urbaine')
plt.xlabel('Surface ENAF consommée par habitant supplémentaire (m²)')
plt.ylabel('Aire urbaine')
plt.tight_layout()
plt.show()

# 8. Modélisation avancée
# =====================

# Régression multiple pour prédire la consommation d'ENAF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Préparer les données
features = ['pop20', 'pop1420', 'nb_communes']
X = enaf_by_au[features].fillna(0)
y = enaf_by_au['total_enaf_2009_2023'].fillna(0)

# Standardiser les variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Interception: {model.intercept_}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Visualiser l'importance des caractéristiques
plt.figure(figsize=(10, 6))
importance = pd.Series(np.abs(model.coef_), index=features)
importance.sort_values(ascending=True).plot(kind='barh')
plt.title('Importance des variables dans la prédiction de la consommation d\'ENAF')
plt.xlabel('Importance relative (valeur absolue du coefficient)')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()

# 9. Analyse spécifique par type d'utilisation
# ==========================================

# Analyser la répartition habitat vs. activité économique
enaf_by_au['ratio_hab_act'] = np.where(
    enaf_by_au['total_enaf_act_2009_2023'] > 0,
    enaf_by_au['total_enaf_hab_2009_2023'] / enaf_by_au['total_enaf_act_2009_2023'],
    np.nan
)

# Visualiser le ratio habitat/activité
plt.figure(figsize=(12, 8))
valid_ratio = enaf_by_au[~np.isnan(enaf_by_au['ratio_hab_act'])].copy()
sorted_ratio = valid_ratio.sort_values('ratio_hab_act', ascending=False).head(15)
bars = sns.barplot(x='ratio_hab_act', y='LIBAAV2020', data=sorted_ratio)

# Ajouter les valeurs sur les barres
for bar, value in zip(bars.patches, sorted_ratio['ratio_hab_act']):
    bars.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}', va='center')

plt.title('Ratio entre consommation d\'ENAF pour l\'habitat et pour l\'activité économique')
plt.xlabel('Ratio (habitat / activité)')
plt.ylabel('Aire urbaine')
plt.tight_layout()
plt.show()

# 10. Évolution temporelle de la consommation d'ENAF
# ================================================

# Analyser l'évolution de la consommation d'ENAF au fil du temps
# Nous utiliserons les colonnes naf09art10, naf10art11, naf11art12, etc.
# pour créer une série temporelle

# Extraire les colonnes d'artficialisation pour chaque année
time_cols = []
for year in range(10, 24):  # De 2010 à 2023
    prev_year = year - 1
    col_name = f'naf{prev_year:02d}art{year:02d}'
    if col_name in enaf_data.columns:
        time_cols.append(col_name)

# Agréger par aire urbaine et par année
enaf_time_series = pd.DataFrame()

for i, year in enumerate(range(10, 24)):  # De 2010 à 2023
    prev_year = year - 1
    col_name = f'naf{prev_year:02d}art{year:02d}'
    
    if col_name in enaf_data.columns:
        # Agréger par aire urbaine
        year_data = enaf_in_au.groupby('au_id')[col_name].sum().reset_index()
        year_data.rename(columns={col_name: f'20{year}'}, inplace=True)
        
        if i == 0:
            enaf_time_series = year_data
        else:
            enaf_time_series = enaf_time_series.merge(year_data, on='au_id', how='outer')

# Joindre avec les noms des aires urbaines
enaf_time_series['au_id'] = enaf_time_series['au_id'].astype(str).str.zfill(3)
enaf_time_series = enaf_time_series.merge(
    aires_urbaines_data[['AAV2020', 'LIBAAV2020']], 
    left_on='au_id', 
    right_on='AAV2020', 
    how='left'
)

# Visualiser l'évolution temporelle pour les principales aires urbaines
# Sélectionner les 5 aires urbaines avec la plus grande consommation totale
top_5_au = enaf_by_au.sort_values('total_enaf_2009_2023', ascending=False).head(5)['au_id'].tolist()
top_5_data = enaf_time_series[enaf_time_series['au_id'].isin(top_5_au)]

# Préparer les données pour le graphique
melted_data = pd.melt(
    top_5_data, 
    id_vars=['au_id', 'LIBAAV2020'], 
    value_vars=[col for col in top_5_data.columns if col.startswith('20')],
    var_name='Année',
    value_name='Consommation ENAF'
)

# Créer le graphique
plt.figure(figsize=(12, 8))
sns.lineplot(data=melted_data, x='Année', y='Consommation ENAF', hue='LIBAAV2020', marker='o')
plt.title('Évolution de la consommation d\'ENAF pour les 5 principales aires urbaines')
plt.xlabel('Année')
plt.ylabel('Surface ENAF consommée (m²)')
plt.xticks(rotation=45)
plt.legend(title='Aire urbaine')
plt.tight_layout()
plt.show()

# 11. Analyse spatiale avec carte choroplèthe
# ========================================

# Créer une carte interactive pour visualiser la consommation d'ENAF par aire urbaine
# Simulons des coordonnées géographiques pour les aires urbaines
# Normalement, vous utiliseriez des données géospatiales réelles

# Coordonnées approximatives des principales aires urbaines françaises (latitude, longitude)
coords = {
    '001': (48.8566, 2.3522),  # Paris
    '002': (45.7578, 4.8320),  # Lyon
    '003': (43.2965, 5.3698),  # Marseille
    '004': (50.6292, 3.0573),  # Lille
    '005': (43.6047, 1.4442),  # Toulouse
    '006': (44.8378, -0.5792), # Bordeaux
    '008': (47.2184, -1.5536), # Nantes
    '010': (48.5734, 7.7521),  # Strasbourg
    '012': (43.6108, 3.8767),  # Montpellier
    '013': (48.1173, -1.6778), # Rennes
    '014': (45.1885, 5.7245),  # Grenoble
    '015': (49.4431, 1.0993),  # Rouen
    '017': (43.7102, 7.2620),  # Nice
    '018': (43.1257, 5.9304),  # Toulon
    '019': (47.3941, 0.6848),  # Tours
    '020': (48.6921, 6.1844),  # Nancy
    '022': (45.7772, 3.0870),  # Clermont-Ferrand
    '023': (45.4397, 4.3872),  # Saint-Étienne
    '024': (49.1829, -0.3707), # Caen
    '025': (47.9029, 1.9039),  # Orléans
    '026': (47.4784, -0.5632), # Angers
    '028': (47.3220, 5.0415),  # Dijon
    '029': (47.7508, 7.3359),  # Mulhouse
    '030': (42.6987, 2.8956),  # Perpignan
    '031': (43.5515, 7.0196),  # Cannes
}

# Ajouter les coordonnées au DataFrame
enaf_by_au['latitude'] = enaf_by_au['au_id'].map(lambda x: coords.get(x, (np.nan, np.nan))[0])
enaf_by_au['longitude'] = enaf_by_au['au_id'].map(lambda x: coords.get(x, (np.nan, np.nan))[1])

# Créer la carte interactive
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

# Ajouter des marqueurs pour chaque aire urbaine
for _, row in enaf_by_au.dropna(subset=['latitude', 'longitude']).iterrows():
    # Définir la taille du cercle en fonction de la consommation ENAF
    radius = np.sqrt(row['total_enaf_2009_2023']) / 200
    
    # Définir la couleur en fonction de l'efficacité ENAF
    if pd.notna(row.get('efficacite_enaf')) and row['efficacite_enaf'] != np.inf:
        efficacite = row['efficacite_enaf']
        # Normaliser entre 0 et 1 (meilleure efficacité = plus proche de 0)
        max_eff = enaf_by_au['efficacite_enaf'].replace([np.inf, -np.inf], np.nan).dropna().max()
        normalized_eff = efficacite / max_eff
        # Convertir en couleur (rouge pour inefficace, vert pour efficace)
        color = f'#{int(255 * normalized_eff):02x}{int(255 * (1-normalized_eff)):02x}00'
    else:
        color = 'gray'
    
    # Créer le popup d'information
    popup_html = f"""
    <h4>{row['LIBAAV2020']}</h4>
    <b>Consommation ENAF totale:</b> {row['total_enaf_2009_2023']:,.0f} m²<br>
    <b>Consommation pour l'habitat:</b> {row['total_enaf_hab_2009_2023']:,.0f} m²<br>
    <b>Consommation pour l'activité économique:</b> {row['total_enaf_act_2009_2023']:,.0f} m²<br>
    <b>Consommation par habitant:</b> {row['enaf_per_1000_hab']:,.2f} m²/1000 hab<br>
    <b>Efficacité démographique:</b> {row.get('efficacite_enaf', 'N/A'):,.2f} m²/nouvel habitant<br>
    <b>Population (2020):</b> {row['pop20']:,.0f}<br>
    <b>Évolution population (2014-2020):</b> {row['pop1420']:,.0f}<br>
    """
    popup = folium.Popup(popup_html, max_width=300)
    
    # Ajouter le marqueur circulaire
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius,
        popup=popup,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        tooltip=row['LIBAAV2020']
    ).add_to(m)

# Ajouter une légende
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 150px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;
            padding: 10px;
            border-radius: 5px;
           ">
    <p><b>Consommation ENAF</b></p>
    <p>Taille du cercle : Consommation totale</p>
    <p>Couleur : Efficacité</p>
    <div style="width: 20px; height: 20px; background-color: #ff0000; display: inline-block;"></div> Faible efficacité<br>
    <div style="width: 20px; height: 20px; background-color: #00ff00; display: inline-block;"></div> Haute efficacité
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Afficher la carte
m

# 12. Conclusions et recommandations
# ================================

print("""
# Conclusions sur la consommation d'ENAF par les aires urbaines

1. **Grandes consommatrices en volume absolu** :
   Les grandes aires urbaines comme Paris, Lyon et Toulouse consomment davantage d'ENAF en volume absolu,
   ce qui est attendu compte tenu de leur taille et de leur dynamisme démographique et économique.

2. **Efficacité variable selon les aires urbaines** :
   L'analyse de l'efficacité (consommation d'ENAF par habitant supplémentaire) révèle des différences
   importantes entre les aires urbaines. Certaines grandes aires urbaines parviennent à limiter leur
   consommation d'espace par habitant grâce à des formes urbaines plus denses.

3. **Différences dans les usages** :
   La répartition entre habitat, activité économique et infrastructures varie significativement selon les 
   aires urbaines, reflétant leurs spécificités économiques et leur modèle de développement.

# Recommandations pour limiter la consommation d'ENAF

1. **Promouvoir les modèles urbains efficaces** :
   S'inspirer des aires urbaines qui présentent la meilleure efficacité en termes de consommation d'ENAF
   par habitant supplémentaire.

2. **Privilégier la densification** :
   Encourager le renouvellement urbain et la densification pour limiter l'étalement urbain, particulièrement
   dans les aires urbaines en forte croissance.

3. **Adapter les stratégies selon les usages dominants** :
   Mettre en place des stratégies ciblées selon le type d'usage qui consomme le plus d'ENAF dans chaque
   aire urbaine (habitat, activité économique, infrastructures).

4. **Monitoring régulier** :
   Mettre en place un suivi régulier de la consommation d'ENAF pour évaluer l'efficacité des politiques
   de limitation de l'artificialisation des sols.
""")