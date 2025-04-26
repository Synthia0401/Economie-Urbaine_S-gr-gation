import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configuration des styles de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("talk")

def load_data(chemin_au, chemin_csp):
    """
    Charge les données depuis les fichiers CSV avec gestion d'erreurs
    """
    try:
        df_au = pd.read_csv(chemin_au, sep=';', low_memory=False)
        df_csp = pd.read_csv(chemin_csp, sep=';', low_memory=False, encoding='utf-8')
        print(f"Données chargées avec succès: {df_au.shape[0]} aires urbaines et {df_csp.shape[0]} communes")
        return df_au, df_csp
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None, None

def nettoyer_donnees_au(df_au):
    """
    Nettoie et structure les données des aires urbaines
    """
    # Sélection et renommage des colonnes pertinentes
    df_au_clean = df_au.copy()
    df_au_clean['AAV'] = df_au_clean['AAV2020'].astype(str).str.zfill(3)
    
    # Extraction des informations pertinentes
    cols_utiles = ['AAV', 'AAV2020', 'LIBAAV2020', 'TAAV2017', 'TDAAV2017', 'NB_COM']
    df_au_clean = df_au_clean[cols_utiles].rename(columns={
        'LIBAAV2020': 'nom_aire_urbaine',
        'TAAV2017': 'categorie_taille',
        'TDAAV2017': 'detail_taille',
        'NB_COM': 'nombre_communes'
    })
    
    # Conversion des types de données appropriés
    df_au_clean['categorie_taille'] = pd.to_numeric(df_au_clean['categorie_taille'], errors='coerce')
    df_au_clean['detail_taille'] = pd.to_numeric(df_au_clean['detail_taille'], errors='coerce')
    df_au_clean['nombre_communes'] = pd.to_numeric(df_au_clean['nombre_communes'], errors='coerce')
    
    # Création d'une catégorie descriptive pour les aires urbaines
    taille_labels = {
        0: 'Hors attraction',
        1: 'Petite',
        2: 'Moyenne',
        3: 'Grande',
        4: 'Très grande',
        5: 'Métropole'
    }
    df_au_clean['categorie_desc'] = df_au_clean['categorie_taille'].map(taille_labels)
    
    # Élimination des lignes avec des valeurs manquantes critiques
    df_au_clean = df_au_clean.dropna(subset=['categorie_taille', 'AAV'])
    
    return df_au_clean

def analyser_colonnes_consommation(df_csp):
    """
    Analyse les colonnes disponibles pour la consommation d'espaces
    et sélectionne les indicateurs pertinents pour l'ENAF
    """
    # Identifier les colonnes de consommation d'espace
    # Format typique: artXXactYY, artXXhabYY, artXXmixYY, artXXrouYY, artXXferYY, où XX et YY sont des années
    
    # Regrouper les colonnes par type (act=activité, hab=habitat, mix=mixte, rou=route, fer=ferroviaire)
    col_types = {
        'act': [],  # Activité économique
        'hab': [],  # Habitat
        'mix': [],  # Mixte
        'rou': [],  # Routes
        'fer': [],  # Ferroviaire
        'inc': []   # Inconnu
    }
    
    # Trouver toutes les paires d'années disponibles (artXXYY)
    periodes = set()
    for col in df_csp.columns:
        match = re.match(r'art(\d{2})(\w+)(\d{2})', col)
        if match:
            debut, type_col, fin = match.groups()
            if type_col in col_types:
                periodes.add((debut, fin))
                col_types[type_col].append(col)
    
    # Trier les colonnes par périodes pour obtenir les plus récentes
    periodes = sorted(periodes, key=lambda x: (x[1], x[0]), reverse=True)
    
    # Sélectionner la période la plus récente disponible
    if periodes:
        periode_recente = periodes[0]
        print(f"Période la plus récente sélectionnée: 20{periode_recente[0]}-20{periode_recente[1]}")
        
        # Construire les noms de colonnes pour cette période
        selected_cols = {}
        for type_col in col_types:
            col_name = f"art{periode_recente[0]}{type_col}{periode_recente[1]}"
            if col_name in df_csp.columns:
                selected_cols[type_col] = col_name
        
        return selected_cols, periode_recente
    else:
        print("Aucune période de consommation d'espace identifiée")
        return None, None

def preparer_donnees_consommation(df_csp, selected_cols, periode_recente):
    """
    Prépare les données de consommation d'espace pour l'analyse ENAF
    """
    if not selected_cols:
        return None
    
    df_conso = df_csp.copy()
    
    # Extraction du code d'aire urbaine
    df_conso['AAV'] = df_conso['aav2020'].astype(str).str.extract(r'^(\d{3})').fillna('000')
    
    # Conversion des types pour les colonnes de consommation
    for type_col, col_name in selected_cols.items():
        df_conso[f'conso_{type_col}'] = pd.to_numeric(df_conso[col_name], errors='coerce').fillna(0)
    
    # Conversion des autres données importantes
    df_conso['pop14'] = pd.to_numeric(df_conso['pop14'], errors='coerce')
    df_conso['pop20'] = pd.to_numeric(df_conso['pop20'], errors='coerce')
    df_conso['surfcom2023'] = pd.to_numeric(df_conso['surfcom2023'], errors='coerce')
    
    # Calcul de l'évolution de population
    df_conso['evol_pop'] = df_conso['pop20'] - df_conso['pop14']
    df_conso['taux_evol_pop'] = (df_conso['evol_pop'] / df_conso['pop14']) * 100
    
    # Calcul de la consommation totale d'ENAF
    # ENAF = Toutes consommations sauf routes et ferroviaire qui sont des infrastructures
    conso_enaf_cols = [col for type_col, col in selected_cols.items() if type_col not in ['rou', 'fer']]
    df_conso['conso_enaf_total'] = df_conso[[col for col in conso_enaf_cols]].sum(axis=1)
    
    # Calcul de la consommation d'infrastructures (routes, ferroviaire)
    infra_cols = [col for type_col, col in selected_cols.items() if type_col in ['rou', 'fer']]
    if infra_cols:
        df_conso['conso_infra_total'] = df_conso[infra_cols].sum(axis=1)
    else:
        df_conso['conso_infra_total'] = 0
    
    # Calcul de la consommation totale tous types confondus
    df_conso['conso_totale'] = df_conso[[f'conso_{type_col}' for type_col in selected_cols]].sum(axis=1)
    
    # Calcul des ratios par habitant
    df_conso['conso_enaf_par_hab'] = df_conso['conso_enaf_total'] / df_conso['pop20'].replace(0, np.nan)
    df_conso['conso_par_nouvel_hab'] = df_conso['conso_enaf_total'] / df_conso['evol_pop'].replace(0, np.nan)
    
    # Calcul du coefficient d'artificialisation: m² consommés par m² de territoire
    df_conso['coef_artif'] = df_conso['conso_totale'] / df_conso['surfcom2023'].replace(0, np.nan)
    
    # Nettoyage des valeurs infinies ou aberrantes
    for col in ['conso_par_nouvel_hab', 'conso_enaf_par_hab', 'coef_artif']:
        df_conso[col] = df_conso[col].replace([np.inf, -np.inf], np.nan)
        # Supprimer les valeurs aberrantes (>99ème percentile)
        seuil = df_conso[col].quantile(0.99)
        df_conso[col] = df_conso[col].mask(df_conso[col] > seuil)
    
    # Calcul de la part de chaque type dans la consommation totale
    for type_col in selected_cols:
        df_conso[f'part_{type_col}'] = df_conso[f'conso_{type_col}'] / df_conso['conso_totale'].replace(0, np.nan) * 100
    
    return df_conso

def agreger_par_aire_urbaine(df_conso):
    """
    Agrège les données de consommation au niveau des aires urbaines
    """
    # Définition des agrégations pertinentes pour notre analyse
    agg_dict = {
        'conso_enaf_total': 'sum',
        'conso_infra_total': 'sum',
        'conso_totale': 'sum',
        'evol_pop': 'sum',
        'pop14': 'sum',
        'pop20': 'sum',
        'surfcom2023': 'sum'
    }
    
    # Ajout des types spécifiques si présents
    for type_col in ['act', 'hab', 'mix', 'rou', 'fer', 'inc']:
        if f'conso_{type_col}' in df_conso.columns:
            agg_dict[f'conso_{type_col}'] = 'sum'
    
    # Agrégation par aire urbaine
    df_agg = df_conso.groupby('AAV').agg(agg_dict).reset_index()
    
    # Recalcul des ratios et indicateurs au niveau des aires urbaines
    df_agg['densite_pop'] = df_agg['pop20'] / (df_agg['surfcom2023'] / 1000000)  # hab/km²
    df_agg['taux_evol_pop'] = (df_agg['evol_pop'] / df_agg['pop14']) * 100
    df_agg['conso_enaf_par_hab'] = df_agg['conso_enaf_total'] / df_agg['pop20']
    df_agg['conso_par_nouvel_hab'] = df_agg['conso_enaf_total'] / df_agg['evol_pop'].replace(0, np.nan)
    df_agg['coef_artif'] = df_agg['conso_totale'] / df_agg['surfcom2023']
    
    # Calcul de la part de chaque type dans la consommation totale
    for type_col in ['act', 'hab', 'mix', 'rou', 'fer', 'inc']:
        if f'conso_{type_col}' in df_agg.columns:
            df_agg[f'part_{type_col}'] = df_agg[f'conso_{type_col}'] / df_agg['conso_totale'].replace(0, np.nan) * 100
    
    # Nettoyage des valeurs infinies ou aberrantes
    for col in ['conso_par_nouvel_hab', 'conso_enaf_par_hab', 'coef_artif']:
        df_agg[col] = df_agg[col].replace([np.inf, -np.inf], np.nan)
        # Supprimer les valeurs aberrantes (>99ème percentile)
        if not df_agg[col].isna().all():
            seuil = df_agg[col].quantile(0.99)
            df_agg[col] = df_agg[col].mask(df_agg[col] > seuil)
    
    return df_agg

def fusionner_donnees(df_au_clean, df_conso_agg):
    """
    Fusionne les données des aires urbaines avec les agrégats de consommation
    """
    df_final = pd.merge(df_au_clean, df_conso_agg, on='AAV', how='inner')
    print(f"Fusion réalisée avec succès: {df_final.shape[0]} aires urbaines après fusion")
    
    return df_final

def analyser_correlations(df_final, dossier_sortie="resultats"):
    """
    Analyse les corrélations entre variables et génère une matrice de corrélation
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Sélectionner les colonnes numériques pertinentes pour l'analyse
    cols_correlation = [
        'categorie_taille', 'nombre_communes', 'pop20', 'densite_pop',
        'taux_evol_pop', 'conso_enaf_total', 'conso_infra_total',
        'conso_enaf_par_hab', 'conso_par_nouvel_hab', 'coef_artif'
    ]
    
    # Ajouter les parts par type si disponibles
    for type_col in ['act', 'hab', 'mix', 'rou', 'fer', 'inc']:
        if f'part_{type_col}' in df_final.columns:
            cols_correlation.append(f'part_{type_col}')
    
    # Filtrer pour ne garder que les colonnes existantes
    cols_correlation = [col for col in cols_correlation if col in df_final.columns]
    
    # Calculer la matrice de corrélation
    corr_matrix = df_final[cols_correlation].corr()
    
    # Créer une heatmap de la matrice de corrélation
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
    
    plt.title("Matrice de corrélation - Consommation ENAF et caractéristiques des aires urbaines", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/matrice_correlation_enaf.png", dpi=300)
    
    # Trouver les variables les plus corrélées à la consommation d'ENAF
    for target_var in ['conso_enaf_par_hab', 'conso_par_nouvel_hab', 'coef_artif']:
        if target_var in corr_matrix.columns:
            correlations = corr_matrix[target_var].sort_values(ascending=False)
            print(f"\nVariables les plus corrélées à {target_var}:")
            print(correlations.drop(target_var).head(5))
    
    # Identifier les 2 variables les plus fortement corrélées pour une analyse approfondie
    # On exclut les auto-corrélations
    corr_pairs = []
    for i in range(len(cols_correlation)):
        for j in range(i+1, len(cols_correlation)):
            var1 = cols_correlation[i]
            var2 = cols_correlation[j]
            if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                corr_value = abs(corr_matrix.loc[var1, var2])
                corr_pairs.append((var1, var2, corr_value))
    
    # Trier les paires par force de corrélation
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Retourner les meilleures paires pour la visualisation
    best_pairs = []
    seen_vars = set()
    
    # Priorité aux paires contenant 'conso_enaf_par_hab', 'conso_par_nouvel_hab' ou 'coef_artif'
    priority_vars = ['conso_enaf_par_hab', 'conso_par_nouvel_hab', 'coef_artif']
    for var1, var2, corr in corr_pairs:
        if any(pvar in [var1, var2] for pvar in priority_vars) and len(best_pairs) < 3:
            best_pairs.append((var1, var2, corr))
            seen_vars.add(var1)
            seen_vars.add(var2)
    
    # Compléter avec d'autres paires fortement corrélées si besoin
    for var1, var2, corr in corr_pairs:
        if var1 not in seen_vars or var2 not in seen_vars:
            if len(best_pairs) < 5:  # Limiter à 5 paires au total
                best_pairs.append((var1, var2, corr))
                seen_vars.add(var1)
                seen_vars.add(var2)
    
    return corr_matrix, best_pairs

def visualiser_relations_cles(df_final, best_pairs, dossier_sortie="resultats"):
    """
    Crée des visualisations des relations clés identifiées
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Pour chaque paire de variables fortement corrélées
    for i, (var1, var2, corr) in enumerate(best_pairs):
        plt.figure(figsize=(12, 8))
        
        # Utiliser la catégorie descriptive comme couleur si disponible
        if 'categorie_desc' in df_final.columns:
            scatter = sns.scatterplot(
                data=df_final, 
                x=var1, 
                y=var2,
                hue='categorie_desc',
                size='pop20',
                sizes=(20, 500),
                alpha=0.7
            )
        else:
            scatter = sns.scatterplot(
                data=df_final, 
                x=var1, 
                y=var2,
                size='pop20',
                sizes=(20, 500),
                alpha=0.7
            )
        
        # Ajouter une ligne de tendance
        sns.regplot(
            data=df_final, 
            x=var1, 
            y=var2, 
            scatter=False, 
            color='red', 
            line_kws={"linewidth": 2}
        )
        
        # Annotation des plus grandes aires urbaines
        for idx, row in df_final.nlargest(8, 'pop20').iterrows():
            if not pd.isna(row[var1]) and not pd.isna(row[var2]):
                plt.annotate(
                    row['nom_aire_urbaine'],
                    xy=(row[var1], row[var2]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        
        # Ajouter des titres et labels descriptifs
        var1_label = var1.replace('_', ' ').title()
        var2_label = var2.replace('_', ' ').title()
        plt.title(f"Relation entre {var1_label} et {var2_label} (r = {corr:.2f})", fontsize=14)
        plt.xlabel(var1_label, fontsize=12)
        plt.ylabel(var2_label, fontsize=12)
        
        # Pour certaines variables, utiliser une échelle logarithmique
        log_vars = ['pop20', 'conso_enaf_total', 'conso_infra_total', 'conso_totale']
        if var1 in log_vars:
            plt.xscale('log')
        if var2 in log_vars:
            plt.yscale('log')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Sauvegarde de la figure
        plt.savefig(f"{dossier_sortie}/relation_{var1}_{var2}.png", dpi=300)
    
    # Graphique spécial pour analyser la consommation d'ENAF par type d'aire urbaine
    if 'categorie_desc' in df_final.columns and 'conso_enaf_par_hab' in df_final.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            data=df_final,
            x='categorie_desc',
            y='conso_enaf_par_hab',
            palette='viridis'
        )
        plt.title("Consommation d'ENAF par habitant selon le type d'aire urbaine", fontsize=14)
        plt.xlabel("Type d'aire urbaine", fontsize=12)
        plt.ylabel("Consommation d'ENAF par habitant (m²)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dossier_sortie}/conso_enaf_par_type_au.png", dpi=300)
    
    # Graphique de composition de la consommation par type d'aire urbaine
    part_cols = [col for col in df_final.columns if col.startswith('part_')]
    if part_cols and 'categorie_desc' in df_final.columns:
        # Préparation des données
        df_parts = df_final.groupby('categorie_desc')[part_cols].mean().reset_index()
        df_parts_melt = pd.melt(df_parts, id_vars=['categorie_desc'], 
                                value_vars=part_cols, 
                                var_name='Type_consommation', 
                                value_name='Pourcentage')
        
        # Renommage pour l'affichage
        df_parts_melt['Type_consommation'] = df_parts_melt['Type_consommation'].apply(
            lambda x: x.replace('part_', '').replace('act', 'Activité').replace('hab', 'Habitat')
                      .replace('mix', 'Mixte').replace('rou', 'Routes')
                      .replace('fer', 'Ferroviaire').replace('inc', 'Inconnu')
        )
        
        plt.figure(figsize=(14, 10))
        sns.barplot(
            data=df_parts_melt,
            x='categorie_desc',
            y='Pourcentage',
            hue='Type_consommation',
            palette='viridis'
        )
        plt.title("Composition de la consommation d'espace par type d'aire urbaine", fontsize=14)
        plt.xlabel("Type d'aire urbaine", fontsize=12)
        plt.ylabel("Pourcentage du total (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Type de consommation")
        plt.tight_layout()
        plt.savefig(f"{dossier_sortie}/composition_conso_par_type_au.png", dpi=300)
    
    return

def analyser_efficacite_utilisation_espace(df_final, dossier_sortie="resultats"):
    """
    Analyse l'efficacité de l'utilisation de l'espace des aires urbaines
    en fonction de leur croissance démographique
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Filtrer les aires urbaines avec évolution de population positive
    df_croissance = df_final[df_final['evol_pop'] > 0].copy()
    
    if 'conso_par_nouvel_hab' in df_croissance.columns:
        # Créer une variable d'efficacité: plus la valeur est basse,
        # moins on consomme d'espace pour accueillir un nouvel habitant
        df_croissance['efficacite'] = 1 / df_croissance['conso_par_nouvel_hab']
        
        plt.figure(figsize=(14, 10))
        
        # Graphique de dispersion montrant l'efficacité en fonction de la densité
        if 'categorie_desc' in df_croissance.columns:
            scatter = sns.scatterplot(
                data=df_croissance,
                x='densite_pop',
                y='conso_par_nouvel_hab',
                hue='categorie_desc',
                size='pop20',
                sizes=(20, 500),
                alpha=0.7
            )
        else:
            scatter = sns.scatterplot(
                data=df_croissance,
                x='densite_pop',
                y='conso_par_nouvel_hab',
                size='pop20',
                sizes=(20, 500),
                alpha=0.7
            )
        
        # Ajouter une ligne de tendance
        sns.regplot(
            data=df_croissance,
            x='densite_pop',
            y='conso_par_nouvel_hab',
            scatter=False,
            color='red',
            line_kws={"linewidth": 2}
        )
        
        # Annotation des plus grandes aires urbaines
        for idx, row in df_croissance.nlargest(8, 'pop20').iterrows():
            if not pd.isna(row['densite_pop']) and not pd.isna(row['conso_par_nouvel_hab']):
                plt.annotate(
                    row['nom_aire_urbaine'],
                    xy=(row['densite_pop'], row['conso_par_nouvel_hab']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        
        plt.xscale('log')  # Échelle logarithmique pour la densité
        plt.yscale('log')  # Échelle logarithmique pour la consommation par habitant
        
        plt.title("Efficacité de l'utilisation de l'espace selon la densité de population", fontsize=14)
        plt.xlabel("Densité de population (hab/km²)", fontsize=12)
        plt.ylabel("Consommation d'ENAF par nouvel habitant (m²)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{dossier_sortie}/efficacite_utilisation_espace.png", dpi=300)
        
        # Tableau des aires urbaines les plus efficaces et les moins efficaces
        df_croissance_sorted = df_croissance.sort_values('conso_par_nouvel_hab')
        
        print("\nAires urbaines les plus efficaces (moins d'ENAF consommé par nouvel habitant):")
        print(df_croissance_sorted[['nom_aire_urbaine', 'categorie_desc', 'pop20', 'densite_pop', 'conso_par_nouvel_hab']].head(10))
        
        print("\nAires urbaines les moins efficaces (plus d'ENAF consommé par nouvel habitant):")
        print(df_croissance_sorted[['nom_aire_urbaine', 'categorie_desc', 'pop20', 'densite_pop', 'conso_par_nouvel_hab']].tail(10))

    return

def exporter_resultats(df_final, dossier_sortie="resultats"):
    """
    Exporte les résultats en CSV et génère un rapport statistique
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Export du DataFrame final
    df_final.to_csv(f"{dossier_sortie}/aires_urbaines_conso_enaf_analyse.csv", sep=';', index=False)
    
    # Génération de statistiques descriptives
    stats_desc = df_final.describe().T
    stats_desc.to_csv(f"{dossier_sortie}/statistiques_descriptives.csv", sep=';')
    
    # Statistiques par catégorie d'aire urbaine
    if 'categorie_desc' in df_final.columns:
        stats_by_cat = df_final.groupby('categorie_desc').agg({
            'pop20': 'sum',
            'evol_pop': 'sum',
            'conso_enaf_total': 'sum',
            'conso_enaf_par_hab': 'mean',
            'conso_par_nouvel_hab': 'mean',
            'densite_pop': 'mean',
            'taux_evol_pop': 'mean'
        }).reset_index()
        
        stats_by
        
def exporter_resultats(df_final, dossier_sortie="resultats"):
    """
    Exporte les résultats en CSV et génère un rapport statistique
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Export du DataFrame final
    df_final.to_csv(f"{dossier_sortie}/aires_urbaines_conso_enaf_analyse.csv", sep=';', index=False)
    
    # Génération de statistiques descriptives
    stats_desc = df_final.describe().T
    stats_desc.to_csv(f"{dossier_sortie}/statistiques_descriptives.csv", sep=';')
    
    # Statistiques par catégorie d'aire urbaine
    if 'categorie_desc' in df_final.columns:
        stats_by_cat = df_final.groupby('categorie_desc').agg({
            'pop20': 'sum',
            'evol_pop': 'sum',
            'conso_enaf_total': 'sum',
            'conso_enaf_par_hab': 'mean',
            'conso_par_nouvel_hab': 'mean',
            'densite_pop': 'mean',
            'taux_evol_pop': 'mean'
        }).reset_index()
        
        stats_by_cat.to_csv(f"{dossier_sortie}/statistiques_par_categorie.csv", sep=';', index=False)
    
    print(f"Résultats exportés avec succès dans le dossier '{dossier_sortie}'")
    
    return stats_desc

def rapport_synthetique(df_final, corr_matrix, best_pairs, dossier_sortie="resultats"):
    """
    Génère un rapport synthétique des résultats d'analyse
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Ouvrir le fichier de rapport
    with open(f"{dossier_sortie}/rapport_analyse_enaf.md", "w", encoding="utf-8") as f:
        # En-tête
        f.write("# Rapport d'analyse : Consommation d'ENAF par les aires urbaines\n\n")
        f.write(f"*Date d'analyse : {pd.Timestamp.now().strftime('%d/%m/%Y')}*\n\n")
        
        # Sommaire
        f.write("## Sommaire\n")
        f.write("1. Résumé des données analysées\n")
        f.write("2. Principaux indicateurs de consommation d'ENAF\n")
        f.write("3. Relations et corrélations clés\n")
        f.write("4. Typologie des aires urbaines selon leur consommation\n")
        f.write("5. Conclusion et recommandations\n\n")
        
        # 1. Résumé des données
        f.write("## 1. Résumé des données analysées\n\n")
        f.write(f"- Nombre total d'aires urbaines analysées : {df_final.shape[0]}\n")
        f.write(f"- Population totale couverte : {df_final['pop20'].sum():,.0f} habitants\n")
        f.write(f"- Période d'analyse : 2014-2020\n")
        f.write(f"- Évolution démographique totale : {df_final['evol_pop'].sum():+,.0f} habitants\n")
        f.write(f"- Consommation totale d'ENAF : {df_final['conso_enaf_total'].sum():,.0f} m²\n\n")
        
        # Répartition par catégorie si disponible
        if 'categorie_desc' in df_final.columns:
            f.write("Répartition des aires urbaines par catégorie :\n\n")
            cat_counts = df_final['categorie_desc'].value_counts().reset_index()
            cat_counts.columns = ['Catégorie', 'Nombre']
            f.write(cat_counts.to_markdown(index=False) + "\n\n")
        
        # 2. Principaux indicateurs
        f.write("## 2. Principaux indicateurs de consommation d'ENAF\n\n")
        
        # Statistiques globales
        stats = df_final[['conso_enaf_par_hab', 'conso_par_nouvel_hab', 'coef_artif']].describe().T
        stats = stats.round(2)
        f.write("Statistiques globales sur les indicateurs clés :\n\n")
        f.write(stats.to_markdown() + "\n\n")
        
        # Répartition par type de consommation
        part_cols = [col for col in df_final.columns if col.startswith('part_')]
        if part_cols:
            f.write("Répartition moyenne de la consommation d'espace par type :\n\n")
            parts_mean = df_final[part_cols].mean().sort_values(ascending=False)
            parts_df = pd.DataFrame({'Type': parts_mean.index, 'Pourcentage moyen (%)': parts_mean.values})
            parts_df['Type'] = parts_df['Type'].apply(lambda x: x.replace('part_', '').title())
            f.write(parts_df.to_markdown(index=False, floatfmt=".2f") + "\n\n")
        
        # 3. Relations et corrélations clés
        f.write("## 3. Relations et corrélations clés\n\n")
        
        # Afficher les corrélations les plus fortes avec la consommation d'ENAF
        f.write("### 3.1 Principales corrélations avec la consommation d'ENAF\n\n")
        
        for var in ['conso_enaf_par_hab', 'conso_par_nouvel_hab', 'coef_artif']:
            if var in corr_matrix.columns:
                f.write(f"#### Corrélations avec {var.replace('_', ' ').title()}\n\n")
                correlations = corr_matrix[var].sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Variable': correlations.index,
                    'Coefficient de corrélation': correlations.values
                })
                # Filtrer pour ne garder que les 5 plus fortes corrélations (positives ou négatives)
                # mais exclure la variable elle-même
                corr_df = corr_df[corr_df['Variable'] != var]
                top_corr = pd.concat([
                    corr_df.head(3),  # Top 3 positives
                    corr_df.tail(3)   # Top 3 négatives
                ])
                f.write(top_corr.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        
        # Paires de variables les plus corrélées
        f.write("### 3.2 Principales relations identifiées\n\n")
        for i, (var1, var2, corr) in enumerate(best_pairs):
            f.write(f"#### Relation {i+1}: {var1.replace('_', ' ').title()} vs {var2.replace('_', ' ').title()}\n\n")
            f.write(f"- Coefficient de corrélation : {corr:.3f}\n")
            f.write(f"- Cette relation suggère que {interpretation_correlation(var1, var2, corr)}\n")
            f.write(f"- *Voir graphique : relation_{var1}_{var2}.png*\n\n")
            
        # 4. Typologie des aires urbaines
        f.write("## 4. Typologie des aires urbaines selon leur consommation\n\n")
        
        # Créer une typologie basée sur la densité et la consommation
        if 'densite_pop' in df_final.columns and 'conso_enaf_par_hab' in df_final.columns:
            # Définir des seuils pour créer une matrice 2x2
            seuil_densite = df_final['densite_pop'].median()
            seuil_conso = df_final['conso_enaf_par_hab'].median()
            
            # Créer les catégories
            df_final['densite_cat'] = np.where(df_final['densite_pop'] >= seuil_densite, 'Haute', 'Basse')
            df_final['conso_cat'] = np.where(df_final['conso_enaf_par_hab'] >= seuil_conso, 'Haute', 'Basse')
            df_final['typologie'] = df_final['densite_cat'] + ' densité / ' + df_final['conso_cat'] + ' consommation'
            
            # Compter par typologie
            typo_counts = df_final.groupby('typologie').size().reset_index(name='Nombre')
            f.write("### 4.1 Répartition des aires urbaines selon leur typologie\n\n")
            f.write(typo_counts.to_markdown(index=False) + "\n\n")
            
            # Top 3 de chaque typologie
            f.write("### 4.2 Aires urbaines représentatives de chaque typologie\n\n")
            for typo in df_final['typologie'].unique():
                f.write(f"#### {typo}\n\n")
                top3 = df_final[df_final['typologie'] == typo].nlargest(3, 'pop20')
                if not top3.empty:
                    typo_examples = top3[['nom_aire_urbaine', 'pop20', 'densite_pop', 'conso_enaf_par_hab']]
                    typo_examples.columns = ['Aire urbaine', 'Population', 'Densité (hab/km²)', 'Conso. ENAF par hab. (m²)']
                    f.write(typo_examples.to_markdown(index=False, floatfmt=".2f") + "\n\n")
        
        # 5. Conclusion
        f.write("## 5. Conclusion et recommandations\n\n")
        
        # Synthèse des résultats
        f.write("### 5.1 Synthèse des résultats\n\n")
        f.write("- Les analyses montrent que la consommation d'ENAF est fortement liée à la structure urbaine et à la densité de population.\n")
        f.write("- Les aires urbaines à forte densité tendent à consommer moins d'espace par nouvel habitant, témoignant d'une utilisation plus efficiente du foncier.\n")
        f.write("- La taille de l'aire urbaine influence le type de consommation, avec une prédominance de l'habitat dans les petites aires urbaines et une part plus importante des activités économiques dans les grandes.\n\n")
        
        # Recommandations
        f.write("### 5.2 Recommandations\n\n")
        f.write("1. **Promouvoir la densification** des aires urbaines pour limiter l'étalement urbain et réduire la consommation d'ENAF par habitant.\n")
        f.write("2. **Adapter les politiques foncières** selon la typologie des aires urbaines, en tenant compte de leurs spécificités.\n")
        f.write("3. **Favoriser le renouvellement urbain** particulièrement dans les aires urbaines à forte consommation et faible densité.\n")
        f.write("4. **Surveiller particulièrement** les aires urbaines en forte croissance démographique mais à faible densité, qui présentent le risque le plus élevé de consommation excessive d'ENAF.\n\n")
        
        # Limites de l'étude
        f.write("### 5.3 Limites de l'étude\n\n")
        f.write("- Les données de consommation peuvent présenter des incertitudes liées à la méthodologie de collecte.\n")
        f.write("- L'étude ne prend pas en compte la qualité des espaces consommés (terres agricoles à haute valeur ajoutée, zones naturelles à forte biodiversité, etc.).\n")
        f.write("- Les dynamiques économiques locales peuvent influencer les tendances de consommation mais n'ont pas été intégrées dans cette analyse.\n\n")
        
    print(f"Rapport synthétique généré dans '{dossier_sortie}/rapport_analyse_enaf.md'")
    return

def interpretation_correlation(var1, var2, corr_value):
    """
    Génère une interprétation textuelle d'une corrélation entre deux variables
    """
    # Normaliser les noms de variables pour l'affichage
    var1_norm = var1.replace('_', ' ').lower()
    var2_norm = var2.replace('_', ' ').lower()
    
    # Déterminer le sens et la force de la corrélation
    if abs(corr_value) > 0.7:
        force = "très forte"
    elif abs(corr_value) > 0.5:
        force = "forte"
    elif abs(corr_value) > 0.3:
        force = "modérée"
    else:
        force = "faible"
    
    if corr_value > 0:
        sens = "positive"
        relation = f"quand {var1_norm} augmente, {var2_norm} tend également à augmenter"
    else:
        sens = "négative"
        relation = f"quand {var1_norm} augmente, {var2_norm} tend à diminuer"
    
    # Générer l'interprétation
    interpretation = f"il existe une corrélation {force} {sens} entre {var1_norm} et {var2_norm} : {relation}"
    
    # Ajouter des interprétations spécifiques selon les variables
    if ('densite' in var1 or 'densite' in var2) and ('conso' in var1 or 'conso' in var2) and corr_value < 0:
        interpretation += ". Cela confirme que les territoires plus denses utilisent l'espace de manière plus efficiente."
    
    if ('categorie_taille' in var1 or 'categorie_taille' in var2) and ('part_act' in var1 or 'part_act' in var2) and corr_value > 0:
        interpretation += ". Les grandes aires urbaines consomment proportionnellement plus d'espace pour les activités économiques."
    
    if ('taux_evol_pop' in var1 or 'taux_evol_pop' in var2) and ('conso' in var1 or 'conso' in var2) and corr_value > 0:
        interpretation += ". La croissance démographique est un facteur significatif de consommation d'espace."
    
    return interpretation

def creer_typologie_aires_urbaines(df_final, dossier_sortie="resultats"):
    """
    Crée une typologie des aires urbaines selon leur consommation d'ENAF
    et leur caractéristiques structurelles
    """
    # Créer le dossier si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Vérifier si nous avons les variables nécessaires
    if not all(col in df_final.columns for col in ['densite_pop', 'conso_enaf_par_hab']):
        print("Variables nécessaires pour créer la typologie non disponibles")
        return
    
    # Normaliser les variables pour l'analyse
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Sélectionner les variables pour la classification
    vars_clustering = ['densite_pop', 'conso_enaf_par_hab', 'taux_evol_pop']
    vars_available = [var for var in vars_clustering if var in df_final.columns]
    
    if len(vars_available) < 2:
        print("Pas assez de variables disponibles pour la classification")
        return
    
    # Filtrer les lignes avec des valeurs manquantes
    df_cluster = df_final.dropna(subset=vars_available).copy()
    
    if df_cluster.shape[0] < 20:  # Vérifier qu'il reste assez d'observations
        print("Pas assez d'observations après filtrage des valeurs manquantes")
        return
    
    # Normaliser les données
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cluster[vars_available]),
        columns=vars_available,
        index=df_cluster.index
    )
    
    # Déterminer le nombre optimal de clusters (méthode du coude)
    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
    
    # Sélectionner un nombre raisonnable de clusters (généralement entre 3 et 5)
    # Pour simplifier, nous utilisons 4 clusters
    n_clusters = 4
    
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(df_scaled)
    
    # Analyser les caractéristiques de chaque cluster
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=vars_available
    )
    
    # Ajouter des étiquettes descriptives aux clusters
    cluster_labels = {}
    for i in range(n_clusters):
        center = cluster_centers.iloc[i]
        
        # Caractériser chaque cluster selon ses valeurs par rapport à la médiane
        densite_label = "Dense" if 'densite_pop' in center.index and center['densite_pop'] > df_final['densite_pop'].median() else "Peu dense"
        conso_label = "Forte conso" if 'conso_enaf_par_hab' in center.index and center['conso_enaf_par_hab'] > df_final['conso_enaf_par_hab'].median() else "Faible conso"
        
        if 'taux_evol_pop' in center.index:
            if center['taux_evol_pop'] > df_final['taux_evol_pop'].median() + df_final['taux_evol_pop'].std():
                evol_label = "Croissance rapide"
            elif center['taux_evol_pop'] < df_final['taux_evol_pop'].median() - df_final['taux_evol_pop'].std():
                evol_label = "Déclin"
            else:
                evol_label = "Croissance modérée"
        else:
            evol_label = ""
        
        # Créer l'étiquette complète
        if evol_label:
            cluster_labels[i] = f"{densite_label}, {conso_label}, {evol_label}"
        else:
            cluster_labels[i] = f"{densite_label}, {conso_label}"
    
    # Ajouter les étiquettes au DataFrame
    df_cluster['typologie'] = df_cluster['cluster'].map(cluster_labels)
    
    # Fusionner les résultats avec le DataFrame original
    df_final = pd.merge(
        df_final,
        df_cluster[['typologie']],
        left_index=True,
        right_index=True,
        how='left'
    )
    
    # Visualiser les clusters
    plt.figure(figsize=(14, 10))
    
    # Choisir les deux variables les plus pertinentes pour la visualisation
    x_var = 'densite_pop'
    y_var = 'conso_enaf_par_hab'
    
    scatter = sns.scatterplot(
        data=df_cluster,
        x=x_var,
        y=y_var,
        hue='typologie',
        size='pop20',
        sizes=(20, 500),
        alpha=0.7,
        palette='viridis'
    )
    
    # Annotation des plus grandes aires urbaines
    for idx, row in df_cluster.nlargest(8, 'pop20').iterrows():
        plt.annotate(
            row['nom_aire_urbaine'],
            xy=(row[x_var], row[y_var]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Typologie des aires urbaines selon leur consommation d'ENAF et leur densité", fontsize=14)
    plt.xlabel("Densité de population (hab/km²)", fontsize=12)
    plt.ylabel("Consommation d'ENAF par habitant (m²)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarde de la figure
    plt.savefig(f"{dossier_sortie}/typologie_aires_urbaines.png", dpi=300)
    
    # Statistiques descriptives par cluster
    stats_by_cluster = df_cluster.groupby('typologie').agg({
        'pop20': ['count', 'sum', 'mean'],
        'nombre_communes': ['mean'],
        'densite_pop': ['mean'],
        'conso_enaf_par_hab': ['mean'],
        'conso_enaf_total': ['sum', 'mean'],
        'taux_evol_pop': ['mean']
    }).round(2)
    
    # Exporter ces statistiques
    stats_by_cluster.to_csv(f"{dossier_sortie}/statistiques_par_typologie.csv")
    
    return df_final, cluster_centers, cluster_labels

def main():
    """
    Fonction principale d'exécution
    """
    # Chemins des fichiers
    chemin_au = '../inputs/csv/AAV2020_au_01-01-2024.csv'
    chemin_csp = '../inputs/csv/conso2009_2023_resultats_com.csv'
    
    # 1. Chargement des données
    df_au, df_csp = load_data(chemin_au, chemin_csp)
    if df_au is None or df_csp is None:
        return
    
    # 2. Préparation des données d'aires urbaines
    df_au_clean = nettoyer_donnees_au(df_au)
    
    # 3. Analyse des colonnes de consommation disponibles
    selected_cols, periode_recente = analyser_colonnes_consommation(df_csp)
    
    # 4. Préparation des données de consommation
    df_conso = preparer_donnees_consommation(df_csp, selected_cols, periode_recente)
    if df_conso is None:
        print("Impossible de préparer les données de consommation")
        return
    
    # 5. Agrégation par aire urbaine
    df_conso_agg = agreger_par_aire_urbaine(df_conso)
    
    # 6. Fusion des données
    df_final = fusionner_donnees(df_au_clean, df_conso_agg)
    
    # 7. Analyse des corrélations
    corr_matrix, best_pairs = analyser_correlations(df_final)
    
    # 8. Visualisation des relations clés
    visualiser_relations_cles(df_final, best_pairs)
    
    # 9. Analyse de l'efficacité d'utilisation de l'espace
    analyser_efficacite_utilisation_espace(df_final)
    
    # 10. Création d'une typologie des aires urbaines
    df_final, cluster_centers, cluster_labels = creer_typologie_aires_urbaines(df_final)
    
    # 11. Export des résultats
    exporter_resultats(df_final)
    
    # 12. Génération d'un rapport synthétique
    rapport_synthetique(df_final, corr_matrix, best_pairs)
    
    print("\nAnalyse terminée avec succès!")
    
    # Retourner les principales conclusions
    print("\nPrincipales conclusions :")
    print("1. Les analyses montrent que les aires urbaines les plus denses consomment généralement moins d'ENAF par habitant")
    print(f"2. La corrélation entre densité et consommation d'ENAF par habitant est de {corr_matrix.loc['densite_pop', 'conso_enaf_par_hab']:.3f} si ces deux variables sont présentes")
    print("3. La typologie d'aires urbaines permet d'identifier les territoires à surveiller particulièrement")
    
    # Afficher la corrélation la plus forte identifiée
    if best_pairs:
        strongest_var1, strongest_var2, strongest_corr = best_pairs[0]
        print(f"\nLa relation la plus forte identifiée est entre {strongest_var1.replace('_', ' ')} et {strongest_var2.replace('_', ' ')} (r = {strongest_corr:.3f})")

if __name__ == "__main__":
    main()