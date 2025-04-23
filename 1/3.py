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
        df_csp = pd.read_csv(chemin_csp, sep=';', low_memory=False)
        print(f"Données chargées avec succès: {df_au.shape[0]} aires urbaines et {df_csp.shape[0]} communes")
        return df_au, df_csp
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None, None

def analyser_structure_donnees(df):
    """
    Analyse la structure et les caractéristiques des données
    """
    # Informations de base
    print(f"Nombre de lignes: {df.shape[0]}, Nombre de colonnes: {df.shape[1]}")
    
    # Identifier les types de colonnes
    types_count = df.dtypes.value_counts()
    print("\nTypes de données dans le DataFrame:")
    print(types_count)
    
    # Identifier les colonnes numériques
    cols_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNombre de colonnes numériques: {len(cols_numeriques)}")
    
    # Valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nColonnes avec valeurs manquantes:")
    print(missing if not missing.empty else "Aucune valeur manquante")
    
    return cols_numeriques

def identifier_colonnes_par_pattern(df, pattern_dict):
    """
    Identifie et groupe les colonnes selon des patterns définis
    """
    colonnes_categorisees = {}
    
    for categorie, pattern in pattern_dict.items():
        colonnes_categorisees[categorie] = [col for col in df.columns if re.match(pattern, col)]
        print(f"Catégorie {categorie}: {len(colonnes_categorisees[categorie])} colonnes identifiées")
    
    return colonnes_categorisees

def preparer_df_analyse(df_csp):
    """
    Prépare un dataframe propre pour l'analyse en convertissant les types
    """
    df_clean = df_csp.copy()
    
    # Extraction du code d'aire urbaine
    df_clean['AAV'] = df_clean['aav2020'].astype(str).str.extract(r'^(\d{3})').fillna('000')
    
    # Identifier les colonnes de population et d'emploi
    cols_pop = ['pop14', 'pop20', 'pop1420']
    cols_men = ['men14', 'men20', 'men1420']
    cols_emp = ['emp14', 'emp20', 'emp1420']
    
    # Convertir en numérique les colonnes sélectionnées
    for col in cols_pop + cols_men + cols_emp:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Extraire et convertir toutes les colonnes "art" numériques
    art_cols = [col for col in df_clean.columns if col.startswith('art') and not col.endswith('txt')]
    for col in art_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Identification de la dernière année disponible pour chaque type d'activité
    derniers_act = {}
    derniers_hab = {}
    derniers_mix = {}
    derniers_rou = {}
    derniers_fer = {}
    
    # Regex pour extraire l'année et le type d'une colonne art
    pattern = r'art(\d{2})(act|hab|mix|rou|fer|inc)(\d{2})'
    
    for col in art_cols:
        match = re.match(pattern, col)
        if match:
            annee_base = match.group(1)
            type_donnee = match.group(2)
            annee_ref = match.group(3)
            
            if type_donnee == 'act':
                derniers_act[annee_base] = col
            elif type_donnee == 'hab':
                derniers_hab[annee_base] = col
            elif type_donnee == 'mix':
                derniers_mix[annee_base] = col
            elif type_donnee == 'rou':
                derniers_rou[annee_base] = col
            elif type_donnee == 'fer':
                derniers_fer[annee_base] = col
    
    # Sélectionner la dernière année disponible pour chaque type
    derniers_act = sorted(derniers_act.items(), key=lambda x: (x[0], x[1]))[-1][1] if derniers_act else None
    derniers_hab = sorted(derniers_hab.items(), key=lambda x: (x[0], x[1]))[-1][1] if derniers_hab else None
    derniers_mix = sorted(derniers_mix.items(), key=lambda x: (x[0], x[1]))[-1][1] if derniers_mix else None
    derniers_rou = sorted(derniers_rou.items(), key=lambda x: (x[0], x[1]))[-1][1] if derniers_rou else None
    derniers_fer = sorted(derniers_fer.items(), key=lambda x: (x[0], x[1]))[-1][1] if derniers_fer else None
    
    print(f"\nColonnes les plus récentes sélectionnées:")
    print(f"Activité: {derniers_act}")
    print(f"Habitat: {derniers_hab}")
    print(f"Mixte: {derniers_mix}")
    print(f"Route: {derniers_rou}")
    print(f"Fer: {derniers_fer}")
    
    # Créer des colonnes dérivées pour l'analyse
    if derniers_act and derniers_hab:
        df_clean['ratio_act_hab'] = df_clean[derniers_act] / df_clean[derniers_hab].replace(0, np.nan)
    
    if derniers_act and derniers_mix:
        df_clean['ratio_act_mix'] = df_clean[derniers_act] / df_clean[derniers_mix].replace(0, np.nan)
    
    if derniers_rou and derniers_fer:
        df_clean['ratio_rou_fer'] = df_clean[derniers_rou] / df_clean[derniers_fer].replace(0, np.nan)
    
    # Calculer la densité
    df_clean['densite_pop'] = df_clean['pop20'] / (df_clean['surfcom2023'] / 10000)  # hab/km²
    
    # Calculer la consommation d'espace par habitant
    cols_artif = [col for col in art_cols if any(col.endswith(f'{type_}{derniers_act[-2:]}') for type_ in ['act', 'hab', 'mix', 'rou', 'fer', 'inc'])]
    df_clean['artif_total'] = df_clean[cols_artif].sum(axis=1)
    df_clean['artif_par_hab'] = df_clean['artif_total'] / df_clean['pop20'].replace(0, np.nan)
    
    return df_clean, derniers_act, derniers_hab, derniers_mix, derniers_rou, derniers_fer

def analyser_correlations(df, cols_interest=None):
    """
    Analyse les corrélations entre variables et identifie les plus significatives
    """
    # Sélectionner les colonnes d'intérêt
    if cols_interest:
        df_corr = df[cols_interest].copy()
    else:
        # Sélectionner toutes les colonnes numériques avec moins de 50% de valeurs manquantes
        num_cols = df.select_dtypes(include=[np.number]).columns
        df_corr = df[num_cols].copy()
        threshold = len(df) * 0.5
        df_corr = df_corr.loc[:, df_corr.isnull().sum() < threshold]
    
    # Calculer la matrice de corrélation
    corr_matrix = df_corr.corr()
    
    return corr_matrix

def visualiser_correlations(corr_matrix, titre, n_top=15, dossier_sortie="resultats"):
    """
    Visualise les corrélations sous forme de heatmap et identifie les plus fortes
    """
    # Création du dossier de sortie si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Créer un masque pour le triangle supérieur
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Figure 1: Heatmap des corrélations
    plt.figure(figsize=(20, 16))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap,
        vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False  # Trop de données pour l'annotation
    )
    
    plt.title(f"Matrice de corrélation: {titre}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/correlation_matrix_{titre.replace(' ', '_').lower()}.png", dpi=300)
    
    # Trouver les n paires les plus corrélées positivement et négativement
    # Convertir la matrice en DataFrame pour faciliter le traitement
    corr_df = corr_matrix.abs().unstack().sort_values(ascending=False)
    corr_df = corr_df[corr_df < 1.0]  # Exclure les corrélations parfaites (diagonale)
    
    print(f"\nTop {n_top} corrélations les plus fortes:")
    for (var1, var2), corr in corr_df.head(n_top).items():
        print(f"{var1} - {var2}: {corr_matrix.loc[var1, var2]:.4f}")
    
    return corr_df.head(n_top*2)

def analyser_variables_specifiques(df, variables_clefs, population_col='pop20'):
    """
    Analyse approfondie de variables clés avec la population
    """
    resultats = {}
    
    for var in variables_clefs:
        # Calculer corrélation avec population
        corr_pop = df[[var, population_col]].corr().iloc[0, 1]
        
        # Calculer par tranches de population
        df['tranche_pop'] = pd.cut(
            df[population_col],
            bins=[0, 1000, 5000, 10000, 50000, float('inf')],
            labels=['< 1000', '1000-5000', '5000-10000', '10000-50000', '> 50000']
        )
        
        # Moyenne par tranche de population
        moyenne_par_tranche = df.groupby('tranche_pop')[var].mean()
        
        resultats[var] = {
            'correlation_pop': corr_pop,
            'moyenne_par_tranche': moyenne_par_tranche
        }
        
        # Visualisation
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='tranche_pop', y=var, data=df)
        plt.title(f"Distribution de {var} par tranche de population", fontsize=16)
        plt.xlabel("Tranche de population", fontsize=14)
        plt.ylabel(var, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"resultats/distribution_{var}_par_pop.png", dpi=300)
    
    return resultats

def regrouper_par_aires_urbaines(df_csp, df_au, variables_clefs):
    """
    Regroupe les données par aires urbaines et calcule les moyennes/sommes des variables clés
    """
    # Préparation des données d'aires urbaines
    df_au_clean = df_au.copy()
    df_au_clean['AAV'] = df_au_clean['AAV2020'].astype(str).str.zfill(3)
    
    # Calcul des agrégations par aire urbaine
    agg_dict = {'pop20': 'sum', 'surfcom2023': 'sum'}
    
    # Ajouter les variables clés à agréger (moyenne par défaut)
    for var in variables_clefs:
        if var in df_csp.columns:
            if var.startswith('art') or var in ['pop20', 'men20', 'emp20']:
                agg_dict[var] = 'sum'
            else:
                agg_dict[var] = 'mean'
    
    # Agrégation par aire urbaine
    df_agg = df_csp.groupby('AAV').agg(agg_dict).reset_index()
    
    # Fusion avec les données des aires urbaines
    df_final = pd.merge(df_au_clean, df_agg, on='AAV', how='inner')
    
    # Recalcul de certains ratios au niveau agrégé
    for var in variables_clefs:
        if var.startswith('ratio_') and var not in df_final.columns:
            parts = var.split('_')[1:]
            num_part = '_'.join(parts[:1])
            denom_part = '_'.join(parts[1:])
            
            # Rechercher les colonnes correspondantes
            num_cols = [col for col in df_final.columns if num_part in col]
            denom_cols = [col for col in df_final.columns if denom_part in col]
            
            if num_cols and denom_cols:
                df_final[var] = df_final[num_cols[0]] / df_final[denom_cols[0]].replace(0, np.nan)
    
    print(f"Agrégation par aires urbaines effectuée: {df_final.shape[0]} aires urbaines")
    
    return df_final

def visualiser_relations_clefs(df_final, var_x, var_y_list, dossier_sortie="resultats"):
    """
    Visualise les relations entre la variable x (typiquement population) et les variables y identifiées
    """
    # Création du dossier de sortie si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Ajouter catégories de taille pour les aires urbaines
    df_final['categorie_taille'] = pd.cut(
        df_final['TAAV2017'],
        bins=[0, 50000, 200000, 500000, float('inf')],
        labels=['Petite', 'Moyenne', 'Grande', 'Très grande']
    )
    
    # Créer un graphique pour chaque variable y
    for var_y in var_y_list:
        plt.figure(figsize=(14, 8))
        
        # Graphique de dispersion avec régression
        scatter = sns.scatterplot(
            data=df_final, 
            x=var_x, 
            y=var_y,
            size='NB_COM',
            sizes=(20, 500),
            hue='categorie_taille',
            palette='viridis',
            alpha=0.7
        )
        
        # Ajouter une ligne de tendance
        sns.regplot(
            data=df_final, 
            x=var_x, 
            y=var_y, 
            scatter=False, 
            color='red', 
            line_kws={"linewidth": 2}
        )
        
        # Annotation des plus grandes aires urbaines
        for idx, row in df_final.nlargest(5, var_x).iterrows():
            plt.annotate(
                row['LIBAAV2020'],
                xy=(row[var_x], row[var_y]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Calculer et afficher le coefficient de corrélation
        corr_val = df_final[[var_x, var_y]].corr().iloc[0, 1]
        plt.text(
            0.05, 0.95, 
            f"Corrélation: {corr_val:.4f}", 
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        plt.title(f"Relation entre {var_x} et {var_y}", fontsize=16)
        plt.xlabel(var_x, fontsize=14)
        plt.ylabel(var_y, fontsize=14)
        plt.xscale('log')  # Échelle logarithmique pour population
        plt.grid(True, alpha=0.3)
        plt.legend(title="Catégorie de taille", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Sauvegarde de la figure
        plt.savefig(f"{dossier_sortie}/relation_{var_x}_{var_y}.png", dpi=300)
    
    return

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
    
    # 2. Analyse de la structure des données
    print("\n--- Analyse de la structure des données CSP ---")
    cols_numeriques = analyser_structure_donnees(df_csp)
    
    # 3. Identification des patterns de colonnes
    pattern_dict = {
        'activite': r'art\d{2}act\d{2}',
        'habitat': r'art\d{2}hab\d{2}',
        'mixte': r'art\d{2}mix\d{2}',
        'route': r'art\d{2}rou\d{2}',
        'fer': r'art\d{2}fer\d{2}',
        'population': r'pop\d{2}'
    }
    
    print("\n--- Identification des colonnes par catégorie ---")
    colonnes_categorisees = identifier_colonnes_par_pattern(df_csp, pattern_dict)
    
    # 4. Préparation des données pour l'analyse
    print("\n--- Préparation des données pour l'analyse ---")
    df_clean, derniers_act, derniers_hab, derniers_mix, derniers_rou, derniers_fer = preparer_df_analyse(df_csp)
    
    # 5. Analyse des corrélations entre toutes les variables numériques
    print("\n--- Analyse des corrélations entre toutes les variables ---")
    corr_matrix_all = analyser_correlations(df_clean)
    top_corr_all = visualiser_correlations(corr_matrix_all, "Toutes variables")
    
    # 6. Analyse des corrélations entre variables d'artificialisation
    cols_artif = [derniers_act, derniers_hab, derniers_mix, derniers_rou, derniers_fer, 
                 'ratio_act_hab', 'ratio_act_mix', 'ratio_rou_fer', 'artif_total', 'artif_par_hab']
    cols_artif = [col for col in cols_artif if col in df_clean.columns]
    
    print("\n--- Analyse des corrélations entre variables d'artificialisation ---")
    corr_matrix_artif = analyser_correlations(df_clean, cols_artif)
    top_corr_artif = visualiser_correlations(corr_matrix_artif, "Variables d'artificialisation")
    
    # 7. Analyse des corrélations entre variables démographiques et d'artificialisation
    cols_demo = ['pop14', 'pop20', 'pop1420', 'men14', 'men20', 'men1420', 'emp14', 'emp20', 'emp1420', 'densite_pop']
    cols_demo_artif = cols_demo + cols_artif
    cols_demo_artif = [col for col in cols_demo_artif if col in df_clean.columns]
    
    print("\n--- Analyse des corrélations entre variables démographiques et d'artificialisation ---")
    corr_matrix_demo_artif = analyser_correlations(df_clean, cols_demo_artif)
    top_corr_demo_artif = visualiser_correlations(corr_matrix_demo_artif, "Variables démographiques et d'artificialisation")
    
    # 8. Identification des variables les plus intéressantes pour l'analyse
    print("\n--- Identification des variables clés pour l'analyse ---")
    # Basé sur les top corrélations et les variables les plus significatives
    variables_clefs = [
        'pop20',               # Population actuelle
        'artif_total',         # Artificialisation totale
        'artif_par_hab',       # Artificialisation par habitant
        derniers_act,          # Surfaces d'activité
        derniers_hab,          # Surfaces d'habitat
        'ratio_act_hab',       # Ratio activité/habitat
        'densite_pop',         # Densité de population
        'mepart1420',          # Évolution du nombre de ménages
        'artpop1420'           # Évolution de l'artificialisation
    ]
    
    # Filtrer pour n'inclure que les variables présentes
    variables_clefs = [var for var in variables_clefs if var in df_clean.columns]
    print(f"Variables clés sélectionnées: {variables_clefs}")
    
    # 9. Analyse approfondie des variables clés
    print("\n--- Analyse approfondie des variables clés ---")
    resultats_var_clefs = analyser_variables_specifiques(df_clean, variables_clefs)
    
    # 10. Regroupement et analyse par aires urbaines
    print("\n--- Regroupement et analyse par aires urbaines ---")
    df_aires_urbaines = regrouper_par_aires_urbaines(df_clean, df_au, variables_clefs)
    
    # 11. Visualisation des relations clés au niveau des aires urbaines
    print("\n--- Visualisation des relations clés ---")
    visualiser_relations_clefs(
        df_aires_urbaines, 
        'TAAV2017',  # Population de l'aire urbaine
        ['ratio_act_hab', 'artif_par_hab', 'densite_pop', derniers_act, derniers_hab]
    )
    
    # 12. Export des résultats
    print("\n--- Export des résultats ---")
    # Sauvegarder les matrices de corrélation
    corr_matrix_all.to_csv("resultats/correlation_matrix_all.csv", sep=';')
    corr_matrix_artif.to_csv("resultats/correlation_matrix_artif.csv", sep=';')
    corr_matrix_demo_artif.to_csv("resultats/correlation_matrix_demo_artif.csv", sep=';')
    
    # Sauvegarder le dataframe des aires urbaines avec variables clés
    df_aires_urbaines.to_csv("resultats/aires_urbaines_variables_clefs.csv", sep=';', index=False)
    
    print("\nAnalyse terminée avec succès!")

if __name__ == "__main__":
    main()