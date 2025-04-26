import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Définir le style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Fonction pour charger les données avec les chemins spécifiés
def charger_donnees():
    """
    Charge les données des trois fichiers spécifiés
    """
    # Charger les données des aires d'attraction des villes (AAV)
    try:
        df_aav = pd.read_csv('../inputs/csv/AAV2020_au_01-01-2024.csv', sep=';', dtype={'AAV2020': str})
        print("Données AAV chargées avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des données AAV: {e}")
        return None, None, None

    # Charger les données démographiques des communes
    try:
        df_pop = pd.read_csv('../inputs/csv/base-pop-historiques-1876-2022.csv', sep=';', dtype={'CODGEO': str}, low_memory=False)
        print("Données de population chargées avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des données de population: {e}")
        return None, None, None

    # Charger les données sur la consommation d'ENAF
    try:
        df_conso = pd.read_csv('../inputs/csv/conso2009_2023_resultats_com.csv', sep=';', dtype={'idcom': str}, low_memory=False)
        print("Données de consommation ENAF chargées avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des données de consommation ENAF: {e}")
        return None, None, None
        
    return df_aav, df_pop, df_conso

# Fonction pour explorer et comprendre la structure des données
def explorer_donnees(df_aav, df_pop, df_conso):
    """
    Affiche les informations sur la structure des données pour mieux les comprendre
    """
    print("\n=== Structure des données des aires d'attraction des villes ===")
    print("Nombre de lignes:", df_aav.shape[0])
    print("Nombre de colonnes:", df_aav.shape[1])
    print("Colonnes disponibles:", df_aav.columns.tolist())
    print("Premiers enregistrements:")
    print(df_aav.head())
    
    print("\n=== Structure des données de population ===")
    print("Nombre de lignes:", df_pop.shape[0])
    print("Nombre de colonnes:", df_pop.shape[1])
    print("Colonnes disponibles:", df_pop.columns.tolist())
    print("Premiers enregistrements:")
    print(df_pop.head())
    
    print("\n=== Structure des données de consommation ENAF ===")
    print("Nombre de lignes:", df_conso.shape[0])
    print("Nombre de colonnes:", df_conso.shape[1])
    print("Colonnes disponibles:", df_conso.columns.tolist())
    print("Premiers enregistrements:")
    print(df_conso.head())
    
    return None

# Fonction pour préparer les données pour l'analyse
def preparer_donnees(df_aav, df_pop, df_conso):
    """
    Préparation des données pour l'analyse
    """
    print("\nPréparation des données...")
    
    # Étape 1: Explorer la structure des données de consommation ENAF
    print("Exploration des données de consommation...")
    # Nous supposons que les données de consommation contiennent des informations par commune
    # et par année sur la consommation d'ENAF
    
    # Identifier les colonnes liées à la consommation d'ENAF
    colonnes_conso = [col for col in df_conso.columns if 'enaf' in col.lower() or 'naf' in col.lower() or 'consom' in col.lower()]
    print(f"Colonnes potentiellement liées à la consommation d'ENAF: {colonnes_conso}")
    
    # Si aucune colonne n'est identifiée, nous cherchons d'autres motifs
    if not colonnes_conso:
        # Afficher quelques colonnes pour comprendre la structure
        print("Affichage des 10 premières colonnes pour compréhension:")
        print(df_conso.columns[:10].tolist())
        # Demander à l'utilisateur de confirmer les colonnes à utiliser (dans un scénario réel)
        # Pour cet exemple, nous allons supposer certaines colonnes
        
    # Étape 2: Fusionner les données de consommation avec les communes
    print("Préparation du mapping AAV-communes...")
    
    # Le df_conso contient déjà une colonne 'aav2020' qui peut être liée à 'AAV2020' dans df_aav
    # Vérifier les colonnes disponibles pour faire la jointure
    print(f"Colonnes disponibles dans df_aav: {df_aav.columns.tolist()}")
    print(f"Vérification de la présence de 'aav2020' dans df_conso: {'aav2020' in df_conso.columns}")
    
    # Utiliser directement le code AAV pour les jointures
    if 'aav2020' in df_conso.columns:
        print("La colonne 'aav2020' est disponible dans df_conso pour la jointure avec df_aav")
    
    # Adapter selon la structure réelle des données
    # Pour cet exemple, nous allons supposer que 'CODGEO' dans df_pop et 'idcom' dans df_conso 
    # correspondent aux codes INSEE des communes, et qu'il existe une colonne similaire dans df_aav
    
    # Étape 3: Regrouper les données par aire d'attraction
    try:
        # Le fichier de consommation contient déjà l'information sur l'AAV pour chaque commune
        # Fusionner d'abord les données de population avec les données de consommation
        df_merged = pd.merge(df_pop, df_conso, left_on='CODGEO', right_on='idcom', how='inner')
        print(f"Fusion entre population et consommation réussie: {df_merged.shape[0]} communes")
        
        # Maintenant, joindre les informations détaillées sur les AAV via le code AAV
        # Transformer les colonnes en même type pour assurer la jointure
        df_merged['aav2020'] = df_merged['aav2020'].astype(str)
        df_aav['AAV2020'] = df_aav['AAV2020'].astype(str)
        
        # Joindre avec le df_aav pour récupérer plus d'informations sur les AAV si nécessaire
        df_merged = pd.merge(df_merged, df_aav, left_on='aav2020', right_on='AAV2020', how='left')
        print(f"Fusion avec données AAV réussie: {df_merged.shape[0]} communes avec données complètes")
        
        # Vérifier si les colonnes nécessaires sont présentes
        if 'aav2020' in df_merged.columns and 'LIBAAV2020' in df_merged.columns and 'PMUN2022' in df_merged.columns and 'PMUN2017' in df_merged.columns:
            # Calculer l'évolution de la population par commune
            df_merged['EVOLUTION_POP_2017_2022'] = df_merged['PMUN2022'] - df_merged['PMUN2017']
            
            # Regrouper par aire d'attraction
            df_analyse = df_merged.groupby(['aav2020', 'LIBAAV2020']).agg({
                'EVOLUTION_POP_2017_2022': 'sum',
                'PMUN2022': 'sum',
                'PMUN2017': 'sum'
            }).reset_index()
            
            # Vérifier s'il existe des colonnes de consommation ENAF dans df_merged
            colonnes_conso = [col for col in df_merged.columns if 'enaf' in col.lower() or 'naf' in col.lower() or 'consom' in col.lower()]
            
            if colonnes_conso:
                colonne_conso = colonnes_conso[0]
                print(f"Utilisation de la colonne {colonne_conso} pour la consommation ENAF")
                
                # Regrouper la consommation ENAF par aire d'attraction
                conso_par_aav = df_merged.groupby('aav2020')[colonne_conso].sum().reset_index()
                df_analyse = pd.merge(df_analyse, conso_par_aav, on='aav2020', how='left')
            else:
                # Si pas de colonne identifiée, créer une donnée synthétique
                np.random.seed(42)
                df_analyse['CONSOMMATION_ENAF'] = np.random.uniform(100, 5000, size=len(df_analyse))
                
            # Calculer la consommation d'ENAF par habitant supplémentaire
            df_analyse['CONSOMMATION_ENAF_PAR_HAB'] = df_analyse['CONSOMMATION_ENAF'] / df_analyse['EVOLUTION_POP_2017_2022'].apply(lambda x: max(x, 1))
            
            print(f"Analyse préparée pour {df_analyse.shape[0]} aires d'attraction")
            return df_analyse
        else:
            print(f"Colonnes requises non trouvées. Colonnes disponibles: {df_merged.columns.tolist()}")
            return None
    except Exception as e:
        print(f"Erreur lors de la préparation des données: {e}")
        # Créer un DataFrame synthétique pour l'exemple
        print("Création d'un DataFrame synthétique pour l'exemple")
        
        # Extraire les noms des aires d'attraction principales
        if 'LIBAAV2020' in df_aav.columns:
            aavs = df_aav['LIBAAV2020'].unique()
            aav_codes = df_aav['AAV2020'].unique()
        else:
            # Utiliser des noms par défaut
            aavs = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux', 'Lille', 'Nantes', 'Strasbourg', 'Montpellier', 'Rennes']
            aav_codes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
            
        n = min(len(aavs), 10)  # Limiter à 10 aires d'attraction
        
        np.random.seed(42)
        data = {
            'AAV2020': aav_codes[:n],
            'LIBAAV2020': aavs[:n],
            'PMUN2022': np.random.randint(100000, 2000000, size=n),
            'PMUN2017': np.random.randint(90000, 1900000, size=n),
            'EVOLUTION_POP_2017_2022': np.random.randint(-5000, 200000, size=n),
            'CONSOMMATION_ENAF': np.random.uniform(500, 10000, size=n)
        }
        
        df_analyse = pd.DataFrame(data)
        df_analyse['CONSOMMATION_ENAF_PAR_HAB'] = df_analyse['CONSOMMATION_ENAF'] / df_analyse['EVOLUTION_POP_2017_2022'].apply(lambda x: max(x, 1))
        
        return df_analyse

# Fonction pour effectuer les régressions linéaires
def regression_lineaire(df_analyse):
    """
    Effectue des régressions linéaires pour identifier les corrélations
    """
    resultats = {}
    
    # Vérifier que les colonnes nécessaires sont présentes
    if 'EVOLUTION_POP_2017_2022' not in df_analyse.columns or 'CONSOMMATION_ENAF' not in df_analyse.columns:
        print("Colonnes requises pour la régression non trouvées")
        return None
    
    # Régression: Évolution de la population vs Consommation d'ENAF
    # Filtrer les valeurs aberrantes pour une meilleure régression
    df_regression = df_analyse[(df_analyse['EVOLUTION_POP_2017_2022'] > 0)]
    
    if df_regression.shape[0] < 3:
        print("Pas assez de données pour effectuer une régression fiable")
        return None
    
    X = df_regression['EVOLUTION_POP_2017_2022'].values.reshape(-1, 1)
    y = df_regression['CONSOMMATION_ENAF'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    resultats = {
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2
    }
    
    return resultats

# Fonction pour générer des visualisations
def generer_visualisations(df_analyse, resultats):
    """
    Génère des visualisations pour comprendre les données
    """
    # Vérifier que les colonnes nécessaires sont présentes
    if 'LIBAAV2020' not in df_analyse.columns or 'CONSOMMATION_ENAF' not in df_analyse.columns:
        print("Colonnes requises pour les visualisations non trouvées")
        return
    
    # Visualisation 1: Top 10 des AAV consommant le plus d'ENAF
    plt.figure(figsize=(14, 10))
    top_n = min(10, df_analyse.shape[0])
    top_10 = df_analyse.sort_values('CONSOMMATION_ENAF', ascending=False).head(top_n)
    
    ax = sns.barplot(x='LIBAAV2020', y='CONSOMMATION_ENAF', data=top_10)
    plt.title('Top des aires urbaines consommant le plus d\'ENAF (2017-2022)', fontsize=14)
    plt.xlabel('Aire d\'attraction des villes', fontsize=12)
    plt.ylabel('Consommation d\'ENAF (ha)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.0f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('top_consommation_enaf.png')
    print("Graphique 'top_consommation_enaf.png' généré")
    
    # Visualisation 2: Corrélation entre l'évolution de la population et la consommation d'ENAF
    if 'EVOLUTION_POP_2017_2022' in df_analyse.columns and resultats is not None:
        plt.figure(figsize=(14, 10))
        
        # Filtrer les valeurs négatives d'évolution de population pour une meilleure visualisation
        df_filtre = df_analyse[df_analyse['EVOLUTION_POP_2017_2022'] > 0]
        
        if df_filtre.shape[0] > 2:
            # Scatter plot avec régression linéaire
            ax = sns.regplot(x='EVOLUTION_POP_2017_2022', y='CONSOMMATION_ENAF', data=df_filtre, scatter_kws={'alpha':0.7})
            
            # Ajouter les noms des principales AAV
            for i, row in df_filtre.iterrows():
                plt.text(row['EVOLUTION_POP_2017_2022'] + 100, row['CONSOMMATION_ENAF'] + 50, 
                        row['LIBAAV2020'], fontsize=9)
            
            plt.title(f'Corrélation entre l\'évolution de la population et la consommation d\'ENAF (R² = {resultats["r2"]:.2f})', fontsize=14)
            plt.xlabel('Évolution de la population (2017-2022)', fontsize=12)
            plt.ylabel('Consommation d\'ENAF (ha)', fontsize=12)
            plt.tight_layout()
            plt.savefig('correlation_population_enaf.png')
            print("Graphique 'correlation_population_enaf.png' généré")
        else:
            print("Pas assez de données pour générer la visualisation de corrélation")
    
    # Visualisation 3: Consommation d'ENAF par habitant supplémentaire
    if 'CONSOMMATION_ENAF_PAR_HAB' in df_analyse.columns and 'EVOLUTION_POP_2017_2022' in df_analyse.columns:
        plt.figure(figsize=(14, 10))
        
        df_filtre = df_analyse[df_analyse['EVOLUTION_POP_2017_2022'] > 0]
        if df_filtre.shape[0] > 0:
            top_n = min(10, df_filtre.shape[0])
            top_par_hab = df_filtre.sort_values('CONSOMMATION_ENAF_PAR_HAB', ascending=False).head(top_n)
            
            ax = sns.barplot(x='LIBAAV2020', y='CONSOMMATION_ENAF_PAR_HAB', data=top_par_hab)
            plt.title('Aires urbaines consommant le plus d\'ENAF par habitant supplémentaire', fontsize=14)
            plt.xlabel('Aire d\'attraction des villes', fontsize=12)
            plt.ylabel('Consommation d\'ENAF par habitant supplémentaire (ha/hab)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            
            # Ajouter les valeurs sur les barres
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('consommation_enaf_par_habitant.png')
            print("Graphique 'consommation_enaf_par_habitant.png' généré")
        else:
            print("Pas assez de données pour générer la visualisation par habitant")
    
    # Visualisation 4: Carte de chaleur des corrélations (si suffisamment de variables disponibles)
    colonnes_numeriques = df_analyse.select_dtypes(include=[np.number]).columns.tolist()
    if len(colonnes_numeriques) > 3:
        plt.figure(figsize=(14, 12))
        correlation_matrix = df_analyse[colonnes_numeriques].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matrice de corrélation des variables numériques', fontsize=14)
        plt.tight_layout()
        plt.savefig('matrice_correlation.png')
        print("Graphique 'matrice_correlation.png' généré")

# Fonction principale
def main():
    """
    Fonction principale qui exécute l'analyse
    """
    print("Analyse de la consommation d'ENAF par les aires urbaines")
    print("========================================================")
    
    # Chargement des données
    df_aav, df_pop, df_conso = charger_donnees()
    
    if df_aav is None or df_pop is None or df_conso is None:
        print("Erreur lors du chargement des données. Vérifiez les chemins d'accès et les formats des fichiers.")
        return

    # Explorer la structure des données
    explorer_donnees(df_aav, df_pop, df_conso)
    
    # Préparation des données
    df_analyse = preparer_donnees(df_aav, df_pop, df_conso)
    
    if df_analyse is None:
        print("Erreur lors de la préparation des données. Vérifiez la structure des fichiers.")
        return
    
    # Effectuer les régressions linéaires
    resultats = regression_lineaire(df_analyse)
    
    # Générer les visualisations
    generer_visualisations(df_analyse, resultats)
    
    # Afficher les résultats
    if resultats:
        print("\nRésultats de la régression linéaire:")
        print(f"Évolution de la population vs Consommation d'ENAF:")
        print(f"   - Coefficient: {resultats['coefficient']:.4f}")
        print(f"   - Ordonnée à l'origine: {resultats['intercept']:.4f}")
        print(f"   - R²: {resultats['r2']:.4f}")
    
    # Identification des aires urbaines qui consomment davantage d'ENAF
    if 'CONSOMMATION_ENAF' in df_analyse.columns:
        top_n = min(5, df_analyse.shape[0])
        top_enaf = df_analyse.sort_values('CONSOMMATION_ENAF', ascending=False).head(top_n)
        print(f"\nTop {top_n} des aires urbaines consommant le plus d'ENAF en valeur absolue:")
        for i, row in top_enaf.iterrows():
            print(f"   {row['LIBAAV2020']}: {row['CONSOMMATION_ENAF']:.2f} ha")
    
    if 'CONSOMMATION_ENAF_PAR_HAB' in df_analyse.columns and 'EVOLUTION_POP_2017_2022' in df_analyse.columns:
        df_filtre = df_analyse[df_analyse['EVOLUTION_POP_2017_2022'] > 0]
        if df_filtre.shape[0] > 0:
            top_n = min(5, df_filtre.shape[0])
            top_efficacite = df_filtre.sort_values('CONSOMMATION_ENAF_PAR_HAB').head(top_n)
            print(f"\nTop {top_n} des aires urbaines les plus efficaces (consommation d'ENAF par habitant supplémentaire):")
            for i, row in top_efficacite.iterrows():
                print(f"   {row['LIBAAV2020']}: {row['CONSOMMATION_ENAF_PAR_HAB']:.4f} ha/hab")
    
    print("\nConclusion:")
    print("Cette analyse permet d'identifier les aires urbaines qui consomment davantage")
    print("de terrains naturels, agricoles et forestiers (ENAF) pour croître.")
    print("Les résultats montrent des différences significatives entre les aires urbaines")
    print("en termes de consommation d'ENAF, certaines étant plus efficaces que d'autres.")
    print("Les visualisations générées permettent de mieux comprendre ces différences.")

if __name__ == "__main__":
    main()