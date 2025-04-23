import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

def main():
    """
    Fonction principale d'exécution
    """
    # Chemins des fichiers avec gestion de chemin relatif
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chemin_au = os.path.join(base_dir, 'inputs', 'csv', 'AAV2020_au_01-01-2024.csv')
    chemin_csp = os.path.join(base_dir, 'inputs', 'csv', 'conso2009_2023_resultats_com.csv')
    
    print(f"Chemins des fichiers:\n - {chemin_au}\n - {chemin_csp}")
    
    # 1. Chargement des données
    df_au, df_csp = load_data(chemin_au, chemin_csp)
    if df_au is None or df_csp is None:
        return
    
    # 2. Préparation des données d'aires urbaines
    df_au_clean = nettoyer_donnees_au(df_au)
    
    # 3. Identification des colonnes ENAF et artificialisation
    col_enaf, col_artif = identifier_colonnes_enaf_artif(df_csp)
    
    # 4. Préparation des données ENAF et artificialisation
    df_enaf_artif_clean = preparer_donnees_enaf_artif(df_csp, col_enaf, col_artif)
    
    # 5. Agrégation par aire urbaine pour ENAF et artificialisation
    df_enaf_artif_agg = df_enaf_artif_clean.groupby('AAV').agg({
        'pop20': 'sum',
        'conso_enaf_hab': 'mean',
        'coef_artif': 'mean'
    }).reset_index()
    
    # 6. Fusion des données
    df_final = fusionner_donnees(df_au_clean, df_enaf_artif_agg)
    
    # 7. Création de visualisations pour ENAF et artificialisation
    creer_visualisations_enaf_artif(df_final)
    
    # 8. Export des résultats
    stats_desc, corr_matrix = exporter_resultats(df_final)
    
    # 9. Affichage des principaux résultats
    print("\nPrincipales statistiques :")
    print(stats_desc.loc[['mean', 'std', 'min', 'max']])
    
    print("\nPrincipales corrélations :")
    print(corr_matrix.iloc[:5, :5])
    
    print("\nAnalyse terminée avec succès!")

def creer_visualisations_enaf_artif(df_final, dossier_sortie="resultats"):
    """
    Crée des visualisations spécifiques à l'analyse ENAF et artificialisation
    """
    # Création du dossier de sortie si nécessaire
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Corrélation entre consommation ENAF par habitant et coefficient d'artificialisation
    plt.figure(figsize=(14, 8))
    
    # Graphique de dispersion
    scatter = sns.scatterplot(
        data=df_final, 
        x='conso_enaf_hab', 
        y='coef_artif',
        size='population_totale',
        sizes=(20, 500),
        hue='categorie_taille',
        palette='viridis',
        alpha=0.7
    )
    
    # Ajout d'une ligne de tendance avec intervalle de confiance
    sns.regplot(
        data=df_final, 
        x='conso_enaf_hab', 
        y='coef_artif', 
        scatter=False, 
        color='red', 
        line_kws={"linewidth": 2}
    )
    
    # Annotation des aires urbaines avec les valeurs les plus élevées
    for idx, row in df_final.nlargest(5, 'conso_enaf_hab').iterrows():
        plt.annotate(
            row['nom_aire_urbaine'],
            xy=(row['conso_enaf_hab'], row['coef_artif']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.title("Relation entre consommation ENAF par habitant et coefficient d'artificialisation", fontsize=16)
    plt.xlabel("Consommation ENAF par habitant (m²)", fontsize=14)
    plt.ylabel("Coefficient d'artificialisation", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Catégorie de taille", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Sauvegarde de la figure
    plt.savefig(f"{dossier_sortie}/relation_enaf_artif.png", dpi=300)
    
    # Figure 2: Distribution des variables par catégorie de taille
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Distribution de la consommation ENAF par habitant
    sns.boxplot(
        data=df_final,
        x='categorie_taille',
        y='conso_enaf_hab',
        palette='viridis',
        ax=axes[0]
    )
    axes[0].set_title("Distribution de la consommation ENAF par habitant\nselon la taille de l'aire urbaine", fontsize=14)
    axes[0].set_xlabel("Catégorie de taille d'aire urbaine", fontsize=12)
    axes[0].set_ylabel("Consommation ENAF par habitant (m²)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Distribution du coefficient d'artificialisation
    sns.boxplot(
        data=df_final,
        x='categorie_taille',
        y='coef_artif',
        palette='viridis',
        ax=axes[1]
    )
    axes[1].set_title("Distribution du coefficient d'artificialisation\nselon la taille de l'aire urbaine", fontsize=14)
    axes[1].set_xlabel("Catégorie de taille d'aire urbaine", fontsize=12)
    axes[1].set_ylabel("Coefficient d'artificialisation", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    plt.savefig(f"{dossier_sortie}/distribution_enaf_artif_par_categorie.png", dpi=300)
    
    # Calcul et affichage du coefficient de corrélation
    corr_coef = df_final['conso_enaf_hab'].corr(df_final['coef_artif'])
    print(f"Coefficient de corrélation entre consommation ENAF par habitant et coefficient d'artificialisation: {corr_coef:.4f}")
    
    # Affichage des figures
    plt.show()
    
    return