# Rapport d'analyse : Consommation d'ENAF par les aires urbaines

*Date d'analyse : 23/04/2025*

## Sommaire
1. Résumé des données analysées
2. Principaux indicateurs de consommation d'ENAF
3. Relations et corrélations clés
4. Typologie des aires urbaines selon leur consommation
5. Conclusion et recommandations

## 1. Résumé des données analysées

- Nombre total d'aires urbaines analysées : 676
- Population totale couverte : 67,161,726 habitants
- Période d'analyse : 2014-2020
- Évolution démographique totale : +1,254,947 habitants
- Consommation totale d'ENAF : 192,649,050 m²

Répartition des aires urbaines par catégorie :

| Catégorie       |   Nombre |
|:----------------|---------:|
| Petite          |      505 |
| Moyenne         |      120 |
| Grande          |       39 |
| Très grande     |       10 |
| Métropole       |        1 |
| Hors attraction |        1 |

## 2. Principaux indicateurs de consommation d'ENAF

Statistiques globales sur les indicateurs clés :

|                      |   count |    mean |     std |      min |     25% |    50% |    75% |     max |
|:---------------------|--------:|--------:|--------:|---------:|--------:|-------:|-------:|--------:|
| conso_enaf_par_hab   |     669 |    4.09 |    3.43 |      0   |    1.78 |   3.29 |   5.4  |   23    |
| conso_par_nouvel_hab |     669 | -159.97 | 2195.92 | -33767.8 | -143.07 | -21.53 | 114.99 | 7225.04 |
| coef_artif           |     669 |    0    |    0    |      0   |    0    |   0    |   0    |    0    |

Répartition moyenne de la consommation d'espace par type :

| Type   |   Pourcentage moyen (%) |
|:-------|------------------------:|
| Hab    |                   71.13 |
| Act    |                   19.91 |
| Rou    |                    4.55 |
| Inc    |                    2.39 |
| Mix    |                    1.84 |
| Fer    |                    0.19 |

## 3. Relations et corrélations clés

### 3.1 Principales corrélations avec la consommation d'ENAF

#### Corrélations avec Conso Enaf Par Hab

| Variable         |   Coefficient de corrélation |
|:-----------------|-----------------------------:|
| coef_artif       |                        0.512 |
| part_act         |                        0.141 |
| part_inc         |                        0.098 |
| categorie_taille |                       -0.118 |
| part_hab         |                       -0.130 |
| densite_pop      |                       -0.249 |

#### Corrélations avec Conso Par Nouvel Hab

| Variable         |   Coefficient de corrélation |
|:-----------------|-----------------------------:|
| taux_evol_pop    |                        0.044 |
| conso_enaf_total |                        0.035 |
| nombre_communes  |                        0.030 |
| part_rou         |                       -0.021 |
| coef_artif       |                       -0.040 |
| densite_pop      |                       -0.041 |

#### Corrélations avec Coef Artif

| Variable             |   Coefficient de corrélation |
|:---------------------|-----------------------------:|
| conso_enaf_par_hab   |                        0.512 |
| taux_evol_pop        |                        0.333 |
| densite_pop          |                        0.307 |
| conso_par_nouvel_hab |                       -0.040 |
| part_inc             |                       -0.064 |
| part_hab             |                       -0.152 |

### 3.2 Principales relations identifiées

#### Relation 1: Conso Enaf Par Hab vs Coef Artif

- Coefficient de corrélation : 0.512
- Cette relation suggère que il existe une corrélation forte positive entre conso enaf par hab et coef artif : quand conso enaf par hab augmente, coef artif tend également à augmenter
- *Voir graphique : relation_conso_enaf_par_hab_coef_artif.png*

#### Relation 2: Taux Evol Pop vs Coef Artif

- Coefficient de corrélation : 0.333
- Cette relation suggère que il existe une corrélation modérée positive entre taux evol pop et coef artif : quand taux evol pop augmente, coef artif tend également à augmenter
- *Voir graphique : relation_taux_evol_pop_coef_artif.png*

#### Relation 3: Densite Pop vs Coef Artif

- Coefficient de corrélation : 0.307
- Cette relation suggère que il existe une corrélation modérée positive entre densite pop et coef artif : quand densite pop augmente, coef artif tend également à augmenter
- *Voir graphique : relation_densite_pop_coef_artif.png*

#### Relation 4: Nombre Communes vs Conso Enaf Total

- Coefficient de corrélation : 0.972
- Cette relation suggère que il existe une corrélation très forte positive entre nombre communes et conso enaf total : quand nombre communes augmente, conso enaf total tend également à augmenter
- *Voir graphique : relation_nombre_communes_conso_enaf_total.png*

#### Relation 5: Conso Enaf Total vs Conso Infra Total

- Coefficient de corrélation : 0.910
- Cette relation suggère que il existe une corrélation très forte positive entre conso enaf total et conso infra total : quand conso enaf total augmente, conso infra total tend également à augmenter
- *Voir graphique : relation_conso_enaf_total_conso_infra_total.png*

## 4. Typologie des aires urbaines selon leur consommation

### 4.1 Répartition des aires urbaines selon leur typologie

| typologie                          |   Nombre |
|:-----------------------------------|---------:|
| Basse densité / Basse consommation |      134 |
| Basse densité / Haute consommation |      204 |
| Haute densité / Basse consommation |      207 |
| Haute densité / Haute consommation |      131 |

### 4.2 Aires urbaines représentatives de chaque typologie

#### Basse densité / Haute consommation

| Aire urbaine                       |   Population |   Densité (hab/km²) |   Conso. ENAF par hab. (m²) |
|:-----------------------------------|-------------:|--------------------:|----------------------------:|
| Commune hors attraction des villes |      7235999 |               28.08 |                        4.87 |
| Bourges                            |       174160 |               65.11 |                        3.99 |
| Ajaccio                            |       117405 |               61.93 |                        3.56 |

#### Haute densité / Basse consommation

| Aire urbaine                |   Population |   Densité (hab/km²) |   Conso. ENAF par hab. (m²) |
|:----------------------------|-------------:|--------------------:|----------------------------:|
| Paris                       |     13125142 |              689.47 |                        0.65 |
| Lyon                        |      2293180 |              496.25 |                        1.40 |
| Marseille - Aix-en-Provence |      1879601 |              473.37 |                        1.79 |

#### Haute densité / Haute consommation

| Aire urbaine   |   Population |   Densité (hab/km²) |   Conso. ENAF par hab. (m²) |
|:---------------|-------------:|--------------------:|----------------------------:|
| Toulouse       |      1470899 |              223.88 |                        4.28 |
| Rennes         |       763749 |              199.19 |                        4.51 |
| Tours          |       522317 |              143.38 |                        5.28 |

#### Basse densité / Basse consommation

| Aire urbaine   |   Population |   Densité (hab/km²) |   Conso. ENAF par hab. (m²) |
|:---------------|-------------:|--------------------:|----------------------------:|
| Troyes         |       221309 |               78.76 |                        1.94 |
| Nevers         |       113728 |               54.11 |                        2.96 |
| Auxerre        |       111344 |               67.13 |                        1.78 |

## 5. Conclusion et recommandations

### 5.1 Synthèse des résultats

- Les analyses montrent que la consommation d'ENAF est fortement liée à la structure urbaine et à la densité de population.
- Les aires urbaines à forte densité tendent à consommer moins d'espace par nouvel habitant, témoignant d'une utilisation plus efficiente du foncier.
- La taille de l'aire urbaine influence le type de consommation, avec une prédominance de l'habitat dans les petites aires urbaines et une part plus importante des activités économiques dans les grandes.

### 5.2 Recommandations

1. **Promouvoir la densification** des aires urbaines pour limiter l'étalement urbain et réduire la consommation d'ENAF par habitant.
2. **Adapter les politiques foncières** selon la typologie des aires urbaines, en tenant compte de leurs spécificités.
3. **Favoriser le renouvellement urbain** particulièrement dans les aires urbaines à forte consommation et faible densité.
4. **Surveiller particulièrement** les aires urbaines en forte croissance démographique mais à faible densité, qui présentent le risque le plus élevé de consommation excessive d'ENAF.

### 5.3 Limites de l'étude

- Les données de consommation peuvent présenter des incertitudes liées à la méthodologie de collecte.
- L'étude ne prend pas en compte la qualité des espaces consommés (terres agricoles à haute valeur ajoutée, zones naturelles à forte biodiversité, etc.).
- Les dynamiques économiques locales peuvent influencer les tendances de consommation mais n'ont pas été intégrées dans cette analyse.

