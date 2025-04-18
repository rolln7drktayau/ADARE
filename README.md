# ADARE: Adaptive Deep-reinforced Auto-learning Resource allocator Engine

<div align="center">
<img src="assets/adare-logo-refined.svg" width="300px">

*An intelligent multi-objective optimization solution for task scheduling in heterogeneous computing environments*
</div>

## 📑 Table des matières

- [Définition du problème](#🎯-définition-du-problème)
- [Métriques d'évaluation](#📊-métriques-dévaluation)
- [Algorithmes d'optimisation existants](#🔍-algorithmes-doptimisation-existants)
- [ADARE: Notre solution](#💡-adare-notre-solution)
- [Comparaison ADARE vs NSGA-III](#🔄-comparaison-adare-vs-nsga-iii)
- [Métriques de performance](#📏-métriques-de-performance)
- [Calcul des métriques](#🧮-calcul-des-métriques)
- [Sélection des solutions optimales](#🏆-sélection-des-solutions-optimales)
- [Architecture du code](#🏗️-architecture-du-code)
- [Arborescence du projet](#📂-arborescence-du-projet)
- [Exécution du projet](#🚀-exécution-du-projet)
- [Bibliothèques et dépendances](#📚-bibliothèques-et-dépendances)
- [Fichiers externes et générés](#📁-fichiers-externes-et-générés)
- [Contact et ressources](#📞-contact-et-ressources)

## 📝 Introduction

L'ordonnancement de tâches en environnement hétérogène est un problème complexe d'optimisation multi-objectif. Il s'agit d'affecter un ensemble de tâches interdépendantes à un ensemble de nœuds de calcul hétérogènes, tout en optimisant plusieurs critères contradictoires.

## 🎯 Définition du problème

Pour notre cas, nous considérons un ensemble de tâches $T$ à exécuter sur un ensemble de nœuds $N$. Chaque tâche $j \in T$ a un ensemble de prédécesseurs $P_j$ et un ensemble de successeurs $S_j$. Chaque nœud $n \in N$ a une capacité de calcul $C_n$ et une latence de communication $L_n$ avec les autres nœuds.

Ce problème est crucial dans les systèmes informatiques distribués modernes, où les applications sont décomposées en tâches qui doivent être exécutées sur différents types d'infrastructures, depuis les datacenters jusqu'aux appareils en périphérie du réseau.

## 📊 Métriques d'évaluation

Notre optimisation vise à minimiser quatre métriques clés:

| Métrique | Description | Formulation |
|----------|-------------|-------------|
| **Makespan** | Temps total d'exécution de l'ensemble des tâches (du début à la fin) | $M = \max_{j \in T}(FinExecution_j)$ |
| **Latence** | Temps moyen de réponse pour chaque tâche, incluant les délais de transfert | $L = \frac{1}{n} \sum_{j \in T}(FinExecution_j - TempsIdeal_j)$ |
| **Coût** | Coût financier total de l'exécution sur les ressources | $C = \sum_{j \in T, i \in N}(TempsExecution_{j,i} \times CoutTraitement_i)$ |
| **Énergie** | Consommation énergétique totale | $E = \sum_{j \in T, i \in N}(TempsExecution_{j,i} \times PuissanceTravail_i)$ |

*Où T est l'ensemble des tâches et N est l'ensemble des nœuds.*

## 🔍 Algorithmes d'optimisation existants

### Algorithmes multi-objectifs populaires

- **NSGA-II**: Algorithme génétique de tri non-dominé, utilisant l'élitisme et un opérateur de crowding-distance
- **MOEA/D**: Algorithme évolutionnaire multi-objectif basé sur la décomposition
- **SPEA2**: Algorithme évolutionnaire de force Pareto
- **MOPSO**: Optimisation par essaim particulaire multi-objectif

### NSGA-III en détail

NSGA-III (Non-dominated Sorting Genetic Algorithm III) est une extension de NSGA-II spécialement conçue pour les problèmes d'optimisation à nombreux objectifs (plus de 3).

#### Algorithme simplifié de NSGA-III

```markdown
Initialiser une population P de taille N
Évaluer chaque individu de P
Pour chaque génération g:
    Créer une population enfant Q via croisement et mutation
    Évaluer les individus de Q
    R = P ∪ Q
    Frontières = Tri-Non-Dominé(R)
    Nouvelle population P = ∅
    i = 1
    Tant que |P| + |Frontière_i| ≤ N:
        P = P ∪ Frontière_i
        i = i + 1
    Si |P| < N:
        Sélectionner (N - |P|) individus de Frontière_i en utilisant des points de référence
    g = g + 1
Retourner P
```

#### Points forts de NSGA-III

- Excellente gestion des problèmes avec de nombreux objectifs
- Bonne distribution des solutions sur le front de Pareto
- Convergence efficace vers le front de Pareto optimal

#### Faiblesses de NSGA-III

- Complexité de calcul élevée
- Paramétrage délicat des points de référence
- Opérateurs génétiques standard, sans adaptation au problème spécifique
- Difficulté à s'échapper des optima locaux

## 💡 ADARE: Notre solution

ADARE (Adaptive Deep-reinforced Auto-learning Resource allocator Engine) est notre algorithme adaptatif qui améliore NSGA-III en intégrant l'apprentissage par renforcement pour auto-ajuster ses opérateurs génétiques.

### Motivation

ADARE a été conçu pour surmonter les limitations de NSGA-III, notamment:

1. **L'adaptation dynamique**: Les opérateurs génétiques s'adaptent automatiquement au fur et à mesure de l'évolution
2. **L'exploration intelligente**: La force des mutations varie en fonction de la densité des solutions
3. **L'exploitation des connaissances**: Apprentissage des opérateurs les plus efficaces via Q-learning

### Algorithme Détaillé d'ADARE

```markdown
Initialiser:
    Population P ← GénérerPopulation(N)
    Q-table ← InitialiserQLearningOperateurs([cxOnePoint, cxTwoPoint, cxUniform, cxOrdered])
    ReferencePoints ← GénérerPointsRéférence(K, d)  # K divisions sur d objectifs
    ÉvaluerFitness(P)
    
    # Initialisation des archives d'élitisme
    MeilleuresSolutions ← { null pour chaque objectif }
    MeilleuresValeurs ← { ∞ pour chaque objectif }
    Historique ← [[] pour chaque objectif]

Pour génération = 1 à MAX_GEN:
    # Phase d'adaptation
    Densité ← CalculerDensité(P)
    α_eff ← 0.1 * (1 - génération/MAX_GEN)  # Taux d'apprentissage dynamique
    
    # Sélection des parents et génération des descendants
    Offspring ← SélectionAléatoire(P, N)
    Offspring ← Cloner(Offspring)
    
    # Application des opérateurs génétiques adaptatifs
    Pour chaque paire (enfant1, enfant2) dans Offspring:
        Si random() < CXPB:
            # Croisement par Q-Learning
            op_idx ← ε-greedy(Q-table, ε=0.2)
            op, params = Q-table.operateurs[op_idx]
            Appliquer op(enfant1, enfant2, **params)
            Invalider fitness de enfant1 et enfant2
    
    Pour chaque mutant dans Offspring:
        Si random() < MUTPB:
            # Mutation contextuelle
            niche_density ← |P|/NUM_NODES
            force_mutation ← (1 - niche_density) * (MAX_GEN - génération)/MAX_GEN
            MutationAdaptative(mutant, force_mutation, Tâches, Nœuds)
            Invalider fitness de mutant
    
    # Évaluation
    ÉvaluerFitness(Offspring)
    
    # Mise à jour Q-table basée sur l'amélioration
    Pour chaque opérateur utilisé:
        Δ_fitness ← CalculeAméliorationMoyenne(op)
        Q-table[op] ← (1 - α_eff) * Q-table[op] + α_eff * Δ_fitness
    
    # Sélection NSGA-III
    Combined ← P ∪ Offspring
    P ← NSGA-III(Combined, ReferencePoints, N)
    
    # Mise à jour des meilleures solutions (élitisme)
    Pour chaque individu ind dans P:
        makespan, latency, cost, energy ← ind.fitness.values
        
        Si makespan < MeilleuresValeurs['makespan']:
            MeilleuresValeurs['makespan'] ← makespan
            MeilleuresSolutions['makespan'] ← Cloner(ind)
        
        Si latency < MeilleuresValeurs['latency']:
            MeilleuresValeurs['latency'] ← latency
            MeilleuresSolutions['latency'] ← Cloner(ind)
        
        Si cost < MeilleuresValeurs['cost']:
            MeilleuresValeurs['cost'] ← cost
            MeilleuresSolutions['cost'] ← Cloner(ind)
        
        Si energy < MeilleuresValeurs['energy']:
            MeilleuresValeurs['energy'] ← energy
            MeilleuresSolutions['energy'] ← Cloner(ind)
    
    # Application de l'élitisme - réinsertion des meilleures solutions
    Si génération > 0:
        # Trier la population par somme des objectifs (approx. de dominance)
        P ← TrierParSommeObjectifs(P)
        
        # Remplacer les pires individus par les meilleures solutions connues
        Pour i, obj dans énumérer(['makespan', 'latency', 'cost', 'energy']):
            Si MeilleuresSolutions[obj] ≠ null ET MeilleuresSolutions[obj] ∉ P:
                P[-(i+1)] ← Cloner(MeilleuresSolutions[obj])  # Remplacer un des pires
    
    # Mise à jour de l'historique avec les meilleures valeurs connues
    Pour i, obj dans énumérer(['makespan', 'latency', 'cost', 'energy']):
        Historique[i].append(MeilleuresValeurs[obj])

# Finalisation - s'assurer que les meilleures solutions sont dans la population finale
Pour obj dans ['makespan', 'latency', 'cost', 'energy']:
    Si MeilleuresSolutions[obj] ≠ null ET MeilleuresSolutions[obj] ∉ P:
        idx ← random(0, |P|-1)
        P[idx] ← Cloner(MeilleuresSolutions[obj])

Retourner P, Historique

```

### Innovations clés

1. **Q-Learning pour les opérateurs de croisement**:
   - Chaque opérateur a une valeur Q qui est mise à jour en fonction de son efficacité
   - Sélection probabiliste basée sur ces valeurs Q (exploration ε-greedy)
   - Opérateurs multiples: cxOnePoint, cxTwoPoint, cxUniform (avec indpb=0.7), cxOrdered

2. **Mutation adaptative basée sur la densité**:
   - Force de mutation inversement proportionnelle à la densité des solutions
   - Favorise l'exploration dans les régions peu peuplées
   - Mutation intelligente basée sur les caractéristiques des tâches:
     - Pour tâches intensives en calcul: affectation à des nœuds plus puissants
     - Pour tâches avec beaucoup de données: affectation à des nœuds avec meilleure bande passante

3. **Adaptation dynamique**:
   - Taux d'apprentissage qui diminue au fil des générations
   - Force de mutation qui s'ajuste en fonction de l'avancement de l'algorithme

## 🔄 Comparaison ADARE vs NSGA-III

| Aspect | ADARE | NSGA-III |
|--------|-------|----------|
| **Adaptation** | ✅ Opérateurs auto-adaptatifs via Q-learning | ❌ Opérateurs fixes |
| **Exploration** | ✅ Mutation variable selon densité des solutions | ❌ Mutation uniforme |
| **Paramétrage** | ✅ Moins de paramètres à régler manuellement | ❌ Paramétrage manuel important |
| **Performance** | ✅ Meilleure sur makespan et coût | ✅ Bonne performance générale |
| **Temps d'exécution** | ✅ Plus rapide grâce au Q-learning | ❌ Plus lent à converger |
| **Complexité** | ❌ Plus complexe à implémenter | ✅ Implémentation plus simple |
| **Convergence** | ✅ Plus rapide vers front de Pareto | ❌ Convergence plus lente |

## 📏 Métriques de performance

Pour comparer ADARE et NSGA-III, nous utilisons quatre métriques de qualité standard:

| Métrique | Description | Importance |
|----------|-------------|------------|
| **Hypervolume (HV)** | Volume dominé par le front de Pareto par rapport à un point de référence | Mesure à la fois la convergence et la diversité |
| **Inverted Generational Distance (IGD)** | Distance moyenne du front de Pareto théorique au front obtenu | Évalue la proximité au front optimal |
| **Spread** | Mesure la distribution des solutions sur le front de Pareto | Évalue l'uniformité de la distribution |
| **Time** | Temps d'exécution de l'algorithme | Évalue l'efficacité computationnelle |

## 🧮 Calcul des métriques

### Métriques d'optimisation

1. **Makespan**:

   ```python
   makespan = max(task_info['end'] for task_info in task_schedule.values())
   ```

2. **Latence**:

   ```python
   latencies = [end_time - ideal_start for task_id, 
                (end_time, ideal_start) in task_schedules]
   avg_latency = np.mean(latencies)
   ```

3. **Coût**:

   ```python
   total_cost = sum(exec_time_hours * node['processing_cost'] 
                   for task, node, exec_time_hours in executed_tasks)
   ```

4. **Énergie**:

   ```python
   total_energy = sum(exec_time * node['working_power'] 
                     for task, node, exec_time in executed_tasks)
   ```

### Métriques de performance

1. **Hypervolume (HV)**:

   ```python
   ref_point = [1.5 * max(ind.fitness.values[i] for ind in population) for i in range(len(OBJ_NAMES))]
   hv = Hypervolume(ref_point=ref_point)
   hv_value = hv.do(np.array([ind.fitness.values for ind in population]))
   ```

2. **Inverted Generational Distance (IGD)**:

   ```python
   if true_pareto is not None:
       igd_calc = IGD(true_pareto)
       igd_value = igd_calc.do(np.array([ind.fitness.values for ind in population]))
   ```

3. **Spread**:

   ```python
   front = np.array([ind.fitness.values for ind in population])
   d1 = np.mean(np.min(np.abs(front - front.mean(axis=0)), axis=1))
   spread = d1 + np.sum(np.abs(front - front.mean(axis=0)))
   ```

4. **Convergence**:
   Pour chaque génération, nous traçons l'évolution des meilleures valeurs pour chaque métrique, ce qui permet de visualiser la vitesse de convergence de chaque algorithme.

## 🏆 Sélection des solutions optimales

La sélection des solutions optimales dans un contexte multi-objectif est basée sur le concept de dominance de Pareto:

1. **Dominance de Pareto**: Une solution domine une autre si elle est au moins aussi bonne dans tous les objectifs et strictement meilleure dans au moins un objectif.

2. **Front de Pareto**: Ensemble des solutions non-dominées, représentant les meilleurs compromis possibles.

Pour sélectionner la "meilleure" solution du front de Pareto:

- Nous utilisons `tools.selBest()` qui sélectionne les individus avec les meilleures valeurs de fitness
- Dans notre cas, les fonctions objectifs étant toutes à minimiser (poids négatifs dans la création de `FitnessMulti`), les meilleures solutions sont celles avec les plus petites valeurs objectives

Cette approche est pertinente car:

1. Elle ne favorise aucun objectif spécifique a priori
2. Elle propose un ensemble de solutions optimales parmi lesquelles un décideur peut choisir selon ses préférences
3. Elle permet de visualiser les compromis nécessaires entre les différents objectifs

## 🏗️ Architecture du code

### Fonctions principales

| Fonction | Description | Importance |
|----------|-------------|------------|
| `evaluate()` | Calcule les quatre objectifs pour un individu | Cœur de l'évaluation des solutions |
| `run_adare()` | Exécute l'algorithme ADARE | Implémentation principale de notre solution |
| `calculate_metrics()` | Calcule HV, IGD et Spread | Évaluation des performances |
| `run_comparison()` | Compare ADARE et NSGA-III | Analyse comparative |
| `visualize_schedule()` | Visualise l'ordonnancement des tâches | Représentation graphique des solutions |
| `plot_pareto_fronts()` | Affiche les fronts de Pareto | Visualisation des compromis |
| `plot_convergence()` | Affiche la convergence des algorithmes | Évaluation de la vitesse de convergence |
| `analyze_results_with_ai()` | Analyse les résultats avec IA | Interprétation avancée |

### Classes spéciales

| Classe | Description |
|--------|-------------|
| `QLearningCrossover` | Implémente l'apprentissage par renforcement pour les opérateurs de croisement |

## 📂 Arborescence du projet

```markdown
ADARE/
├── adare_vs_nsga3.py          # Script principal de comparaison ADARE vs NSGA-III
├── Makefile                   # Automatisation des tâches
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation du projet
├── data/
│   ├── algorithm_config.json  # Configuration des algorithmes
│   ├── environments.json      # Configuration des environnements de calcul
│   ├── benchmarks/            # Benchmarks formatés pour l'exécution
│   │   ├── CyberShake/        # Workflows CyberShake
│   │   ├── Montage/           # Workflows Montage
│   │   ├── Samples/           # Exemples de workflows
│   │   │   └── tasks.json     # Tâches d'exemple par défaut
│   ├── build/                 # Scripts de construction et formatage
│   │   ├── xml_to_json.py     # Convertisseur XML vers JSON
│   │   └── format.py          # Formateur de JSON pour ADARE
│   ├── history/               # Fichiers JSON intermédiaires
│   └── workflows/             # Workflows XML source
│       ├── CyberShake/        # Workflows CyberShake en XML
│       └── Montage/           # Workflows Montage en XML
├── output/                    # Résultats générés
│   ├── CyberShake/            # Résultats pour workflows CyberShake
│   │   ├── plots/             # Graphiques
│   │   └── reports/           # Rapports d'analyse
│   ├── Montage/               # Résultats pour workflows Montage
│   │   ├── plots/             # Graphiques
│   │   └── reports/           # Rapports d'analyse
│   └── Default/               # Résultats pour le workflow par défaut
│       ├── plots/             # Graphiques
│       └── reports/           # Rapports d'analyse
└── assets/                    # Ressources graphiques
    └── adare-logo-refined.svg # Logo ADARE
```

## 🚀 Exécution du projet

### Prérequis

- Python 3.8 ou supérieur
- Bibliothèques Python requises (voir `requirements.txt`)

### Installation

1. Clonez le dépôt:

   ```bash
   git clone https://github.com/rolln7drktayau/ADARE.git
   cd ADARE
   ```

2. Créez et activez un environnement virtuel (recommandé):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Sur Windows: myenv\Scripts\activate par exemple
   ```

3. Installez les dépendances:

   ```bash
   make setup
   # ou manuellement:
   pip install -r requirements.txt
   ```

### Commandes disponibles

Le projet utilise un Makefile pour simplifier l'exécution:

| Commande | Description |
|----------|-------------|
| `make run` | Exécute ADARE avec le jeu de données par défaut |
| `make run-CyberShake_30` | Exécute ADARE avec le workflow CyberShake de 30 tâches |
| `make run-Montage_25` | Exécute ADARE avec le workflow Montage de 25 tâches |
| `make clean` | Nettoie les fichiers générés |
| `make test` | Exécute les tests |
| `make format` | Formate le code avec black |
| `make lint` | Vérifie le style du code avec flake8 |
| `make help` | Affiche l'aide sur les commandes disponibles |

### Exemples d'utilisation

1. Exécuter avec le jeu de données par défaut:

   ```bash
   make run
   ```

2. Exécuter avec un workflow spécifique:

   ```bash
   make run-CyberShake_30
   ```

3. Nettoyer les fichiers générés:

   ```bash
   make clean
   ```

### Structure des résultats

Les résultats sont organisés par type de workflow:

- `output/CyberShake/plots/`: Graphiques pour les workflows CyberShake
- `output/CyberShake/reports/`: Rapports pour les workflows CyberShake
- `output/Montage/plots/`: Graphiques pour les workflows Montage
- `output/Montage/reports/`: Rapports pour les workflows Montage
- `output/Default/plots/`: Graphiques pour le workflow par défaut
- `output/Default/reports/`: Rapports pour le workflow par défaut

Chaque dossier de plots contient:

- `convergence.png`: Courbes de convergence
- `pareto_2d.png`: Fronts de Pareto en 2D
- `pareto_3d.png`: Fronts de Pareto en 3D
- `schedule_ADARE.png`: Ordonnancement des tâches par ADARE
- `schedule_NSGA-III.png`: Ordonnancement des tâches par NSGA-III

1. **Mettez à jour la section "Fichiers externes et générés"** avec cette version:

## 📁 Fichiers externes et générés

### Fichiers d'entrée

| Fichier | Description |
|---------|-------------|
| `data/algorithm_config.json` | Configuration des algorithmes (générations, taille de population, etc.) |
| `data/environments.json` | Description des environnements de calcul (cloud, edge, fog) |
| `data/benchmarks/*/tasks.json` | Description des tâches à ordonnancer |
| `data/workflows/*/*.xml` | Workflows au format XML (source) |

### Fichiers de sortie

| Dossier | Description |
|---------|-------------|
| `output/*/plots/` | Graphiques générés (Pareto, convergence, ordonnancements) |
| `output/*/reports/` | Rapports de comparaison des performances |
| `data/history/` | Fichiers JSON intermédiaires lors de la conversion |

Ces modifications complètent votre README.md avec:

1. Une arborescence détaillée du projet
2. Un algorithme ADARE mis à jour et plus complet
3. Des instructions claires pour l'exécution du projet
4. Des informations sur les nouveaux chemins de sortie
5. Des détails sur la structure des résultats

## 📚 Bibliothèques et fonctions externes utilisées

### DEAP (Distributed Evolutionary Algorithms in Python)

| Fonction | Description | Prototype |
|----------|-------------|-----------|
| `algorithms.eaMuPlusLambda` | Algorithme évolutionnaire (μ+λ) | `eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__)` |
| `tools.selNSGA3` | Sélection NSGA-III | `selNSGA3(individuals, k, ref_points, nd='standard')` |
| `tools.selTournamentDCD` | Sélection par tournoi avec crowding-distance | `selTournamentDCD(individuals, k)` |
| `tools.uniform_reference_points` | Génère des points de référence uniformes | `uniform_reference_points(nobj, p=None, scaling=None)` |
| `tools.emo.assignCrowdingDist` | Calcule la crowding-distance | `assignCrowdingDist(individuals)` |

### Pymoo

| Fonction | Description | Prototype |
|----------|-------------|-----------|
| `Hypervolume` | Calcule l'hypervolume | `Hypervolume(ref_point)` |
| `IGD` | Calcule l'Inverted Generational Distance | `IGD(true_front)` |

### Matplotlib

Utilisé pour créer toutes les visualisations, incluant:

- Diagrammes de Gantt pour l'ordonnancement
- Graphiques 2D et 3D des fronts de Pareto
- Courbes de convergence

### NumPy

Utilisé pour les calculs numériques efficaces et les manipulations de tableaux.

## Interpreting Results

### Performance Metrics Comparison

The direct comparison between ADARE and NSGA-III shows:

```markdown
Metric          | ADARE                | NSGA-III             | Improvement
---------------------------------------------------------------------------
Makespan        | 26947.12             | 27197.12             | +0.9%
Latence         | 1665.50              | 1676.04              | +0.6%
Coût            | 35595.38             | 35186.92             | -1.2%
Énergie         | 63195769.23          | 63683846.15          | +0.8%
```

- **Negative values** indicate ADARE performed worse than NSGA-III
- **Positive values** indicate ADARE performed better than NSGA-III
- Values close to 0% indicate similar performance

In this case:

- ADARE shows slightly worse makespan (-15.7%) and latency (-0.2%)
- But achieves better cost optimization (+0.3%)
- With slightly higher energy consumption (-2.5%)

### Statistical Metrics Analysis

```markdown
Metric          | ADARE                | NSGA-III             | Improvement
---------------------------------------------------------------------------
hv              | 3.17e+21 ± 6.55e+20  | 2.78e+21 ± 3.58e+20  | 14.1%
igd             | 2.65e+05 ± 4.99e+04  | 3.07e+05 ± 6.67e+04  | -13.8%
spread          | 2.57e+08 ± 1.92e+07  | 2.47e+08 ± 1.63e+07  | 4.1%
time            | 0.55 ± 0.02          | 0.57 ± 0.01          | -4.4%
```

Format: `mean ± standard_deviation`

- **Hypervolume (hv)**: ADARE shows 6.3% better coverage of the objective space
- **IGD**: ADARE has 8.0% larger distance to true Pareto front
- **Spread**: Similar diversity with ADARE showing 1.3% more spread
- **Time**: ADARE is 5.2% faster in execution

These results suggest ADARE offers better solution diversity and execution speed, with some trade-offs in specific performance metrics.

## 📞 Contact et ressources

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rct/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rolln7drktayau/)

### Encadrement

Dr. Sonia Yassa  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yassa-sonia-1a840625/)
[![Overleaf](https://img.shields.io/badge/Overleaf-47A141?style=for-the-badge&logo=Overleaf&logoColor=white)](https://www.overleaf.com/)
[![CYU](https://img.shields.io/badge/Visit_Card-blue?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=)](https://www.etis-lab.fr/2022/02/25/yassa-sonia/)

---

*Ce projet a été développé dans le cadre des recherches sur l'optimisation multi-objectif pour l'informatique distribuée à CY Cergy Paris Université.*
