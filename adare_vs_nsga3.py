import json
import numpy as np
import time
import random
import sys
import os
from deap import algorithms, base, creator, tools # type: ignore
# Fix for pymoo imports
from pymoo.indicators.hv import Hypervolume # type: ignore
from scipy.spatial.distance import directed_hausdorff # type: ignore
from functools import partial # type: ignore
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
from matplotlib.patches import Patch # type: ignore
import g4f # type: ignore

# ==================================================================
# CHARGEMENT DES FICHIERS DE CONFIGURATION
# ==================================================================
with open('data/algorithm_config.json') as f:
    config = json.load(f)

with open('data/environments.json') as f:
    environments = json.load(f)

# Determine which tasks file to use based on command-line argument
if len(sys.argv) > 1:
    benchmark_name = sys.argv[1]
    workflow_type, workflow_size = benchmark_name.split('_')
    tasks_file = f'data/benchmarks/{workflow_type}/{benchmark_name}.json'
    print(f"Workflow Type : {workflow_type}, Workflow Size : {workflow_size}")
    print(f"Using tasks file: {tasks_file}")
else:
    tasks_file = 'data/benchmarks/Samples/tasks.json'

# Check if the file exists
if not os.path.exists(tasks_file):
    print(f"Error: Tasks file {tasks_file} not found.")
    sys.exit(1)

# Load the tasks data
with open(tasks_file) as f:
    tasks_data = json.load(f)

print(f"Using tasks data from: {tasks_file}")

# ==================================================================
# CONFIGURATION BASÉE SUR LES FICHIERS
# ==================================================================
NUM_TASKS = len(tasks_data['tasks'])
NUM_NODES = sum(env['devices'] for env in environments.values())
OBJ_NAMES = config['nsga3']['objectives']
NICHE_COUNT = config['nsga3']['reference_points_divisions']
MAX_GEN = config['nsga3']['generations']
POP_SIZE = config['nsga3']['population_size']
CXPB = config['nsga3']['crossover_probability']
MUTPB = config['nsga3']['mutation_probability']

# Configuration des nœuds
NODES = []
node_id = 0
for env_type, env in environments.items():
    for _ in range(env['devices']):
        NODES.append({
            'id': node_id,
            'type': env_type,
            'processing_rate': env['processing_rate'],
            'processing_cost': env['processing_cost'],
            'idle_power': env['idle_power'],
            'working_power': env['working_power'],
            'uplink_bandwidth': env['uplink_bandwidth'],
            'downlink_bandwidth': env['downlink_bandwidth']
        })
        node_id += 1

# Configuration des tâches
TASKS = tasks_data['tasks']

# ==================================================================
# INITIALISATION DEAP
# ==================================================================
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * len(OBJ_NAMES))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_node", np.random.randint, 0, NUM_NODES)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_node, n=NUM_TASKS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==================================================================
# FONCTION D'ÉVALUATION
# ==================================================================
def evaluate(individual):
    try:
        # Initialisation des structures de données
        node_available_time = {n['id']: 0 for n in NODES}
        task_schedule = {}
        metrics = {
            'makespan': 0,
            'latency': 0,
            'cost': 0,
            'energy': 0
        }

        # Tri topologique des tâches selon les dépendances
        sorted_tasks = []
        visited = set()
        
        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in TASKS[task_id]['dependencies']:
                visit(dep - 1)
            sorted_tasks.append(task_id)
        
        for i in range(len(TASKS)):
            visit(i)

        # Ordonnancement des tâches
        for task_id in sorted_tasks:
            node_id = individual[task_id]
            node = NODES[node_id]
            task = TASKS[task_id]
            
            # Calcul du temps de début
            if task['dependencies']:
                start_time = max(task_schedule[dep-1]['end'] for dep in task['dependencies'])
            else:
                start_time = node_available_time[node_id]
            
            # Calcul des temps d'exécution et de transfert
            exec_time = task['instructions'] / node['processing_rate']
            transfer_time = task['data_size'] / node['uplink_bandwidth']
            end_time = start_time + exec_time + transfer_time
            
            # Mise à jour des disponibilités
            node_available_time[node_id] = end_time
            task_schedule[task_id] = {
                'start': start_time,
                'end': end_time,
                'exec_time': exec_time
            }
            
            # Calcul des métriques
            metrics['makespan'] = max(metrics['makespan'], end_time)
            metrics['latency'] += end_time - start_time
            metrics['cost'] += exec_time * node['processing_cost']
            metrics['energy'] += exec_time * node['working_power']

        return (metrics['makespan'], 
                metrics['latency'], 
                metrics['cost'], 
                metrics['energy'])
    
    except Exception as e:
        print(f"Erreur dans l'évaluation: {str(e)}")
        return (float('inf'),) * len(OBJ_NAMES)

toolbox.register("evaluate", evaluate)

# ==================================================================
# OPÉRATEURS GÉNÉTIQUES ADARE
# ==================================================================
class QLearningCrossover:
    def __init__(self):
        # Enhanced operators with more variation
        self.operators = [
            (tools.cxOnePoint, {}),
            (tools.cxTwoPoint, {}),
            # (tools.cxUniform, {'indpb': 0.5}),
            (tools.cxUniform, {'indpb': 0.7}),  # More aggressive uniform crossover
            (tools.cxOrdered, {})  # Adding ordered crossover for better exploitation
        ]
        self.q_table = [1.0] * len(self.operators)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor for future rewards
        self.last_op_index = None
        
    def select_operator(self):
        # Epsilon-greedy selection
        if np.random.random() < 0.2:  # 20% exploration chance
            op_index = np.random.choice(len(self.operators))
        else:
            op_index = np.argmax(self.q_table)
            
        self.last_op_index = op_index
        return op_index
    
    def update_q(self, improvement, generation, max_gen):
        if self.last_op_index is not None:
            # Dynamic learning rate that decreases over generations
            effective_alpha = self.alpha * (1 - generation / max_gen)
            # Update Q-value with more weight on recent improvements
            self.q_table[self.last_op_index] = (1 - effective_alpha) * self.q_table[self.last_op_index] + effective_alpha * improvement
            self.last_op_index = None

q_crossover = QLearningCrossover()

def adare_mate(ind1, ind2):
    op_index = q_crossover.select_operator()
    operator, params = q_crossover.operators[op_index]
    operator(ind1, ind2, **params)
    return ind1, ind2

def adaptive_mutation(individual, density=0.5, gen=0, max_gen=MAX_GEN):
    # Calcul du facteur d'exploration
    exploration_factor = 1 - (gen / max_gen)
    
    # Force de mutation basée sur la densité
    mutation_strength = max(1, int((1 - density) * len(individual) * exploration_factor))
    
    for _ in range(mutation_strength):
        pos = random.randint(0, len(individual) - 1)
        
        # Mutation intelligente basée sur les caractéristiques des tâches
        if random.random() < 0.8 and pos < len(TASKS):
            task = TASKS[pos]
            current_node = individual[pos]
            
            # Pour tâches intensives en calcul
            if task['instructions'] > 500000:
                suitable = [n['id'] for n in NODES 
                          if n['processing_rate'] > NODES[current_node]['processing_rate']]
                if suitable:
                    individual[pos] = random.choice(suitable)
                    continue
            
            # Pour tâches avec beaucoup de données
            if task['data_size'] > 1000:
                suitable = [n['id'] for n in NODES 
                          if n['uplink_bandwidth'] > NODES[current_node]['uplink_bandwidth']]
                if suitable:
                    individual[pos] = random.choice(suitable)
                    continue
        
        # Mutation aléatoire standard
        individual[pos] = random.randint(0, NUM_NODES - 1)
    
    return individual,

# ==================================================================
# ALGORITHME ADARE COMPLET
# ==================================================================
def run_adare():
    # Initialisation
    population = toolbox.population(n=POP_SIZE)
    reference_points = tools.uniform_reference_points(len(OBJ_NAMES), NICHE_COUNT)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    
    # Évaluation initiale
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Historique des métriques
    history = [[] for _ in range(len(OBJ_NAMES))]
    start_time = time.time()
    
    q_crossover = QLearningCrossover()
    
    # Archive pour stocker les meilleures solutions par objectif
    best_solutions = {
        'makespan': None,
        'latency': None,
        'cost': None,
        'energy': None
    }
    best_values = {
        'makespan': float('inf'),
        'latency': float('inf'),
        'cost': float('inf'),
        'energy': float('inf')
    }
    
    for gen in range(MAX_GEN):
        try:
            # Nouvelle méthode de sélection sans crowding_dist
            offspring = tools.selRandom(population, len(population))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            # Application des opérateurs génétiques
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    adare_mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    niche_density = len(population)/NUM_NODES
                    adaptive_mutation(mutant, niche_density, gen, MAX_GEN)
                    del mutant.fitness.values
            
            # Évaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Sélection NSGA-III
            combined = population + offspring
            for ind in combined:
                if not hasattr(ind.fitness, 'crowding_dist'):
                    ind.fitness.crowding_dist = 0.0  # Initialiser si nécessaire
            
            # Sélection avec élitisme
            population = tools.selNSGA3(combined, POP_SIZE, reference_points)
            
            # Mise à jour des meilleures solutions par objectif
            for ind in population:
                makespan, latency, cost, energy = ind.fitness.values
                
                if makespan < best_values['makespan']:
                    best_values['makespan'] = makespan
                    best_solutions['makespan'] = toolbox.clone(ind)
                
                if latency < best_values['latency']:
                    best_values['latency'] = latency
                    best_solutions['latency'] = toolbox.clone(ind)
                
                if cost < best_values['cost']:
                    best_values['cost'] = cost
                    best_solutions['cost'] = toolbox.clone(ind)
                
                if energy < best_values['energy']:
                    best_values['energy'] = energy
                    best_solutions['energy'] = toolbox.clone(ind)
            
            # Assurer l'élitisme en remplaçant les pires individus par les meilleurs connus
            if gen > 0:  # Commencer après la première génération
                # Trier la population par dominance
                population.sort(key=lambda ind: sum(ind.fitness.values))
                
                # Remplacer les 4 pires individus par les meilleurs connus pour chaque objectif
                for i, obj in enumerate(['makespan', 'latency', 'cost', 'energy']):
                    if best_solutions[obj] is not None:
                        # Vérifier si la meilleure solution n'est pas déjà dans la population
                        if best_solutions[obj] not in population:
                            # Remplacer un des pires individus
                            population[-(i+1)] = toolbox.clone(best_solutions[obj])
            
            # Mise à jour de l'historique avec les meilleures valeurs connues
            for i, obj in enumerate(['makespan', 'latency', 'cost', 'energy']):
                history[i].append(best_values[obj])
                
        except Exception as e:
            print(f"Erreur génération {gen}: {str(e)}")
            # Remplir l'historique avec la dernière valeur valide
            for i in range(len(OBJ_NAMES)):
                history[i].append(history[i][-1] if history[i] else 0)
            continue
    
    # Ajouter les meilleures solutions connues à la population finale
    for obj in ['makespan', 'latency', 'cost', 'energy']:
        if best_solutions[obj] is not None and best_solutions[obj] not in population:
            # Remplacer un individu aléatoire
            idx = random.randint(0, len(population) - 1)
            population[idx] = best_solutions[obj]
    
    return population, history, time.time() - start_time

# ==================================================================
# ALGORITHME NSGA-III STANDARD
# ==================================================================
def run_nsga3():
    # Créer une toolbox dédiée pour NSGA-III
    nsga_toolbox = base.Toolbox()
    nsga_toolbox.register("attr_node", np.random.randint, 0, NUM_NODES)
    nsga_toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         nsga_toolbox.attr_node, n=NUM_TASKS)
    nsga_toolbox.register("population", tools.initRepeat, list, nsga_toolbox.individual)
    nsga_toolbox.register("evaluate", evaluate)
    
    # Opérateurs standard (sans les améliorations ADARE)
    nsga_toolbox.register("mate", tools.cxTwoPoint)
    nsga_toolbox.register("mutate", tools.mutUniformInt, 
                         low=0, up=NUM_NODES-1, indpb=0.1)
    nsga_toolbox.register("select", tools.selNSGA3)
    
    # Points de référence
    reference_points = tools.uniform_reference_points(len(OBJ_NAMES), NICHE_COUNT)
    
    # Historique des métriques
    history = [[] for _ in range(len(OBJ_NAMES))]
    start_time = time.time()
    
    # Initialisation de la population
    population = nsga_toolbox.population(n=POP_SIZE)
    
    # Évaluation initiale
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = nsga_toolbox.map(nsga_toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    for gen in range(MAX_GEN):
        try:
            # Génération des descendants
            offspring = algorithms.varOr(
                population, nsga_toolbox,
                lambda_=POP_SIZE,
                cxpb=CXPB,
                mutpb=MUTPB
            )
            
            # Évaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = nsga_toolbox.map(nsga_toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Sélection
            population = tools.selNSGA3(population + offspring, POP_SIZE, reference_points)
            
            # Enregistrement des métriques : Historique NSGA-III
            best_ind = tools.selBest(population, 1)[0]
            for i in range(len(OBJ_NAMES)):
                history[i].append(best_ind.fitness.values[i])
                
        except Exception as e:
            print(f"Erreur dans NSGA-III (gen {gen}): {str(e)}")
            continue
    
    return population, history, time.time() - start_time

# ==================================================================
# MÉTRIQUES DE COMPARAISON
# ==================================================================
def calculate_metrics(population, true_pareto=None):
    # Hypervolume
    ref_point = [1.5 * max(ind.fitness.values[i] for ind in population) for i in range(len(OBJ_NAMES))]
    hv = Hypervolume(ref_point=ref_point)
    hv_value = hv.do(np.array([ind.fitness.values for ind in population]))
    
    # Hausdorff Distance
    hausdorff_value = None
    if true_pareto is not None:
        pop_front = np.array([ind.fitness.values for ind in population])
        hausdorff_1 = directed_hausdorff(pop_front, true_pareto)[0]
        hausdorff_2 = directed_hausdorff(true_pareto, pop_front)[0]
        hausdorff_value = max(hausdorff_1, hausdorff_2)
    
    # Spread (mesure de diversité)
    front = np.array([ind.fitness.values for ind in population])
    if len(front) > 1:
        d_mean = np.mean(np.linalg.norm(front - front.mean(axis=0), axis=1))
        spread = d_mean + np.sum(np.abs(front - front.mean(axis=0)))
    else:
        spread = 0.0
    
    return hv_value, hausdorff_value, spread

def adaptive_hyperparameter_tuning(initial_config):
    """Fonction d'auto-réglage des hyperparamètres pour ADARE"""
    # Définir une plage de paramètres à tester avec des combinaisons valides
    param_ranges = {
        'reference_points_divisions': [4, 6, 8, 10, 12],
        'population_size': [50, 100, 150, 200],
        'crossover_probability': [0.6, 0.7, 0.8],
        'mutation_probability': [0.1, 0.15, 0.2]
    }
    
    best_hv = 0
    best_config = initial_config.copy()
    
    print("Lancement de l'optimisation automatique des hyperparamètres...")
    
    # Enregistrer les opérateurs de base
    if hasattr(toolbox, 'mate'):
        toolbox.unregister("mate")
    if hasattr(toolbox, 'mutate'):
        toolbox.unregister("mutate")
        
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_NODES-1, indpb=0.1)
    toolbox.register("select", tools.selNSGA3)
    
    # Pour chaque paramètre, tester individuellement les valeurs
    for param, values in param_ranges.items():
        print(f"\nOptimisation de {param}...")
        current_config = best_config.copy()
        
        for value in values:
            # Vérifier la validité des probabilités
            if param == 'crossover_probability':
                if value + best_config['mutation_probability'] > 1.0:
                    print(f"  Combinaison invalide: CXPB={value} + MUTPB={best_config['mutation_probability']} > 1.0")
                    continue
            elif param == 'mutation_probability':
                if best_config['crossover_probability'] + value > 1.0:
                    print(f"  Combinaison invalide: CXPB={best_config['crossover_probability']} + MUTPB={value} > 1.0")
                    continue
            
            # Mettre à jour la configuration
            current_config[param] = value
            
            # Mettre à jour les variables globales
            global NICHE_COUNT, POP_SIZE, CXPB, MUTPB
            if param == 'reference_points_divisions':
                NICHE_COUNT = value
            elif param == 'population_size':
                POP_SIZE = value
            elif param == 'crossover_probability':
                CXPB = value
            elif param == 'mutation_probability':
                MUTPB = value
                
            print(f"  Test avec {param}={value} (CXPB={CXPB}, MUTPB={MUTPB})")
            
            # Régénérer les points de référence
            reference_points = tools.uniform_reference_points(len(OBJ_NAMES), NICHE_COUNT)
            
            # Exécuter une version courte d'ADARE
            population = toolbox.population(n=POP_SIZE)
            
            # Évaluation initiale
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Exécuter quelques générations pour tester
            for gen in range(5):  # Réduit pour accélérer le tuning
                offspring = algorithms.varOr(
                    population, 
                    toolbox,
                    lambda_=POP_SIZE,
                    cxpb=CXPB,
                    mutpb=MUTPB
                )
                
                # Évaluer les nouveaux individus
                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                
                # Sélection NSGA-III
                population = tools.selNSGA3(population + offspring, POP_SIZE, reference_points)
            
            # Calculer l'hypervolume
            try:
                front = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
                if len(front) > 0:
                    ref_point = np.max(front, axis=0) * 1.2
                    hv = Hypervolume(ref_point=ref_point)
                    hv_value = hv.do(front)
                else:
                    hv_value = 0
            except Exception as e:
                print(f"  Erreur de calcul HV: {str(e)}")
                hv_value = 0
            
            print(f"  {param}={value}: HV={hv_value:.4f}")
            
            # Mettre à jour la meilleure configuration
            if hv_value > best_hv:
                best_hv = hv_value
                best_config[param] = value
                print(f"  ! Nouvelle meilleure configuration !")

    print("\nOptimisation terminée. Configuration optimisée:")
    print(json.dumps(best_config, indent=4))
    
    # Restaurer les opérateurs ADARE pour l'exécution finale
    toolbox.register("mate", adare_mate)
    toolbox.register("mutate", adaptive_mutation)
    
    return best_config

# ==================================================================
# COMPARAISON ALGORITHMES
# ==================================================================
def run_comparison(runs=10):
    """Exécute la comparaison entre ADARE et NSGA-III sur plusieurs runs"""
    metrics = {
        'adare': {'hv': [], 'hausdorff': [], 'spread': [], 'time': []},
        'nsga3': {'hv': [], 'hausdorff': [], 'spread': [], 'time': []}
    }
    
    try:
        # Génération du front de référence
        print("Génération du front de référence...")
        reference_front = []
        for _ in range(3):  # 3 exécutions suffisent pour le front de référence
            adare_pop, _, _ = run_adare()
            nsga_pop, _, _ = run_nsga3()
            reference_front.extend([ind.fitness.values for ind in adare_pop + nsga_pop])
        
        # Filtrage des solutions non-dominées
        reference_front = np.unique(np.array(reference_front), axis=0)
        reference_front = reference_front[np.argsort(reference_front[:, 0])]  # Tri par makespan
        
        # Points de référence pour le calcul HV
        ref_point = np.max(reference_front, axis=0) * 1.1
        
        # Exécutions de comparaison
        for run in range(1, runs+1):
            print(f"\nExécution {run}/{runs}")
            
            # ADARE
            start = time.time()
            adare_pop, _, _ = run_adare()
            adare_time = time.time() - start
            hv, hd, sp = calculate_metrics(adare_pop, reference_front)
            metrics['adare']['hv'].append(hv)
            metrics['adare']['hausdorff'].append(hd)
            metrics['adare']['spread'].append(sp)
            metrics['adare']['time'].append(adare_time)
            
            # NSGA-III
            start = time.time()
            nsga_pop, _, _ = run_nsga3()
            nsga_time = time.time() - start
            hv, hd, sp = calculate_metrics(nsga_pop, reference_front)
            metrics['nsga3']['hv'].append(hv)
            metrics['nsga3']['hausdorff'].append(hd)
            metrics['nsga3']['spread'].append(sp)
            metrics['nsga3']['time'].append(nsga_time)
        
        # Préparation des résultats
        results = []
        for metric in ['hv', 'hausdorff', 'spread', 'time']:
            adare_mean = np.mean(metrics['adare'][metric])
            nsga_mean = np.mean(metrics['nsga3'][metric])
            
            if metric == 'hausdorff':  # Plus petit est mieux
                improvement = (nsga_mean - adare_mean)/nsga_mean * 100
            else:  # Plus grand est mieux
                improvement = (adare_mean - nsga_mean)/nsga_mean * 100
                
            # Formatage des valeurs
            if metric == 'time':
                adare_fmt = f"{adare_mean:.2f} ± {np.std(metrics['adare'][metric]):.2f}"
                nsga_fmt = f"{nsga_mean:.2f} ± {np.std(metrics['nsga3'][metric]):.2f}"
            else:
                adare_fmt = f"{adare_mean:.2e} ± {np.std(metrics['adare'][metric]):.2e}"
                nsga_fmt = f"{nsga_mean:.2e} ± {np.std(metrics['nsga3'][metric]):.2e}"
            
            results.append((
                metric,
                adare_fmt,
                nsga_fmt,
                f"{improvement:.1f}%"
            ))
        
        return results
    
    except KeyboardInterrupt:
        print("\nComparaison interrompue par l'utilisateur")
        return []
    except Exception as e:
        print(f"\nErreur durant la comparaison: {str(e)}")
        return []

# ==================================================================
# FONCTION D'AFFICHAGE DES METRIQUES [Makespan, Latence, Coût, Énergie]
# ==================================================================
def print_metrics_comparison(adare_metrics, nsga_metrics):
    print("\nComparaison des meilleures solutions ADARE vs NSGA-III:\n")
    print(f"{'Metric':<15} | {'ADARE':<20} | {'NSGA-III':<20} | Improvement")
    print("-"*75)
    
    metrics = {
        'Makespan': (adare_metrics['makespan'], nsga_metrics['makespan']),
        'Latence': (adare_metrics['latency'], nsga_metrics['latency']),
        'Coût': (adare_metrics['cost'], nsga_metrics['cost']),
        'Énergie': (adare_metrics['energy'], nsga_metrics['energy'])
    }
    
    for metric, (adare_val, nsga_val) in metrics.items():
        improvement = ((nsga_val - adare_val) / nsga_val) * 100
        print(f"{metric:<15} | {adare_val:<20.2f} | {nsga_val:<20.2f} | {improvement:>+.1f}%")

# ==================================================================
# FONCTION D'AFFICHAGE DE L'ORDONNANCEMENT DES TÂCHES
# ==================================================================
def visualize_schedule(individual, algorithm_name=""):
    # Compute schedule for the individual
    task_schedule = {}
    device_available_time = {node['id']: 0 for node in NODES}
    
    # Calculate metrics
    total_cost = 0
    total_energy = 0
    latencies = []
    
    # Tri topologique des tâches selon les dépendances
    sorted_tasks = []
    visited = set()
    
    def visit(task_id):
        if task_id in visited:
            return
        visited.add(task_id)
        for dep in TASKS[task_id]['dependencies']:
            # Ajuster l'indice pour correspondre à l'index de base 0
            dep_idx = dep - 1
            if dep_idx >= 0 and dep_idx < len(TASKS):  # Vérifier que l'indice est valide
                visit(dep_idx)
        sorted_tasks.append(task_id)
    
    for i in range(len(TASKS)):
        visit(i)
    
    # Ordonnancement des tâches
    for task_id in sorted_tasks:
        node_id = individual[task_id]
        node = NODES[node_id]
        task = TASKS[task_id]
        
        # Calcul du temps de début
        start_time = 0
        if task['dependencies']:
            # Vérifier que toutes les dépendances sont dans task_schedule
            valid_deps = [dep-1 for dep in task['dependencies'] if (dep-1) in task_schedule]
            if valid_deps:
                start_time = max(task_schedule[dep]['end'] for dep in valid_deps)
        
        # Utiliser le temps de disponibilité du nœud si nécessaire
        start_time = max(start_time, device_available_time[node_id])
        
        # Calcul des temps d'exécution et de transfert
        exec_time = task['instructions'] / node['processing_rate']
        transfer_time = task['data_size'] / node['uplink_bandwidth']
        end_time = start_time + exec_time + transfer_time
        
        # Mise à jour des disponibilités
        device_available_time[node_id] = end_time
        task_schedule[task_id] = {
            'task_id': task_id + 1,
            'node_id': node_id,
            'node_type': node['type'],
            'start': start_time,
            'end': end_time,
            'exec_time': exec_time
        }
        
        # Calcul des métriques
        if task['dependencies']:
            valid_deps = [dep-1 for dep in task['dependencies'] if (dep-1) in task_schedule]
            if valid_deps:
                ideal_start = max([task_schedule[dep]['exec_time'] for dep in valid_deps], default=0)
            else:
                ideal_start = 0
        else:
            ideal_start = 0
        
        latencies.append(end_time - ideal_start)
        total_cost += exec_time * node['processing_cost']
        total_energy += exec_time * node['working_power']
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(TASKS)))
    
    # Draw bars for each task
    for task_id, task_info in task_schedule.items():
        node_type = task_info['node_type']
        node_id = task_info['node_id']
        y_position = list(environments.keys()).index(node_type) * 5 + (node_id % environments[node_type]['devices'])
        
        ax.barh(y_position, 
                task_info['exec_time'], 
                left=task_info['start'], 
                color=colors[task_id], 
                edgecolor='black')
        
        text_x = task_info['start'] + task_info['exec_time'] / 2
        ax.text(text_x, y_position, f"T{task_info['task_id']}", 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Configure axes
    y_ticks = []
    y_labels = []
    for i, env_type in enumerate(environments.keys()):
        for j in range(environments[env_type]['devices']):
            y_ticks.append(i * 5 + j)
            y_labels.append(f"{env_type} {j+1}")
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_title(f'Task Schedule - {algorithm_name}')
    
    handles = [Patch(facecolor=colors[i], label=f"Task {i+1}") 
              for i in range(len(TASKS))]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()

    # Utiliser le chemin spécifique au workflow
    if len(sys.argv) > 1:
        workflow_type = sys.argv[1].split('_')[0]
        output_dir = f'output/{workflow_type}/plots'
    else:
        output_dir = 'output/Default/plots'
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/schedule_{algorithm_name}.png')
    plt.show()
    plt.close()
    
    # Calculate and print all metrics
    makespan = max(task_info['end'] for task_info in task_schedule.values())
    avg_latency = np.mean(latencies)
    
    return task_schedule

# ==================================================================
# FONCTION D'AFFICHAGE DES FRONTS DE PARETO
# ==================================================================
def plot_pareto_fronts(adare_population, nsga_population):
    # Déterminer le chemin de sortie
    if len(sys.argv) > 1:
        workflow_type = sys.argv[1].split('_')[0]
        output_dir = f'output/{workflow_type}/plots'
    else:
        output_dir = 'output/Default/plots'
    
    os.makedirs(output_dir, exist_ok=True)

    # Convert populations to metrics format
    adare_metrics = [{
        "Solution": i+1,
        "Makespan": ind.fitness.values[0],
        "Latence": ind.fitness.values[1],
        "Coût": ind.fitness.values[2],
        "Énergie": ind.fitness.values[3]
    } for i, ind in enumerate(adare_population)]
    
    nsga_metrics = [{
        "Solution": i+1,
        "Makespan": ind.fitness.values[0],
        "Latence": ind.fitness.values[1],
        "Coût": ind.fitness.values[2],
        "Énergie": ind.fitness.values[3]
    } for i, ind in enumerate(nsga_population)]

    # Matrix of 2D scatter plots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Cost vs Energy
    axs[0, 0].scatter([m["Coût"] for m in adare_metrics], 
                      [m["Énergie"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[0, 0].scatter([m["Coût"] for m in nsga_metrics], 
                      [m["Énergie"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[0, 0].set_xlabel('Coût')
    axs[0, 0].set_ylabel('Énergie')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Makespan vs Latency
    axs[0, 1].scatter([m["Makespan"] for m in adare_metrics], 
                      [m["Latence"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[0, 1].scatter([m["Makespan"] for m in nsga_metrics], 
                      [m["Latence"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[0, 1].set_xlabel('Makespan')
    axs[0, 1].set_ylabel('Latence')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Cost vs Makespan
    axs[0, 2].scatter([m["Coût"] for m in adare_metrics], 
                      [m["Makespan"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[0, 2].scatter([m["Coût"] for m in nsga_metrics], 
                      [m["Makespan"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[0, 2].set_xlabel('Coût')
    axs[0, 2].set_ylabel('Makespan')
    axs[0, 2].grid(True, linestyle='--', alpha=0.7)
    
    # Energy vs Makespan
    axs[1, 0].scatter([m["Énergie"] for m in adare_metrics], 
                      [m["Makespan"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[1, 0].scatter([m["Énergie"] for m in nsga_metrics], 
                      [m["Makespan"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[1, 0].set_xlabel('Énergie')
    axs[1, 0].set_ylabel('Makespan')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Energy vs Latency
    axs[1, 1].scatter([m["Énergie"] for m in adare_metrics], 
                      [m["Latence"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[1, 1].scatter([m["Énergie"] for m in nsga_metrics], 
                      [m["Latence"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[1, 1].set_xlabel('Énergie')
    axs[1, 1].set_ylabel('Latence')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Cost vs Latency
    axs[1, 2].scatter([m["Coût"] for m in adare_metrics], 
                      [m["Latence"] for m in adare_metrics],
                      c='blue', label='ADARE', alpha=0.7)
    axs[1, 2].scatter([m["Coût"] for m in nsga_metrics], 
                      [m["Latence"] for m in nsga_metrics],
                      c='red', label='NSGA-III', alpha=0.7)
    axs[1, 2].set_xlabel('Coût')
    axs[1, 2].set_ylabel('Latence')
    axs[1, 2].grid(True, linestyle='--', alpha=0.7)

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    
    plt.suptitle('Comparaison des fronts de Pareto pour tous les objectifs', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f'{output_dir}/pareto_2d.png')
    plt.show()
    plt.close()

    # 3D plots
        # Combined 3D Pareto fronts
    fig = plt.figure(figsize=(15, 10))

    # Create a grid 2x3 of 3D subplots
    axs = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(2, 3, i*3+j+1, projection='3d')
            axs.append(ax)

    # Cost vs Energy vs Makespan
    axs[0].scatter([m["Coût"] for m in adare_metrics],
                   [m["Énergie"] for m in adare_metrics],
                   [m["Makespan"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[0].scatter([m["Coût"] for m in nsga_metrics],
                   [m["Énergie"] for m in nsga_metrics],
                   [m["Makespan"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[0].set_xlabel('Coût')
    axs[0].set_ylabel('Énergie')
    axs[0].set_zlabel('Makespan')
    axs[0].set_title('Coût vs Énergie vs Makespan')

    # Energy vs Makespan vs Latency
    axs[1].scatter([m["Énergie"] for m in adare_metrics],
                   [m["Makespan"] for m in adare_metrics],
                   [m["Latence"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[1].scatter([m["Énergie"] for m in nsga_metrics],
                   [m["Makespan"] for m in nsga_metrics],
                   [m["Latence"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[1].set_xlabel('Énergie')
    axs[1].set_ylabel('Makespan')
    axs[1].set_zlabel('Latence')
    axs[1].set_title('Énergie vs Makespan vs Latence')

    # Cost vs Latency vs Makespan
    axs[2].scatter([m["Coût"] for m in adare_metrics],
                   [m["Latence"] for m in adare_metrics],
                   [m["Makespan"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[2].scatter([m["Coût"] for m in nsga_metrics],
                   [m["Latence"] for m in nsga_metrics],
                   [m["Makespan"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[2].set_xlabel('Coût')
    axs[2].set_ylabel('Latence')
    axs[2].set_zlabel('Makespan')
    axs[2].set_title('Coût vs Latence vs Makespan')

    # Cost vs Energy vs Latency
    axs[3].scatter([m["Coût"] for m in adare_metrics],
                   [m["Énergie"] for m in adare_metrics],
                   [m["Latence"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[3].scatter([m["Coût"] for m in nsga_metrics],
                   [m["Énergie"] for m in nsga_metrics],
                   [m["Latence"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[3].set_xlabel('Coût')
    axs[3].set_ylabel('Énergie')
    axs[3].set_zlabel('Latence')
    axs[3].set_title('Coût vs Énergie vs Latence')

    # Energy vs Makespan vs Latency (repeated)
    axs[4].scatter([m["Énergie"] for m in adare_metrics],
                   [m["Makespan"] for m in adare_metrics],
                   [m["Latence"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[4].scatter([m["Énergie"] for m in nsga_metrics],
                   [m["Makespan"] for m in nsga_metrics],
                   [m["Latence"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[4].set_xlabel('Énergie')
    axs[4].set_ylabel('Makespan')
    axs[4].set_zlabel('Latence')
    axs[4].set_title('Énergie vs Makespan vs Latence')

    # Cost vs Latency vs Energy
    axs[5].scatter([m["Coût"] for m in adare_metrics],
                   [m["Latence"] for m in adare_metrics],
                   [m["Énergie"] for m in adare_metrics],
                   c='blue', label='ADARE', alpha=0.7)
    axs[5].scatter([m["Coût"] for m in nsga_metrics],
                   [m["Latence"] for m in nsga_metrics],
                   [m["Énergie"] for m in nsga_metrics],
                   c='red', label='NSGA-III', alpha=0.7)
    axs[5].set_xlabel('Coût')
    axs[5].set_ylabel('Latence')
    axs[5].set_zlabel('Énergie')
    axs[5].set_title('Coût vs Latence vs Énergie')

    # Add common legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)

    plt.tight_layout()

    plt.savefig(f'{output_dir}/pareto_3d.png')
    plt.show()
    plt.close()
# ==================================================================
# FONCTION D'AFFICHAGE DE LA CONVERGENCE
# ==================================================================
def plot_convergence(adare_history, nsga_history):
    # Déterminer le chemin de sortie
    if len(sys.argv) > 1:
        workflow_type = sys.argv[1].split('_')[0]
        output_dir = f'output/{workflow_type}/plots'
    else:
        output_dir = 'output/Default/plots'
    
    os.makedirs(output_dir, exist_ok=True)

    # Vérifier et ajuster les dimensions
    min_len = min(len(adare_history[0]), len(nsga_history[0])) 
    generations = range(min_len)
    
    metrics = ['Makespan', 'Latence', 'Coût', 'Énergie']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        # Tronquer les historiques à la même longueur
        adare_vals = adare_history[i][:min_len] 
        nsga_vals = nsga_history[i][:min_len]
        
        axes[i].plot(generations, adare_vals, 'b-', label='ADARE')
        axes[i].plot(generations, nsga_vals, 'r-', label='NSGA-III')
        axes[i].set_xlabel('Génération')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Convergence - {metric}')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence.png')
    plt.show()
    plt.close()

# ==================================================================
# ANALYSE WITH AI.I.
## ==================================================================
def analyze_results_with_ai(comparison_results, adare_metrics, nsga_metrics):
    # Format the metrics into a prompt
    prompt = f"""
    Analyse comparative des algorithmes ADARE et NSGA-III:

    Please analyze these algorithm comparison results and provide insights in French.
    Rassure-toi de comparer la robuustesse et l'efficacité sachant qu'ici, il existe
    The whole analyze may be provide with cli model display.
    Fais ensuite une moyenne des métriques directes pour chaque algorithme et les afficher, et déduire qui a mieux minimiser les métriques directes (MakeSpan, Latence, Coût, Énergie).
    Fais ensuite une moyenne des métriques statistiques obtenues pour chaque algorithme et les afficher, et déduire qui a mieux minimiser ces métriques statistiques.

    Métriques directes:
    - Makespan: ADARE {adare_metrics['makespan']:.2f} vs NSGA-III {nsga_metrics['makespan']:.2f}
    - Latence: ADARE {adare_metrics['latency']:.2f} vs NSGA-III {nsga_metrics['latency']:.2f} 
    - Coût: ADARE {adare_metrics['cost']:.2f} vs NSGA-III {nsga_metrics['cost']:.2f}
    - Énergie: ADARE {adare_metrics['energy']:.2f} vs NSGA-III {nsga_metrics['energy']:.2f}
    
    Métriques statistiques (moyenne sur {len(comparison_results)} exécutions):

    Explique aussi brièvement chaque métrique statisitque et leur importance dans le contexte de l'optimisation.
    """
    
    for res in comparison_results:
        metric, adare_val, nsga_val, improvement = res
        prompt += f"\n- {metric}: ADARE {adare_val} vs NSGA-III {nsga_val} (Amélioration: {improvement})"
    
    # Get AI response using g4f
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        print("\nAnalyse IA des résultats:\n")
        print(response)

        return response
    except Exception as e:
        print(f"Erreur lors de l'analyse IA: {str(e)}")
        return "Analyse IA non disponible"
    
# ==================================================================
# FONCTION DE REDACTION DES RAPPORTS DE L'EXPERIMENTATION
# ==================================================================
def save_report(comparison_results, adare_metrics, nsga_metrics, ai_analysis):
    # Déterminer le chemin de sortie
    if len(sys.argv) > 1:
        workflow_type = sys.argv[1].split('_')[0]
        output_dir = f'output/{workflow_type}/reports'
    else:
        output_dir = 'output/Default/reports'
    
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/comparison_report.txt', 'w') as f:
        f.write("ADARE vs NSGA-III Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Metrics comparison
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        metrics = {
            'Makespan': (adare_metrics['makespan'], nsga_metrics['makespan']),
            'Latence': (adare_metrics['latency'], nsga_metrics['latency']),
            'Coût': (adare_metrics['cost'], nsga_metrics['cost']),
            'Énergie': (adare_metrics['energy'], nsga_metrics['energy'])
        }
        
        for metric, (adare_val, nsga_val) in metrics.items():
            improvement = ((nsga_val - adare_val) / nsga_val) * 100
            f.write(f"{metric}:\n")
            f.write(f"  ADARE: {adare_val:.2f}\n")
            f.write(f"  NSGA-III: {nsga_val:.2f}\n")
            f.write(f"  Improvement: {improvement:>+.1f}%\n\n")
        
        # Statistical comparison
        f.write("\nStatistical Comparison:\n")
        f.write("-" * 20 + "\n")
        for metric, adare_val, nsga_val, improvement in comparison_results:
            f.write(f"{metric}:\n")
            f.write(f"  ADARE: {adare_val}\n")
            f.write(f"  NSGA-III: {nsga_val}\n")
            f.write(f"  {improvement}\n\n")
        
        # AI Analysis
        f.write("\nAI Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(ai_analysis)

# ==================================================================
# EXÉCUTION PRINCIPALE
# ==================================================================
if __name__ == "__main__":
    # Déterminer le type de workflow et configurer les chemins de sortie
    workflow_type = "Default"
    if len(sys.argv) > 1:
        benchmark_name = sys.argv[1]
        workflow_type, workflow_size = benchmark_name.split('_')
        tasks_file = f'data/benchmarks/{workflow_type}/{benchmark_name}.json'
        
        # Vérifier si le fichier existe
        if not os.path.exists(tasks_file):
            print(f"Erreur: Le fichier {tasks_file} n'a pas été trouvé.")
            sys.exit(1)
        
        print(f"Utilisation des données de tâches depuis: {tasks_file}")
    
    # Initialisation des dossiers de sortie spécifiques au workflow
    # output_plots_dir = f'output/{workflow_type}/plots'
    # output_reports_dir = f'output/{workflow_type}/reports'
    
    # os.makedirs(output_plots_dir, exist_ok=True)
    # os.makedirs(output_reports_dir, exist_ok=True)
    
    # Configuration initiale
    initial_config = {
        'reference_points_divisions': NICHE_COUNT,
        'population_size': POP_SIZE,
        'crossover_probability': CXPB,
        'mutation_probability': MUTPB
    }

    # Optimisation des hyperparamètres
    print("\n=== Optimisation des hyperparamètres ===")
    optimized_config = adaptive_hyperparameter_tuning(initial_config)
    
    # Configuration des opérateurs
    toolbox.register("mate", adare_mate)
    toolbox.register("mutate", adaptive_mutation)
    toolbox.register("select", tools.selNSGA3, 
                    ref_points=tools.uniform_reference_points(len(OBJ_NAMES), NICHE_COUNT))

    # Exécution des algorithmes
    print("\n=== Exécution des algorithmes ===")
    
    # ADARE
    print("Exécution d'ADARE...")
    adare_population, adare_history, adare_time = run_adare()
    print(f"ADARE terminé en {adare_time:.2f} secondes")
    
    # NSGA-III
    print("\nExécution de NSGA-III...")
    nsga_population, nsga_history, nsga_time = run_nsga3()
    print(f"NSGA-III terminé en {nsga_time:.2f} secondes")

    # Visualisation
    print("\n=== Génération des visualisations ===")
    
    # Courbes de convergence
    plot_convergence(adare_history, nsga_history)
    
    # Fronts de Pareto
    plot_pareto_fronts(adare_population, nsga_population)

    # Meilleures solutions
    best_adare = tools.selBest(adare_population, 1)[0]
    best_nsga = tools.selBest(nsga_population, 1)[0]
    
    # Ordonnancement
    print("\n=== Analyse des solutions ===")
    adare_schedule = visualize_schedule(best_adare, "ADARE")
    nsga_schedule = visualize_schedule(best_nsga, "NSGA-III")
    
    # Métriques
    adare_metrics = {
        'makespan': max(t['end'] for t in adare_schedule.values()),
        'latency': np.mean([t['end'] - t['start'] for t in adare_schedule.values()]),
        'cost': best_adare.fitness.values[2],
        'energy': best_adare.fitness.values[3]
    }
    
    nsga_metrics = {
        'makespan': max(t['end'] for t in nsga_schedule.values()),
        'latency': np.mean([t['end'] - t['start'] for t in nsga_schedule.values()]),
        'cost': best_nsga.fitness.values[2],
        'energy': best_nsga.fitness.values[3]
    }
    
    # Affichage des résultats
    print_metrics_comparison(adare_metrics, nsga_metrics)

    # Comparaison statistique
    print("\n=== Comparaison statistique ===")
    comparison_results = run_comparison(runs=10)
    
    if comparison_results:
        print("\nRésultats statistiques:")
        print(f"{'Metric':<15} | {'ADARE':<20} | {'NSGA-III':<20} | Improvement")
        print("-"*75)
        for metric, adare_val, nsga_val, improvement in comparison_results:
            print(f"{metric:<15} | {adare_val:<20} | {nsga_val:<20} | {improvement}")
        
        # Analyse IA
        ai_analysis = analyze_results_with_ai(comparison_results, adare_metrics, nsga_metrics)
        
        # Sauvegarde du rapport
        save_report(comparison_results, adare_metrics, nsga_metrics, ai_analysis)
        # print(f"\nRapport sauvegardé dans {output_reports_dir}/comparison_report.txt")

        if len(sys.argv) > 1:
            workflow_type = sys.argv[1].split('_')[0]
            output_dir = f'output/{workflow_type}/reports'
        else:
            output_dir = 'output/Default/reports'
            print(f"\nRapport sauvegardé dans {output_dir}/comparison_report.txt")
    
    print("\n=== Exécution terminée ===")