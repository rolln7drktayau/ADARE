import json
import re
import argparse
import os

def parse_job_id(job_id):
    """Convertit 'ID00000' en 0"""
    return int(re.search(r'\d+', job_id).group())

def convert_workflow(input_json_path, output_json_path):
    """Convertit un workflow JSON Montage en format de tâches avec IDs normalisés commençant à 1"""
    # Paramètres de conversion
    INSTRUCTION_FACTOR = 75000  # Convertir le runtime en "instructions"
    DATA_SIZE_FACTOR = 1e6      # Convertir les octets en MB
    
    # Charger le workflow Montage
    with open(input_json_path) as f:
        montage = json.load(f)
    
    # Créer un mapping des dépendances et obtenir les IDs originaux
    original_ids = [parse_job_id(job['id']) for job in montage['jobs']]
    min_id = min(original_ids)
    
    # Créer un mapping entre les anciens IDs et les nouveaux IDs (commençant à 1)
    id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(original_ids))}
    
    # Créer un mapping des dépendances avec les nouveaux IDs
    dependencies_map = {id_mapping[parse_job_id(job['id'])]: [] for job in montage['jobs']}
    for dep in montage['dependencies']:
        to_id = parse_job_id(dep['to'])
        from_id = parse_job_id(dep['from'])
        dependencies_map[id_mapping[to_id]].append(id_mapping[from_id])
    
    # Convertir les jobs avec les nouveaux IDs
    tasks = []
    for job in montage['jobs']:
        original_job_id = parse_job_id(job['id'])
        new_job_id = id_mapping[original_job_id]
        
        # Calculer la taille totale des données d'entrée
        data_size = sum([f['size'] for f in job['inputs']]) / DATA_SIZE_FACTOR
        
        tasks.append({
            'id': new_job_id,
            'instructions': int(job['runtime'] * INSTRUCTION_FACTOR),
            'data_size': round(data_size, 1),
            'dependencies': dependencies_map.get(new_job_id, []),
            'deadline': int(job['runtime'] * 2)  # Deadline = 2x runtime
        })
    
    # Trier par ID de tâche
    tasks = sorted(tasks, key=lambda x: x['id'])
    
    # Sauvegarder le résultat
    output = {'tasks': tasks}
    with open(output_json_path, 'w') as f:
        json.dump(output, f, indent=4)  # Utiliser indent=4 comme dans tasks.json
    
    print(f"Conversion terminée! Fichier sauvegardé: {output_json_path}")
    print(f"Les IDs des tâches ont été normalisés pour commencer à 1.")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Convertir un workflow Montage JSON en format de tâches avec IDs normalisés.')
    parser.add_argument('input_json', help='Chemin vers le fichier JSON d\'entrée')
    parser.add_argument('-o', '--output', help='Chemin vers le fichier JSON de sortie (optionnel)')
    
    args = parser.parse_args()
    
    # Déterminer le chemin de sortie
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_json)[0]
        output_path = f"{base_name}_tasks.json"
    
    # Effectuer la conversion
    convert_workflow(args.input_json, output_path)