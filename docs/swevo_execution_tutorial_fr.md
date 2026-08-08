# ADARE — tutoriel d'exécution de la major revision SwEvo

Date limite indiquée dans la lettre : **26 août 2026**.

Ce protocole doit être lancé depuis PowerShell, à la racine du projet :

```powershell
Set-Location "C:\Users\Rolan\Desktop\Project\ADARE"
```

Ne pas lancer immédiatement tout le preset `full`. Les étapes longues doivent être exécutées séparément afin de contrôler chaque résultat avant d'engager plusieurs heures de calcul.

## 1. Préparer l'environnement

```powershell
python --version
python -m pip install -r requirements.txt
```

Temps indicatif : 1 à 10 minutes selon les dépendances déjà installées.

Puis générer le plan sans lancer d'expérience :

```powershell
.\run_swevo_major_revision.ps1 -Preset quick -PlanOnly
.\run_swevo_major_revision.ps1 -Preset full -PlanOnly
```

Temps : moins d'une minute. Ces commandes créent les manifestes, protocoles Markdown et lanceurs sous `output/major_revision/`.

## 2. Validation rapide obligatoire

```powershell
.\run_swevo_major_revision.ps1 -Preset quick
```

Temps indicatif : 5 à 15 minutes. Cette étape vérifie la comparaison multi-algorithmes, l'ablation progressive, la trace du contrôleur et l'ablation contrôlée.

Me notifier dès la fin de cette étape, ou immédiatement si une erreur apparaît. Fournir :

```text
output/major_revision/reports/major_revision_summary.md
output/major_revision/reports/major_revision_statistics.csv
output/major_revision/reports/runtime_breakdown.csv
output/major_revision/reports/reward_survival_correlation.csv
output/major_revision/reports/evaluation_budget.csv
output/major_revision/logs/
```

Ne pas commencer les runs longs avant mon contrôle de ces fichiers.

## 3. Expériences petites et moyennes

Lancer chaque commande séparément.

### 3.1 Ablation progressive des modules

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 10_ablation_v1_v5_r20
```

Temps indicatif : 10 à 25 minutes.

### 3.2 Ablation contrôlée du contrôleur

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 11_controller_ablation_r20_g70
```

Temps indicatif : 1 à 3 heures. Cette expérience compare sélection statique, UCB global, UCB contextuel avec reward simple, puis reward ADARE, tout le reste étant fixé.

### 3.3 Sensibilité reward et apprentissage

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 12_reward_sensitivity_r20_g70
```

Temps indicatif après optimisation Pareto : 45 à 90 minutes. Elle teste les poids, un clipping plus serré, l'absence de clipping et le taux d'apprentissage.

### 3.4 Comparaison complète sur petits workflows

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 20_small_all_algorithms_r20_g70
```

Temps indicatif après optimisation Pareto : 40 à 90 minutes.

Après les quatre étapes, reconstruire les rapports :

```powershell
.\run_swevo_major_revision.ps1 -Preset reports
```

Me notifier à ce point avec le dossier `output/major_revision/reports/`. J'examinerai les résultats avant les calculs 1000/3000 tâches.

## 4. Expériences longues

### 4.1 Workflows 1000 tâches, 50 générations

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 30_1000_budget_sweep_r20_g50
```

Temps indicatif : 1 h 30 à 4 h.

Me notifier à la fin. Si les résultats ou logs sont anormaux, ne pas lancer la suite.

### 4.2 Workflows 1000 tâches, 100 générations

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 31_1000_budget_sweep_r20_g100
```

Temps indicatif : 3 à 8 heures ; lancement de nuit conseillé.

Me notifier à la fin avant le protocole 3000 tâches.

### 4.3 Workflows 3000 tâches

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 40_3000_budget_sweep_r10_g20
```

Temps indicatif : 2 à 6 heures.

### 4.4 Traces détaillées du contrôleur

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 50_adare_controller_traces_1000_r20_g100
```

Temps indicatif : 45 minutes à 2 heures.

Cette étape produit aussi la décomposition du temps ADARE et les variables nécessaires pour corréler reward immédiat et survie des descendants.

### 4.5 Diagnostic final : évaluations, temps, mémoire et ressources

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 60_scaling_resource_diagnostics_r10_g30
```

Temps indicatif : 30 à 90 minutes. Ce protocole ciblé ne remplace pas les campagnes principales. Il compare ADARE,
NSGA-III et QL-NSGA-III sur Montage à 25, 1000 et 3000 tâches, puis sur trois configurations de ressources à
1000 tâches. Il enregistre la convergence selon le nombre réel d'évaluations et le temps écoulé, ainsi que le pic RSS
échantillonné de l'arbre de processus. Me notifier à la fin avant toute modification des tables du manuscrit.

## 5. Rapport final des calculs

```powershell
.\run_swevo_major_revision.ps1 -Preset reports
```

Temps : moins d'une minute.

Me notifier avec les fichiers suivants :

```text
output/major_revision/reports/major_revision_summary.md
output/major_revision/reports/major_revision_statistics.csv
output/major_revision/reports/controller_behavior.csv
output/major_revision/reports/runtime_breakdown.csv
output/major_revision/reports/reward_survival_correlation.csv
output/major_revision/reports/evaluation_budget.csv
output/major_revision/reports/reviewer_response_matrix.md
output/major_revision/logs/
```

## 6. En cas d'interruption ou d'erreur

Ne pas supprimer les sorties existantes. Noter l'identifiant de l'étape et envoyer le fichier correspondant sous `output/major_revision/logs/`.

Pour relancer une seule étape :

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only IDENTIFIANT_ETAPE
```

Le script réécrit actuellement les CSV de l'étape sélectionnée ; conserver une copie du dossier de l'étape si un run partiel doit être diagnostiqué avant relance.

## 7. Étape finale : compiler et préparer le dépôt Elsevier

Les expériences, leur interprétation, la réécriture de l'article et la réponse point par point sont terminées. Le modèle déterministe, ses limites physiques, la sensibilité des ressources, la filiation NSGA-III et les adaptations de baselines sont maintenant explicités.

Cette dernière étape ne relance aucune expérience. Elle régénère les rapports, compile l'article et la réponse, puis crée un ZIP distinct de l'ancienne soumission.

1. Vérifier `pdflatex --version`. Si la commande n'existe pas, installer MiKTeX et autoriser l'installation automatique des paquets manquants.
2. Fermer puis rouvrir PowerShell à la racine du projet.
3. Lancer :

```powershell
.\build_swevo_revision.ps1
```

Temps attendu : 1 à 3 minutes, hors installation de MiKTeX. Me notifier à la première erreur LaTeX, ou lorsque la console affiche `Revision package ready`. Les fichiers finaux seront sous `submission\swevo_revision_2026` et dans `submission\swevo_revision_2026.zip`.

Les durées indiquées sont des fourchettes. Elles sont calibrées à partir des CSV déjà présents dans le dépôt (environ 3,2 s par run-algorithme sur petits workflows, 15 s sur les anciens runs 1000 tâches peu profonds et 19 s sur les anciens runs 3000 tâches peu profonds), puis élargies pour les budgets plus profonds et la variabilité de la machine.
