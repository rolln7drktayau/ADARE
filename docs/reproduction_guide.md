# Guide de reproduction ADARE

Ce guide décrit l'utilisation publique de l'artefact de recherche ADARE. Le dépôt de référence est :

```text
https://github.com/rolln7drktayau/ADARE
```

La révision SwEvo 2026 est développée sur la branche :

```text
review/swevo-major-revision-2026
```

## 1. Installation

Prérequis : Git et Python 3.11 ou plus récent. La campagne publiée a utilisé Python 3.13.1.

```bash
git clone https://github.com/rolln7drktayau/ADARE.git
cd ADARE
git checkout review/swevo-major-revision-2026
python -m venv .venv
```

Sous PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous Linux/macOS :

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Vérification rapide

ADARE seul :

```bash
python scripts/run_adare.py --benchmarks Montage_25 --runs 1 --generations 5 --population-size 20 --output-dir output/smoke_adare
```

Comparaison appariée courte :

```bash
python scripts/run_extended_comparison.py --benchmarks Montage_25 --algorithms ADARE NSGA-III QL-NSGA-III --runs 2 --generations 5 --population-size 20 --output-dir output/smoke_comparison
```

Ces commandes vérifient le chargement du benchmark, l'évaluateur, les algorithmes et l'écriture des rapports. Elles ne reproduisent pas les conclusions statistiques du papier.

## 3. Protocole complet de la major revision

Sous Windows, afficher d'abord le protocole sans l'exécuter :

```powershell
.\run_swevo_major_revision.ps1 -Preset full
```

Puis lancer une étape précise :

```powershell
.\run_swevo_major_revision.ps1 -Preset full -Only 20_small_all_algorithms_r20_g70
```

Les identifiants, temps attendus et règles de reprise sont détaillés dans [swevo_execution_tutorial_fr.md](swevo_execution_tutorial_fr.md). Les sorties sont placées dans `output/major_revision/` et ne sont pas versionnées.

## 4. Budgets confirmatoires

| Protocole | Répétitions | Population | Générations |
|---|---:|---:|---:|
| Petits workflows | 20 | 100 | 70 |
| 1000 tâches, budget intermédiaire | 20 | 80 | 50 |
| 1000 tâches, pression longue | 20 | 80 | 100 |
| Classe 3000 tâches | 10 | 60 | 20 |
| Diagnostic ressources/scalabilité | 10 | 60 | 30 |

Les graines et populations initiales sont appariées. Le cache peut produire des nombres d'évaluations objectives uniques différents malgré des budgets population/générations identiques ; le fichier `evaluation_budget.csv` documente cette différence.

## 5. Régénérer les résultats synthétiques

Après les expériences :

```bash
python scripts/major_revision_report.py --root output/major_revision
python scripts/build_revision_assets.py
```

Les résumés destinés au contrôle et à la réutilisation sont versionnés dans `results/major_revision/`. Les tableaux LaTeX sont dans `papers/generated/` et les figures finales dans `Figures/`.

## 6. Compiler l'article

Le PDF versionné est `papers/article_swevo.pdf`. Avec MiKTeX/TeXLive :

```bash
cd papers
pdflatex -interaction=nonstopmode -halt-on-error article_swevo.tex
pdflatex -interaction=nonstopmode -halt-on-error article_swevo.tex
```

Sous Windows, `build_swevo_revision.ps1` compile aussi la réponse et crée localement un paquet dans `submission/`, dossier ignoré par Git.

## 7. Réutiliser l'évaluateur ou ajouter un algorithme

Un nouvel optimiseur doit utiliser :

- les mêmes objets workflow/ressources chargés par `problem/` ;
- la fonction d'évaluation commune de `BaseAlgorithm` ;
- une population initiale fournie par le runner apparié ;
- `survival_population` pour les comparaisons finales ;
- les mêmes graines et budgets que ses comparateurs.

Il faut rapporter les indicateurs séparément, les évaluations réellement exécutées, le temps, et une inférence appariée. Une moyenne de pourcentages entre HV, IGD, spacing, epsilon et coverage ne doit pas servir de conclusion principale.

## 8. Portée scientifique

Le modèle est déterministe et utilise des unités de simulateur. L'énergie est un proxy compute-only. Les pannes, la contention, l'énergie de communication/idle et les coûts de transfert sont hors périmètre. ADARE est une extension adaptative de NSGA-III et non un nouvel opérateur de survie. Les adaptations OVEA-style et QMOEA/D-AWA-style ne sont pas des reproductions officielles de leurs auteurs.
