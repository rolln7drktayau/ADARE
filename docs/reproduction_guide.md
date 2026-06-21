# ADARE - Guide de reproduction pour Madame Sonia YASSA

Ce document accompagne la version revisee du projet ADARE apres les retours ECML PKDD.

## 1. Depot GitHub

Depot :

```text
https://github.com/rolln7drktayau/ADARE
```

Branche de travail a verifier :

```text
master
```

## 2. Recuperer le projet

```bash
git clone https://github.com/rolln7drktayau/ADARE.git
cd ADARE
git checkout master
```

## 3. Prerequis

- Python 3.10+ recommande.
- Git.
- Make.
- LaTeX/MiKTeX seulement si vous voulez recompiler le PDF de l'article.

Sur Windows, si `make` n'est pas disponible, l'installer via Git Bash, MSYS2, Chocolatey, ou utiliser directement les commandes Python indiquees dans le README.

## 4. Demarrage recommande

La commande principale prepare l'environnement puis ouvre un menu interactif :

```bash
make
```

Cette commande fait :

1. installation/mise a jour des dependances via `requirements.txt`,
2. creation des dossiers de sortie,
3. verification syntaxique des scripts principaux,
4. ouverture du menu ADARE avec les temps estimes.

Si l'environnement est deja pret :

```bash
make menu
```

Pour voir toutes les commandes :

```bash
make help
```

## 5. Commandes utiles

### Tester rapidement que tout fonctionne

```bash
make smoke
```

But : lancer un petit test comparatif ADARE vs NSGA-III.

### Lancer ADARE seul

```bash
make adare
```

But : executer uniquement ADARE sur `Montage_25`, sans baseline.

Pour un workflow plus grand :

```bash
make adare-1000
```

But : executer ADARE seul sur `CyberShake_1000`.

### Voir l'evolution de l'algorithme

```bash
make live
```

Cette commande lance une vue d'evolution avec :

- generation courante,
- courbes de convergence,
- evolution des metriques HV, IGD, spacing, epsilon, coverage-to-reference,
- projection du front Pareto.

Les sorties sont sauvegardees dans :

```text
output/live_view/
```

Note : cette vue ajoute un petit surcout car elle capture les populations par generation et calcule les metriques pour l'affichage. Les runs normaux ne sont pas ralentis.

### Relancer le protocole principal du papier

```bash
make main20
```

But : relancer le protocole principal en 20 repetitions sur les petits workflows du papier.

### Relancer les gros protocoles 1000/3000

```bash
make extended-1000-r20
make extended-3000-r20
```

Attention : ces commandes peuvent durer plusieurs heures selon la machine.

## 6. Ou trouver les resultats

Les sorties nouvellement generees vont principalement dans :

```text
output/
```

Exemples :

- `output/<Workflow>/reports/` : rapports CSV/TXT par workflow.
- `output/<Workflow>/plots/` : figures de convergence et projections Pareto.
- `output/adare_only/` : resultats ADARE seul.
- `output/live_view/` : dashboard d'evolution et CSV associe.
- `output/extended_1000_r20/` : runs longs 1000 taches.
- `output/extended_3000_r20/` : runs longs 3000 taches.

Les resultats consolides deja versionnes sont dans :

```text
results/extended/
```

Fichiers importants :

```text
results/extended/extended_global_summary.csv
results/extended/extended_1000_r20_summary.csv
results/extended/extended_3000_r20_summary.csv
```

Les figures de l'article sont dans :

```text
Figures/
```

Les PDFs de l'article sont :

```text
papers/article_ecml.pdf
ADARE_Adaptive_Data-driven_Algorithm_for_Resource_Evolution.pdf
```

## 7. Structure du projet

```text
algorithms/        implementations ADARE, NSGA-III, NSGA-II, MOEA/D, QL-NSGA-III, OVEA-style, QMOEA/D-AWA-style
config/            parametres experimentaux et algorithmiques
data/benchmarks/   workflows disponibles
evaluation/        metriques et visualisation
scripts/           scripts executables
results/extended/  resultats consolides versionnes
output/            resultats generes localement
docs/              notes et audits de revision
```

## 8. Ce qui a ete modifie apres les reviews

Principales revisions :

- ADARE est maintenant presente comme une approche autonome de controle adaptatif multi-objectif, pas seulement comme une extension optimisee de NSGA-III.
- Ajout de runs sur workflows 1000 et 3000 taches.
- Ajout de comparaisons avec des baselines adaptatives/recentes : QL-NSGA-III, OVEA-style, QMOEA/D-AWA-style.
- Ajout de protocoles longs en 20 repetitions sur 1000/3000.
- Ajout d'un runner ADARE seul.
- Ajout d'une vue d'evolution.
- Reorganisation des scripts dans `scripts/`.
- Ajout d'un menu Make reproductible.
- Mise a jour du README, du rapid-startup, des figures et des PDFs.

## 9. Points a discuter pour IEEE TEVC

IEEE TEVC semble une cible ambitieuse mais coherente, car la version revisee met davantage l'accent sur :

- optimisation evolutionnaire multi-objectif,
- controle adaptatif des operateurs,
- evaluation comparative contre des familles MOEA classiques et adaptatives,
- scalabilite sur workflows 1000/3000.

Points à valider avant soumission journal :

- niveau de detail theorique attendu par TEVC,
- longueur et format journal,
- besoin eventuel d'ajouter plus de workflows/domaines,
- opportunite d'ajouter des implementations exactes publiques de certains baselines si disponibles.
