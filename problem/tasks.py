from __future__ import annotations

"""Task loading and DAG topological ordering utilities."""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _repo_root() -> Path:
    """Return repository root path from this module location."""
    return Path(__file__).resolve().parents[1]


def _as_non_negative_float(value: Any, field: str, task_id: int) -> float:
    """Validate and cast a non-negative float for task attributes."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valeur invalide pour '{field}' dans la tache {task_id}.") from exc
    if parsed < 0.0:
        raise ValueError(f"'{field}' doit etre >= 0 dans la tache {task_id}.")
    return parsed


def resolve_benchmark_path(benchmark: str | None) -> Path:
    """Resolve benchmark alias into an absolute JSON benchmark path."""
    if benchmark is None:
        return _repo_root() / "data" / "benchmarks" / "Samples" / "tasks.json"

    if benchmark.endswith(".json"):
        candidate = Path(benchmark)
        if not candidate.is_absolute():
            candidate = _repo_root() / candidate
        return candidate

    if "_" not in benchmark:
        raise ValueError(
            f"Benchmark '{benchmark}' invalide. Format attendu: <Workflow>_<Taille> (ex: Montage_25)."
        )

    workflow_name = benchmark.split("_", 1)[0]
    return _repo_root() / "data" / "benchmarks" / workflow_name / f"{benchmark}.json"


def load_tasks(benchmark: str | None = None) -> List[Dict[str, Any]]:
    """Load and normalize task definitions from a benchmark file."""
    path = resolve_benchmark_path(benchmark)
    if not path.exists():
        raise FileNotFoundError(f"Fichier benchmark introuvable: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Format benchmark invalide dans {path}.")
    tasks = payload.get("tasks", [])
    if not tasks:
        raise ValueError(f"Aucune tache trouvee dans {path}")
    if not isinstance(tasks, list):
        raise ValueError(f"Le champ 'tasks' doit etre une liste dans {path}.")

    normalized: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Tache invalide (index {index}) dans {path}.")
        task_id = int(task.get("id", index + 1))
        instructions = _as_non_negative_float(task.get("instructions"), "instructions", task_id)
        data_size = _as_non_negative_float(task.get("data_size", 0.0), "data_size", task_id)
        deadline = _as_non_negative_float(task.get("deadline", 0.0), "deadline", task_id)
        dependencies_raw = task.get("dependencies", [])
        if dependencies_raw is None:
            dependencies_raw = []
        if not isinstance(dependencies_raw, list):
            raise ValueError(f"'dependencies' doit etre une liste dans la tache {task_id}.")
        dependencies = [int(dep) for dep in dependencies_raw]

        normalized.append(
            {
                "id": task_id,
                "instructions": instructions,
                "data_size": data_size,
                "dependencies": dependencies,
                "deadline": deadline,
            }
        )
    return normalized


def topological_sort(tasks: Sequence[Dict[str, Any]]) -> List[int]:
    """Return DAG node indices ordered topologically."""
    n_tasks = len(tasks)
    indegree = [0] * n_tasks
    successors: List[List[int]] = [[] for _ in range(n_tasks)]

    for idx, task in enumerate(tasks):
        for dep_id in task.get("dependencies", []):
            dep_idx = int(dep_id) - 1
            if dep_idx < 0 or dep_idx >= n_tasks:
                raise ValueError(f"Dependance invalide pour la tache {idx + 1}: {dep_id}")
            successors[dep_idx].append(idx)
            indegree[idx] += 1

    queue = deque(i for i, d in enumerate(indegree) if d == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for succ in successors[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(order) != n_tasks:
        raise ValueError("Le graphe de taches contient un cycle.")
    return order
