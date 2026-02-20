from __future__ import annotations

"""Environment loading and node-construction utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    """Return repository root path from this module location."""
    return Path(__file__).resolve().parents[1]


def _as_positive_float(value: Any, field: str, layer: str) -> float:
    """Validate and cast a strictly positive float configuration value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valeur invalide pour '{field}' dans la couche '{layer}'.") from exc
    if parsed <= 0.0:
        raise ValueError(f"'{field}' doit etre > 0 dans la couche '{layer}'.")
    return parsed


def _as_non_negative_float(value: Any, field: str, layer: str) -> float:
    """Validate and cast a non-negative float configuration value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valeur invalide pour '{field}' dans la couche '{layer}'.") from exc
    if parsed < 0.0:
        raise ValueError(f"'{field}' doit etre >= 0 dans la couche '{layer}'.")
    return parsed


def load_environments(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load environment layer definitions from JSON."""
    env_path = Path(path) if path else _repo_root() / "data" / "environments.json"
    with env_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Format environnement invalide dans {env_path}.")
    return data


def build_nodes(environments: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand layer configs into explicit node dictionaries used by schedulers."""
    if not environments:
        raise ValueError("Aucun environnement disponible pour construire les noeuds.")

    nodes: List[Dict[str, Any]] = []
    node_id = 0
    for layer_name, cfg in environments.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Configuration invalide pour la couche '{layer_name}'.")
        required = {
            "devices",
            "processing_rate",
            "processing_cost",
            "idle_power",
            "working_power",
            "uplink_bandwidth",
            "downlink_bandwidth",
        }
        missing = sorted(required - set(cfg.keys()))
        if missing:
            raise KeyError(f"Champs manquants pour '{layer_name}': {', '.join(missing)}")

        try:
            devices = int(cfg["devices"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Valeur 'devices' invalide dans la couche '{layer_name}'.") from exc
        if devices <= 0:
            raise ValueError(f"'devices' doit etre > 0 dans la couche '{layer_name}'.")

        processing_rate = _as_positive_float(cfg["processing_rate"], "processing_rate", layer_name)
        processing_cost = _as_non_negative_float(cfg["processing_cost"], "processing_cost", layer_name)
        idle_power = _as_non_negative_float(cfg["idle_power"], "idle_power", layer_name)
        working_power = _as_non_negative_float(cfg["working_power"], "working_power", layer_name)
        uplink = _as_positive_float(cfg["uplink_bandwidth"], "uplink_bandwidth", layer_name)
        downlink = _as_positive_float(cfg["downlink_bandwidth"], "downlink_bandwidth", layer_name)

        for _ in range(devices):
            nodes.append(
                {
                    "id": node_id,
                    "type": layer_name,
                    "processing_rate": processing_rate,
                    "processing_cost": processing_cost,
                    "idle_power": idle_power,
                    "working_power": working_power,
                    "uplink_bandwidth": uplink,
                    "downlink_bandwidth": downlink,
                }
            )
            node_id += 1
    return nodes
