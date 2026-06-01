import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from motndp.city import City, matrix_from_file
from motndp.constraints import MetroConstraints
from motndp.motndp import MOTNDP

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to export dashboard data.") from exc

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except ImportError:  # pragma: no cover
    wandb_internal_pb2 = None
    DataStore = None


ACTION_LABELS = [
    "Up",
    "Up-right",
    "Right",
    "Down-right",
    "Down",
    "Down-left",
    "Left",
    "Up-left",
]


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepStartValueNetwork(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _to_python(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _city_name_from_env_id(env_id):
    stem = env_id.replace("motndp_", "")
    return stem.split("-v", 1)[0]


def _groups_file_for_city(city_name, nr_groups):
    if city_name in {"amsterdam", "amsterdam_10x10", "xian"}:
        return f"price_groups_{nr_groups}.txt"
    return "groups.txt"


def _resolve_city_path(repo_root, city_name):
    candidates = [repo_root / "cities" / city_name]
    if city_name == "dilemma":
        candidates.append(repo_root / "cities" / "dilemma_5x5")
    if city_name == "margins":
        candidates.append(repo_root / "cities" / "margins_5x5")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not locate city data for {city_name}. Checked: {candidates}")


def _format_price_band(min_value, max_value):
    lower = int(np.floor(min_value))
    upper = int(np.ceil(max_value))
    return f"EUR{lower}k-EUR{upper}k"


def _socioeconomic_labels_from_prices(city_path, overlay_grid):
    price_file = city_path / "average_house_price_gid.txt"
    if not price_file.exists():
        return []

    prices = matrix_from_file(price_file, overlay_grid.shape[0], overlay_grid.shape[1])
    prices[prices == 0] = np.nan
    labels = []
    groups = np.unique(overlay_grid[~np.isnan(overlay_grid)])
    for group in groups:
        values = prices[overlay_grid == group]
        values = values[~np.isnan(values)]
        if values.size == 0:
            labels.append(f"Socio-economic group {int(group)}")
            continue
        labels.append(_format_price_band(values.min(), values.max()))
    return labels


def _socioeconomic_values_from_prices(city_path, overlay_grid):
    price_file = city_path / "average_house_price_gid.txt"
    if not price_file.exists():
        return {
            "grid": _to_python(overlay_grid),
            "min": None,
            "max": None,
        }

    prices = matrix_from_file(price_file, overlay_grid.shape[0], overlay_grid.shape[1])
    prices[prices == 0] = np.nan
    prices[np.isnan(overlay_grid)] = np.nan
    finite_values = prices[np.isfinite(prices)]
    return {
        "grid": _to_python(prices),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }


def _socioeconomic_overlay(city_name, city_path, fallback_groups):
    if city_name in {"amsterdam", "amsterdam_10x10", "xian"}:
        overlay_file = city_path / "price_groups_5.txt"
        if overlay_file.exists():
            overlay_city = City(city_path, groups_file=overlay_file.name, ignore_existing_lines=True)
            labels = _socioeconomic_labels_from_prices(city_path, overlay_city.grid_groups)
            price_values = _socioeconomic_values_from_prices(city_path, overlay_city.grid_groups)
            return {
                "grid": _to_python(overlay_city.grid_groups),
                "values": price_values["grid"],
                "value_min": price_values["min"],
                "value_max": price_values["max"],
                "labels": labels or [f"Socio-economic group {int(group)}" for group in overlay_city.groups],
            }

    return {
        "grid": _to_python(fallback_groups),
        "values": _to_python(fallback_groups),
        "value_min": None,
        "value_max": None,
        "labels": [],
    }


def _load_run_config(run_id, repo_root):
    matches = sorted(repo_root.glob(f"wandb/**/run-*-{run_id}/files/config.yaml"))
    matches += sorted(repo_root.glob(f"wandb/**/offline-run-*-{run_id}/files/config.yaml"))
    if not matches:
        return None

    with matches[-1].open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "value" in value:
            config[key] = value["value"]
    return config


def _load_local_run_config(run_id, repo_root):
    candidates = [
        repo_root / "q_tables" / f"{run_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8-sig"))
    return None


def _relative_dashboard_path(path, repo_root):
    dashboard_root = repo_root / "dashboard"
    try:
        relative_path = path.relative_to(dashboard_root)
        return f"./{relative_path.as_posix()}"
    except ValueError:
        relative_path = Path("..") / path.relative_to(repo_root)
        return relative_path.as_posix()


def _coerce_json_value(value):
    if isinstance(value, dict):
        return {key: _coerce_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    return value


def _wandb_run_directory(repo_root, run_id):
    matches = sorted(repo_root.glob(f"wandb/**/run-*-{run_id}")) + sorted(repo_root.glob(f"wandb/**/offline-run-*-{run_id}"))
    return matches[-1] if matches else None


def _parse_json_scalar(raw_value):
    try:
        value = json.loads(raw_value)
    except Exception:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _downsample_series(points, max_points=400):
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _history_item_key(item):
    if item.nested_key:
        return ".".join(item.nested_key)
    return item.key


def _wandb_history_series(run_dir):
    if DataStore is None or wandb_internal_pb2 is None:
        return {}

    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return {}

    series_keys = [
        "reward",
        "average_reward",
        "Average-Test-Reward",
        "best_episode_reward",
        "epsilon",
        "training_step",
        "training_phase_code",
        "train_q_network",
        "train_start_network",
        "bonus_existing_connection",
        "penalty_existing_overlap",
        "episode",
    ]
    series = {key: [] for key in series_keys}

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_files[-1]))
    try:
        while True:
            data = datastore.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            row = {}
            for item in record.history.item:
                key = _history_item_key(item)
                if key in series or key == "_step":
                    scalar_value = _parse_json_scalar(item.value_json)
                    if scalar_value is not None:
                        row[key] = scalar_value

            x_value = row.get("episode", row.get("_step"))
            if x_value is None:
                continue

            for key in series:
                if key in row:
                    series[key].append([float(x_value), float(row[key])])
    except Exception:
        return {}
    finally:
        datastore.close()

    return {key: _downsample_series(points) for key, points in series.items() if points}


def _wandb_system_series(run_dir):
    if DataStore is None or wandb_internal_pb2 is None:
        return {}

    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return {}

    series_keys = [
        "gpu.0.powerWatts",
        "gpu.0.gpu",
        "gpu.0.memory",
        "gpu.0.memoryAllocated",
        "gpu.0.temp",
        "gpu.0.smClock",
        "gpu.0.memoryClock",
        "cpu",
        "memory_percent",
        "proc.memory.rssMB",
        "proc.memory.percent",
        "proc.cpu.threads",
    ]
    series = {key: [] for key in series_keys}
    first_timestamp = None

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_files[-1]))
    try:
        while True:
            data = datastore.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "stats":
                continue

            timestamp = record.stats.timestamp.seconds + record.stats.timestamp.nanos / 1_000_000_000
            if first_timestamp is None:
                first_timestamp = timestamp
            elapsed_seconds = timestamp - first_timestamp

            for item in record.stats.item:
                if item.key not in series:
                    continue
                scalar_value = _parse_json_scalar(item.value_json)
                if scalar_value is not None:
                    series[item.key].append([float(elapsed_seconds), float(scalar_value)])
    except Exception:
        return {}
    finally:
        datastore.close()

    return {key: _downsample_series(points) for key, points in series.items() if points}


def _series_summary(series):
    summary = {}
    for key, points in series.items():
        values = [float(point[1]) for point in points if len(point) > 1]
        if not values:
            continue
        summary[key] = {
            "latest": values[-1],
            "avg": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }
    return summary


def _latest_carbon_metrics(repo_root, run_id):
    emissions_path = repo_root / "carbon_logs" / "emissions.csv"
    if not emissions_path.exists():
        return None

    matches = []
    with emissions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("project_name") == run_id:
                matches.append(row)

    if not matches:
        return None

    row = matches[-1]
    numeric_keys = [
        "duration",
        "emissions",
        "emissions_rate",
        "cpu_power",
        "gpu_power",
        "ram_power",
        "cpu_energy",
        "gpu_energy",
        "ram_energy",
        "energy_consumed",
        "water_consumed",
        "cpu_utilization_percent",
        "gpu_utilization_percent",
        "ram_utilization_percent",
        "ram_used_gb",
    ]
    metrics = {}
    for key, value in row.items():
        if key in numeric_keys:
            metrics[key] = _parse_json_scalar(value)
        elif key in {
            "timestamp",
            "country_name",
            "region",
            "os",
            "python_version",
            "codecarbon_version",
            "cpu_model",
            "gpu_model",
        }:
            metrics[key] = value
    return metrics


def _wandb_details(run_id, repo_root):
    run_dir = _wandb_run_directory(repo_root, run_id)
    if run_dir is None:
        return None

    files_dir = run_dir / "files"
    summary_path = files_dir / "wandb-summary.json"
    metadata_path = files_dir / "wandb-metadata.json"
    output_log_path = files_dir / "output.log"

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    media_items = []
    media_dir = files_dir / "media" / "images"
    if media_dir.exists():
        for image_path in sorted(media_dir.glob("*")):
            label = re.sub(r"_[0-9]+_[a-f0-9]+$", "", image_path.stem)
            media_items.append(
                {
                    "label": label,
                    "path": _relative_dashboard_path(image_path, repo_root),
                }
            )

    summary_metrics = {}
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)):
            summary_metrics[key] = value

    history_series = _wandb_history_series(run_dir)
    system_series = _wandb_system_series(run_dir)
    system_summary = _series_summary(system_series)
    carbon_metrics = _latest_carbon_metrics(repo_root, run_id)
    for key, points in history_series.items():
        if key not in summary_metrics and points:
            summary_metrics[key] = points[-1][1]

    return {
        "run_id": run_id,
        "run_dir": _relative_dashboard_path(run_dir, repo_root),
        "summary_metrics": _coerce_json_value(summary_metrics),
        "history_series": _coerce_json_value(history_series),
        "system_series": _coerce_json_value(system_series),
        "system_summary": _coerce_json_value(system_summary),
        "carbon_metrics": _coerce_json_value(carbon_metrics),
        "metadata": _coerce_json_value(
            {
                "startedAt": metadata.get("startedAt"),
                "python": metadata.get("python"),
                "os": metadata.get("os"),
                "gpu": metadata.get("gpu"),
                "cpu_count_logical": metadata.get("cpu_count_logical"),
                "program": metadata.get("program"),
                "args": metadata.get("args", []),
            }
        ),
        "media": media_items,
        "output_log_path": _relative_dashboard_path(output_log_path, repo_root) if output_log_path.exists() else None,
    }


def _discover_tabular_models(repo_root):
    q_table_dir = repo_root / "q_tables"
    models = []
    for q_path in sorted(q_table_dir.glob("*.npy")):
        if q_path.stem.endswith("_qstart"):
            continue

        run_id = q_path.stem
        if run_id.startswith("smoke-"):
            continue
        config = _load_run_config(run_id, repo_root) or _load_local_run_config(run_id, repo_root)
        if not config:
            continue

        q_start_path = q_table_dir / f"{run_id}_qstart.npy"
        models.append(
            {
                "id": run_id,
                "kind": "tabular",
                "q_path": q_path,
                "q_start_path": q_start_path if q_start_path.exists() else None,
                "config": config,
            }
        )
    return models


def _discover_deep_models(repo_root):
    if torch is None:
        return []

    models = []
    for checkpoint_path in sorted((repo_root / "deep_models").glob("*.pt")):
        if checkpoint_path.stem.startswith("smoke-"):
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config")
        if not config:
            continue

        models.append(
            {
                "id": checkpoint_path.stem,
                "kind": "deep_rl",
                "checkpoint_path": checkpoint_path,
                "config": config,
            }
        )
    return models


def _serialize_grid_lines(city, lines):
    serialized = []
    for line in lines:
        station_locs = city.index_to_grid(line.flatten())
        serialized.append(station_locs.tolist())
    return serialized


def _build_tabular_layers(city, q_table, q_start):
    q_best_value = q_table.max(axis=1).reshape(city.grid_x_size, city.grid_y_size)
    q_best_action = q_table.argmax(axis=1).reshape(city.grid_x_size, city.grid_y_size)

    return {
        "aggregate_od": _to_python(city.agg_od_mx()),
        "q_best_value": _to_python(q_best_value),
        "q_best_action": _to_python(q_best_action),
        "q_start": _to_python(q_start),
    }


def _build_deep_layers(city, q_network, start_network, device):
    grid_size = city.grid_size
    start_values = []
    q_best_values = []
    q_best_actions = []

    start_features = []
    for row in range(city.grid_x_size):
        for col in range(city.grid_y_size):
            start_features.append([row / max(city.grid_x_size - 1, 1), col / max(city.grid_y_size - 1, 1)])

    with torch.no_grad():
        start_tensor = torch.as_tensor(start_features, dtype=torch.float32, device=device)
        start_values = start_network(start_tensor).reshape(city.grid_x_size, city.grid_y_size).cpu().numpy()

        observations = np.zeros((grid_size, grid_size * 2), dtype=np.float32)
        for idx in range(grid_size):
            observations[idx, idx] = 1.0
            observations[idx, grid_size + idx] = 1.0

        q_values = q_network(torch.as_tensor(observations, dtype=torch.float32, device=device)).cpu().numpy()
        q_best_value = q_values.max(axis=1).reshape(city.grid_x_size, city.grid_y_size)
        q_best_action = q_values.argmax(axis=1).reshape(city.grid_x_size, city.grid_y_size)

    return {
        "aggregate_od": _to_python(city.agg_od_mx()),
        "q_best_value": _to_python(q_best_value),
        "q_best_action": _to_python(q_best_action),
        "q_start": _to_python(start_values),
    }


def _valid_start_mask(city):
    if not hasattr(city, "grid_groups"):
        return None
    mask = np.isfinite(city.grid_groups)
    return mask if np.any(mask) else None


def _masked_start_values(q_start, valid_start_mask):
    q_start_array = np.asarray(q_start, dtype=np.float64)
    if valid_start_mask is None:
        return q_start_array
    return np.where(valid_start_mask, q_start_array, -np.inf)


def _best_start_location(q_start, valid_start_mask):
    masked_values = _masked_start_values(q_start, valid_start_mask)
    if not np.any(np.isfinite(masked_values)):
        masked_values = np.asarray(q_start, dtype=np.float64)
    return np.unravel_index(int(np.argmax(masked_values)), masked_values.shape)


def _summarize_action_mask(mask):
    return [ACTION_LABELS[idx] for idx, allowed in enumerate(mask) if allowed]


def _base_model_payload(model, city, config, socioeconomic_overlay, timeline, info, cumulative_reward, base_layers, starting_location, title):
    valid_mask = _valid_start_mask(city)
    ranked_start_values = _masked_start_values(base_layers["q_start"], valid_mask)
    start_rank = np.argsort(ranked_start_values.ravel())[::-1]
    top_starts = []
    q_start_array = np.asarray(base_layers["q_start"])
    for index in start_rank:
        row, col = np.unravel_index(index, q_start_array.shape)
        if not np.isfinite(ranked_start_values[row, col]):
            continue
        top_starts.append(
            {
                "location": [int(row), int(col)],
                "value": float(q_start_array[row, col]),
            }
        )
        if len(top_starts) >= 10:
            break

    return {
        "id": model["id"],
        "kind": model["kind"],
        "title": title,
        "city_name": _city_name_from_env_id(config["env_id"]),
        "env_id": config["env_id"],
        "nr_groups": int(config["nr_groups"]),
        "nr_stations": int(config["nr_stations"]),
        "od_type": config["od_type"],
        "chained_reward": bool(config.get("chained_reward", False)),
        "starting_location": [int(starting_location[0]), int(starting_location[1])],
        "grid": {
            "rows": city.grid_x_size,
            "cols": city.grid_y_size,
            "size": city.grid_size,
        },
        "socioeconomic_group_labels": socioeconomic_overlay["labels"],
        "group_labels": [f"Group {int(group)}" for group in city.groups],
        "group_totals": _to_python(city.group_od_sum),
        "existing_lines": _serialize_grid_lines(city, city.existing_lines),
        "base_layers": base_layers,
        "socioeconomic_groups": socioeconomic_overlay["grid"],
        "socioeconomic_values": socioeconomic_overlay["values"],
        "socioeconomic_value_min": socioeconomic_overlay["value_min"],
        "socioeconomic_value_max": socioeconomic_overlay["value_max"],
        "top_starting_cells": top_starts,
        "timeline": timeline,
        "summary": {
            "steps_taken": len(timeline) - 1,
            "final_location": _to_python(info["location_grid_coordinates"]),
            "covered_cells": _to_python(info["covered_cells_coordinates"]),
            "final_reward_vector": _to_python(cumulative_reward),
            "final_reward_total": float(cumulative_reward.sum()),
        },
        "wandb": _wandb_details(config.get("wandb_run_id", model["id"]), Path(__file__).resolve().parents[1]),
        "config": _to_python(config),
    }


def _simulate_tabular_model(model, repo_root):
    config = model["config"]
    city_name = _city_name_from_env_id(config["env_id"])
    city_path = _resolve_city_path(repo_root, city_name)
    groups_file = _groups_file_for_city(city_name, int(config["nr_groups"]))

    city = City(city_path, groups_file=groups_file, ignore_existing_lines=bool(config.get("ignore_existing_lines", False)))
    socioeconomic_overlay = _socioeconomic_overlay(city_name, city_path, city.grid_groups)
    env = MOTNDP(
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=int(config["nr_stations"]),
        od_type=config["od_type"],
        chained_reward=bool(config.get("chained_reward", False)),
    )

    q_table = np.load(model["q_path"])
    q_start = np.load(model["q_start_path"]) if model["q_start_path"] is not None else q_table.max(axis=1).reshape(city.grid_x_size, city.grid_y_size)
    start_loc = _best_start_location(q_start, _valid_start_mask(city))
    observation, info = env.reset(options={"loc": start_loc})
    timeline = [
        {
            "step": 0,
            "kind": "start",
            "location": _to_python(info["location_grid_coordinates"]),
            "location_index": int(info["location_grid_index"]),
            "path": _to_python(info["covered_cells_coordinates"]),
            "allowed_actions": _to_python(info["action_mask"]),
            "allowed_action_labels": _summarize_action_mask(info["action_mask"]),
            "q_values": _to_python(q_table[int(info["location_grid_index"])]),
            "reward_vector": [0.0 for _ in range(int(config["nr_groups"]))],
            "reward_total": 0.0,
            "done": False,
            "description": f"Agent starts at cell {tuple(start_loc)}.",
        }
    ]
    cumulative_reward = np.zeros(int(config["nr_groups"]), dtype=np.float64)

    while True:
        state_index = int(info["location_grid_index"])
        masked_q_values = np.where(info["action_mask"], q_table[state_index], -np.inf)
        action = int(np.argmax(masked_q_values))
        from_location = np.asarray(info["location_grid_coordinates"]).tolist()
        observation, reward, done, _, next_info = env.step(action)
        to_location = np.asarray(next_info["location_grid_coordinates"]).tolist()
        reward = np.asarray(reward, dtype=np.float64)
        cumulative_reward += reward
        timeline.append(
            {
                "step": len(timeline),
                "kind": "move",
                "action": action,
                "action_label": ACTION_LABELS[action],
                "from": from_location,
                "to": to_location,
                "location": to_location,
                "location_index": int(next_info["location_grid_index"]),
                "path": _to_python(next_info["covered_cells_coordinates"]),
                "allowed_actions": _to_python(next_info["action_mask"]),
                "allowed_action_labels": _summarize_action_mask(next_info["action_mask"]),
                "q_values": _to_python(q_table[state_index]),
                "reward_vector": _to_python(reward),
                "reward_total": float(reward.sum()),
                "cumulative_reward": _to_python(cumulative_reward),
                "done": bool(done),
                "description": f"Move {ACTION_LABELS[action].lower()} from {tuple(from_location)} to {tuple(to_location)}.",
            }
        )
        info = next_info
        if done:
            break

    base_layers = _build_tabular_layers(city, q_table, q_start)
    return _base_model_payload(
        model,
        city,
        config,
        socioeconomic_overlay,
        timeline,
        info,
        cumulative_reward,
        base_layers,
        start_loc,
        f"{city_name.title()} tabular policy {model['id']}",
    )


def _simulate_deep_model(model, repo_root):
    if torch is None:
        raise RuntimeError("PyTorch is required to export deep RL dashboard data.")

    config = model["config"]
    city_name = _city_name_from_env_id(config["env_id"])
    city_path = _resolve_city_path(repo_root, city_name)
    groups_file = _groups_file_for_city(city_name, int(config["nr_groups"]))

    city = City(city_path, groups_file=groups_file, ignore_existing_lines=bool(config.get("ignore_existing_lines", False)))
    socioeconomic_overlay = _socioeconomic_overlay(city_name, city_path, city.grid_groups)
    env = MOTNDP(
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=int(config["nr_stations"]),
        od_type=config["od_type"],
        chained_reward=bool(config.get("chained_reward", False)),
    )

    device = torch.device("cpu")
    checkpoint = torch.load(model["checkpoint_path"], map_location=device)
    q_network = DeepQNetwork(city.grid_size * 2, env.action_space.n, config["hidden_dims"]).to(device)
    start_network = DeepStartValueNetwork(config["start_hidden_dims"]).to(device)
    q_network.load_state_dict(checkpoint["q_network"])
    start_network.load_state_dict(checkpoint["start_network"])
    q_network.eval()
    start_network.eval()

    base_layers = _build_deep_layers(city, q_network, start_network, device)
    q_start_array = np.asarray(base_layers["q_start"])
    start_loc = _best_start_location(q_start_array, _valid_start_mask(city))
    observation, info = env.reset(options={"loc": start_loc})

    timeline = [
        {
            "step": 0,
            "kind": "start",
            "location": _to_python(info["location_grid_coordinates"]),
            "location_index": int(info["location_grid_index"]),
            "path": _to_python(info["covered_cells_coordinates"]),
            "allowed_actions": _to_python(info["action_mask"]),
            "allowed_action_labels": _summarize_action_mask(info["action_mask"]),
            "q_values": _to_python(q_network(torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()),
            "reward_vector": [0.0 for _ in range(int(config["nr_groups"]))],
            "reward_total": 0.0,
            "done": False,
            "description": f"Deep RL agent starts at cell {tuple(start_loc)}.",
        }
    ]
    cumulative_reward = np.zeros(int(config["nr_groups"]), dtype=np.float64)

    while True:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(obs_tensor).squeeze(0).cpu().numpy()
        masked_q_values = np.where(info["action_mask"], q_values, -np.inf)
        action = int(np.argmax(masked_q_values))
        from_location = np.asarray(info["location_grid_coordinates"]).tolist()
        observation, reward, done, _, next_info = env.step(action)
        to_location = np.asarray(next_info["location_grid_coordinates"]).tolist()
        reward = np.asarray(reward, dtype=np.float64)
        cumulative_reward += reward
        timeline.append(
            {
                "step": len(timeline),
                "kind": "move",
                "action": action,
                "action_label": ACTION_LABELS[action],
                "from": from_location,
                "to": to_location,
                "location": to_location,
                "location_index": int(next_info["location_grid_index"]),
                "path": _to_python(next_info["covered_cells_coordinates"]),
                "allowed_actions": _to_python(next_info["action_mask"]),
                "allowed_action_labels": _summarize_action_mask(next_info["action_mask"]),
                "q_values": _to_python(q_values),
                "reward_vector": _to_python(reward),
                "reward_total": float(reward.sum()),
                "cumulative_reward": _to_python(cumulative_reward),
                "done": bool(done),
                "description": f"Deep RL move {ACTION_LABELS[action].lower()} from {tuple(from_location)} to {tuple(to_location)}.",
            }
        )
        info = next_info
        if done:
            break

    return _base_model_payload(
        model,
        city,
        config,
        socioeconomic_overlay,
        timeline,
        info,
        cumulative_reward,
        base_layers,
        start_loc,
        f"{city_name.title()} deep RL policy {model['id']}",
    )


def _serialize_model(model, repo_root):
    if model["kind"] == "tabular":
        return _simulate_tabular_model(model, repo_root)
    return _simulate_deep_model(model, repo_root)


def _default_comparison_model(models, selected_model):
    apple_pairs = {
        "tabular-rl-apple": "deep-rl-apple",
        "deep-rl-apple": "tabular-rl-apple",
        "tabular-rl-orange": "deep-rl-orange",
        "deep-rl-orange": "tabular-rl-orange",
        "tabular-rl-amsterdam-10x10-20k": "deep-rl-amsterdam-10x10-20k",
        "deep-rl-amsterdam-10x10-20k": "tabular-rl-amsterdam-10x10-20k",
        "tabular-rl-amsterdam-10x10-20k-rawls": "deep-rl-amsterdam-10x10-20k-rawls",
        "deep-rl-amsterdam-10x10-20k-rawls": "tabular-rl-amsterdam-10x10-20k-rawls",
    }
    paired_model_id = apple_pairs.get(selected_model["id"])
    if paired_model_id and any(model["id"] == paired_model_id for model in models):
        return paired_model_id

    same_city = [model for model in models if model["city_name"] == selected_model["city_name"] and model["id"] != selected_model["id"]]
    if not same_city:
        return None

    alternate_kind = [model for model in same_city if model["kind"] != selected_model["kind"]]
    if alternate_kind:
        return alternate_kind[0]["id"]

    amsterdam_deep = [model for model in same_city if model["kind"] == "deep_rl"]
    if amsterdam_deep:
        return amsterdam_deep[0]["id"]

    return same_city[0]["id"]


def export_dashboard_data(output_path):
    repo_root = Path(__file__).resolve().parents[1]
    models = _discover_tabular_models(repo_root) + _discover_deep_models(repo_root)
    serialized_models = [_serialize_model(model, repo_root) for model in models]

    comparison_defaults = {}
    for model in serialized_models:
        comparison_model_id = _default_comparison_model(serialized_models, model)
        if comparison_model_id:
            comparison_defaults[model["id"]] = comparison_model_id

    payload = {
        "generated_from": str(repo_root),
        "model_count": len(serialized_models),
        "models": serialized_models,
        "comparison_defaults": comparison_defaults,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "window.MOTNDP_DASHBOARD_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export dashboard data for the MO-TNDP visualizer.")
    parser.add_argument("--output", default="dashboard/data.js", type=Path, help="Path to the generated JavaScript data bundle.")
    args = parser.parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    payload = export_dashboard_data(output_path)
    print(f"Exported {payload['model_count']} model(s) to {output_path}")


if __name__ == "__main__":
    main()
