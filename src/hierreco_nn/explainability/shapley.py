#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch

from hierreco_nn.config import add_config_argument, load_project_config_from_cli
from hierreco_nn.dataset import HierrecoDataset


SHAPLEY_MODES = ("exact", "permutation")


@dataclass(frozen=True)
class FeatureGroup:
    """One Shapley player: a semantically grouped node or edge feature block."""

    name: str
    kind: str
    indices: tuple[int, ...]


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint across PyTorch default variants."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_run_model_class(run_dir: Path):
    """Load ``EdgeNodeGNN`` from the model snapshot stored in a run directory."""
    model_path = run_dir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Run model snapshot not found: {model_path}")

    module_name = f"hierreco_shapley_model_{run_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model snapshot: {model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "EdgeNodeGNN"):
        raise AttributeError(f"{model_path} does not define EdgeNodeGNN")
    return module.EdgeNodeGNN


def model_kwargs_for_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Build constructor kwargs supported by the run snapshot's model class."""
    config = dict(checkpoint.get("model_config") or {})
    kwargs = {
        "node_feature_dim": checkpoint["node_feature_dim"],
        "edge_feature_dim": checkpoint["edge_feature_dim"],
    }
    key_map = {
        "hidden_dim": "hidden_dim",
        "layers": "num_layers",
        "num_layers": "num_layers",
        "attention_heads": "attention_heads",
        "gnn_branch": "gnn_branch",
        "edge_competition_layers": "edge_competition_layers",
        "incident_selector": "incident_selector",
        "dropout": "dropout",
    }
    for source_key, target_key in key_map.items():
        if source_key in config:
            kwargs[target_key] = config[source_key]
    return kwargs


def instantiate_model(run_dir: Path, checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the exact model class used by a run and load weights."""
    model_cls = load_run_model_class(run_dir)
    kwargs = model_kwargs_for_checkpoint(checkpoint)
    signature = inspect.signature(model_cls)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    model = model_cls(**supported_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def config_value(config: dict[str, Any], key: str, default):
    """Return a checkpoint config value with a fallback default."""
    value = config.get(key, default)
    return default if value is None else value


def parse_bin_tuple(value):
    if value is None:
        return None
    return tuple(value)


def build_dataset(args, checkpoint: dict[str, Any]) -> HierrecoDataset:
    """Build a dataset using checkpoint construction parameters plus CLI root."""
    data_config = dict(checkpoint.get("data_config") or {})
    root = (
        args.dataset_root
        or getattr(args, "configured_dataset_root", None)
        or data_config.get("dataset_root")
    )
    if root is None:
        raise ValueError("--dataset-root is required when checkpoint has no data_config")

    return HierrecoDataset(
        root,
        candidate_density=float(config_value(data_config, "candidate_density", 1.0)),
        split_sensitivity=float(config_value(data_config, "split_sensitivity", 1.0)),
        edge_length_strictness=float(config_value(data_config, "edge_length_strictness", 1.0)),
        triangle_min_angle_degrees=float(
            config_value(data_config, "triangle_min_angle_degrees", 30.0)
        ),
        min_sub_nodes=int(config_value(data_config, "min_sub_nodes", 2)),
        min_nodes=int(config_value(data_config, "min_nodes", 5)),
        bin_count=int(config_value(data_config, "bin_count", 5)),
        split_seed=config_value(data_config, "split_seed", 13),
        train_bins=parse_bin_tuple(data_config.get("train_bins")),
        validation_bins=parse_bin_tuple(data_config.get("validation_bins")),
        require_target_coverage=False,
    )


def undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Collapse a directed edge_index tensor to canonical undirected tuples."""
    edges: set[tuple[int, int]] = set()
    if edge_index.numel() == 0:
        return edges

    for source, target in edge_index.t().tolist():
        if source != target:
            edges.add(HierrecoDataset.normalize_edge(int(source), int(target)))
    return edges


def target_edges_from_data(data) -> set[tuple[int, int]]:
    return undirected_edges(data.target_edge_index)


def candidate_graph_recall(data) -> float:
    """Return candidate graph recall against target undirected edges."""
    target_edges = target_edges_from_data(data)
    if not target_edges:
        return 1.0
    candidate_edges = undirected_edges(data.edge_index)
    return len(target_edges & candidate_edges) / len(target_edges)


def prf_f1(
    predicted_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
) -> float:
    """Return F1 from predicted and target undirected edge sets."""
    true_positive = len(predicted_edges & target_edges)
    false_positive = len(predicted_edges - target_edges)
    false_negative = len(target_edges - predicted_edges)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def build_feature_groups(node_dim: int, edge_dim: int) -> list[FeatureGroup]:
    """Return semantic feature groups used as Shapley players."""
    groups: list[FeatureGroup] = []
    node_groups = [
        ("node.raw_xy", (0, 1)),
        ("node.normalized_xy", (2, 3)),
        ("node.main_axis_rank", (4,)),
        ("node.local_geometry", (5, 6, 7)),
        ("node.bbox", (8, 9, 10, 11)),
    ]
    for name, indices in node_groups:
        valid = tuple(index for index in indices if index < node_dim)
        if valid:
            groups.append(FeatureGroup(name, "node", valid))
    if node_dim > 12:
        groups.append(FeatureGroup("node.zernike", "node", tuple(range(12, node_dim))))

    edge_groups = [
        ("edge.signed_delta", (0, 1)),
        ("edge.distance_geometry", (2, 3, 4)),
        ("edge.axis_delta_signed", (5, 6)),
        ("edge.axis_delta_abs", (7, 8)),
        ("edge.axis_order", (9, 10, 11)),
    ]
    for name, indices in edge_groups:
        valid = tuple(index for index in indices if index < edge_dim)
        if valid:
            groups.append(FeatureGroup(name, "edge", valid))
    return groups


def split_indices(dataset: HierrecoDataset, split: str) -> list[int]:
    if split == "train":
        return list(dataset.train_indices)
    if split == "validation":
        return list(dataset.validation_indices)
    raise ValueError(f"Unsupported split: {split}")


def filtered_indices(
    dataset: HierrecoDataset,
    split: str,
    *,
    min_candidate_recall: float,
    limit: int | None,
) -> list[int]:
    """Return split sample indices passing candidate-recall filtering."""
    selected = []
    for sample_index in split_indices(dataset, split):
        data = dataset[sample_index]
        if candidate_graph_recall(data) + 1e-12 < min_candidate_recall:
            continue
        selected.append(sample_index)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def feature_baselines(dataset: HierrecoDataset, indices: list[int]):
    """Return split-level mean feature baselines for node and edge features."""
    node_sum = None
    edge_sum = None
    node_count = 0
    edge_count = 0
    for sample_index in indices:
        data = dataset[sample_index]
        if data.x.numel():
            node_values = data.x.float()
            node_sum = (
                node_values.sum(dim=0)
                if node_sum is None
                else node_sum + node_values.sum(dim=0)
            )
            node_count += int(node_values.size(0))
        if data.edge_attr.numel():
            edge_values = data.edge_attr.float()
            edge_sum = (
                edge_values.sum(dim=0)
                if edge_sum is None
                else edge_sum + edge_values.sum(dim=0)
            )
            edge_count += int(edge_values.size(0))

    if node_sum is None or edge_sum is None:
        raise ValueError("Cannot compute Shapley baselines from an empty split")

    return {
        "node": node_sum / max(node_count, 1),
        "edge": edge_sum / max(edge_count, 1),
    }


def mask_data_to_subset(
    data,
    *,
    groups: list[FeatureGroup],
    keep: frozenset[int],
    baselines: dict[str, torch.Tensor],
):
    """Return a clone where feature groups outside ``keep`` are set to baseline."""
    masked = data.clone()
    masked.x = data.x.clone()
    masked.edge_attr = data.edge_attr.clone()

    for group_index, group in enumerate(groups):
        if group_index in keep:
            continue
        indices = torch.tensor(group.indices, dtype=torch.long)
        if group.kind == "node":
            masked.x[:, indices] = baselines["node"][indices].to(masked.x.dtype)
        elif group.kind == "edge":
            masked.edge_attr[:, indices] = baselines["edge"][indices].to(masked.edge_attr.dtype)

    return masked


@torch.no_grad()
def score_sample(
    model: torch.nn.Module,
    data,
    *,
    device: torch.device,
    threshold: float,
) -> float:
    """Return thresholded undirected edge F1 for one sample."""
    target_edges = target_edges_from_data(data)
    logits = model(data.to(device)).detach().cpu()
    data = data.cpu()
    probabilities = torch.sigmoid(logits)
    predicted_edges = set()
    for edge_id, (source, target) in enumerate(data.edge_index.t().tolist()):
        if float(probabilities[edge_id]) >= threshold and source != target:
            predicted_edges.add(HierrecoDataset.normalize_edge(int(source), int(target)))
    return prf_f1(predicted_edges, target_edges)


@torch.no_grad()
def score_samples(
    model: torch.nn.Module,
    samples: list,
    *,
    device: torch.device,
    threshold: float,
) -> list[float]:
    """Return thresholded F1 for a list of PyG samples in one model call."""
    if not samples:
        return []

    batch = Batch.from_data_list(samples).to(device)
    logits = model(batch).detach().cpu()
    scores = []
    offset = 0
    for sample in samples:
        edge_count = int(sample.edge_index.size(1))
        sample_logits = logits[offset : offset + edge_count]
        offset += edge_count

        probabilities = torch.sigmoid(sample_logits)
        predicted_edges = set()
        for edge_id, (source, target) in enumerate(sample.edge_index.t().tolist()):
            if float(probabilities[edge_id]) >= threshold and source != target:
                predicted_edges.add(
                    HierrecoDataset.normalize_edge(int(source), int(target))
                )
        scores.append(prf_f1(predicted_edges, target_edges_from_data(sample)))
    return scores


def score_keep_subsets(
    model: torch.nn.Module,
    data,
    *,
    groups: list[FeatureGroup],
    baselines: dict[str, torch.Tensor],
    keeps: list[frozenset[int]],
    device: torch.device,
    threshold: float,
    subset_batch_size: int,
) -> dict[frozenset[int], float]:
    """Evaluate masked subset scores in batches."""
    scores: dict[frozenset[int], float] = {}
    for start in range(0, len(keeps), subset_batch_size):
        keep_batch = keeps[start : start + subset_batch_size]
        masked_batch = [
            mask_data_to_subset(
                data,
                groups=groups,
                keep=keep,
                baselines=baselines,
            )
            for keep in keep_batch
        ]
        batch_scores = score_samples(
            model,
            masked_batch,
            device=device,
            threshold=threshold,
        )
        scores.update(zip(keep_batch, batch_scores))
    return scores


def shapley_for_sample(
    model: torch.nn.Module,
    data,
    *,
    groups: list[FeatureGroup],
    baselines: dict[str, torch.Tensor],
    mode: str,
    permutations: int,
    subset_batch_size: int,
    rng: random.Random,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Compute or estimate grouped Shapley values for one sample."""
    contributions = {group.name: 0.0 for group in groups}
    group_indices = list(range(len(groups)))
    score_cache: dict[frozenset[int], float] = {}

    def value(keep: frozenset[int]) -> float:
        if keep not in score_cache:
            masked = mask_data_to_subset(
                data,
                groups=groups,
                keep=keep,
                baselines=baselines,
            )
            score_cache[keep] = score_sample(
                model,
                masked,
                device=device,
                threshold=threshold,
            )
        return score_cache[keep]

    if mode == "exact":
        player_count = len(group_indices)
        all_keeps = [
            frozenset(
                group_index
                for group_index in group_indices
                if mask & (1 << group_index)
            )
            for mask in range(1 << player_count)
        ]
        score_cache = score_keep_subsets(
            model,
            data,
            groups=groups,
            baselines=baselines,
            keeps=all_keeps,
            device=device,
            threshold=threshold,
            subset_batch_size=subset_batch_size,
        )

        for group_index in group_indices:
            group_name = groups[group_index].name
            for keep, previous in score_cache.items():
                if group_index in keep:
                    continue
                subset_size = len(keep)
                weight = 1.0 / (
                    player_count * math.comb(player_count - 1, subset_size)
                )
                updated_keep = frozenset((*keep, group_index))
                contributions[group_name] += weight * (
                    score_cache[updated_keep] - previous
                )
        return contributions

    for _ in range(permutations):
        permutation = list(group_indices)
        rng.shuffle(permutation)
        keep = frozenset()
        previous = value(keep)
        for group_index in permutation:
            updated_keep = frozenset((*keep, group_index))
            current = value(updated_keep)
            contributions[groups[group_index].name] += current - previous
            keep = updated_keep
            previous = current

    return {
        name: value / max(permutations, 1)
        for name, value in contributions.items()
    }


def progress_bar(current: int, total: int, *, split: str) -> None:
    """Write an in-place terminal-only progress bar."""
    terminal = sys.__stdout__
    if total <= 0 or terminal is None:
        return

    width = 28
    filled = int(round(width * current / total))
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * current / total
    terminal.write(f"\r  {split}: [{bar}] {current}/{total} ({percent:5.1f}%)")
    if current == total:
        terminal.write("\n")
    terminal.flush()


def aggregate_shapley(
    sample_values: list[dict[str, float]],
    groups: list[FeatureGroup],
) -> dict[str, dict[str, float]]:
    """Aggregate per-sample Shapley values into median and IQR summaries."""
    summary = {}
    for group in groups:
        values = np.asarray(
            [sample[group.name] for sample in sample_values],
            dtype=float,
        )
        if values.size == 0:
            summary[group.name] = {
                "median": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "iqr": 0.0,
                "mean": 0.0,
            }
            continue
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        summary[group.name] = {
            "median": float(np.median(values)),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "mean": float(np.mean(values)),
        }
    return summary


def explain_split(
    *,
    model: torch.nn.Module,
    dataset: HierrecoDataset,
    split: str,
    groups: list[FeatureGroup],
    mode: str,
    permutations: int,
    subset_batch_size: int,
    seed: int,
    device: torch.device,
    threshold: float,
    min_candidate_recall: float,
    limit: int | None,
) -> dict[str, Any]:
    """Compute grouped Shapley summaries for one dataset split."""
    indices = filtered_indices(
        dataset,
        split,
        min_candidate_recall=min_candidate_recall,
        limit=limit,
    )
    baselines = feature_baselines(dataset, indices)
    rng = random.Random(seed)
    sample_values = []
    base_scores = []
    full_scores = []

    for order, sample_index in enumerate(indices, start=1):
        data = dataset[sample_index]
        sample_shapley = shapley_for_sample(
            model,
            data,
            groups=groups,
            baselines=baselines,
            mode=mode,
            permutations=permutations,
            subset_batch_size=subset_batch_size,
            rng=rng,
            device=device,
            threshold=threshold,
        )
        sample_values.append(sample_shapley)
        base_scores.append(
            score_sample(
                model,
                mask_data_to_subset(
                    data,
                    groups=groups,
                    keep=frozenset(),
                    baselines=baselines,
                ),
                device=device,
                threshold=threshold,
            )
        )
        full_scores.append(score_sample(model, data, device=device, threshold=threshold))
        progress_bar(order, len(indices), split=split)

    return {
        "split": split,
        "samples": len(indices),
        "mode": mode,
        "subset_evaluations_per_sample": 2 ** len(groups) if mode == "exact" else None,
        "permutations": permutations if mode == "permutation" else None,
        "target_score": "f1",
        "threshold": threshold,
        "score_summary": {
            "baseline_median": float(np.median(base_scores)) if base_scores else 0.0,
            "full_median": float(np.median(full_scores)) if full_scores else 0.0,
            "baseline_mean": float(np.mean(base_scores)) if base_scores else 0.0,
            "full_mean": float(np.mean(full_scores)) if full_scores else 0.0,
        },
        "groups": aggregate_shapley(sample_values, groups),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def parse_args():
    project_config = load_project_config_from_cli()
    parser = argparse.ArgumentParser(
        description="Grouped Shapley explainability for Hierreco node/edge features.",
    )
    add_config_argument(parser, project_config)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help=(
            "Dataset root. Defaults to config paths.dataset_root, then "
            "checkpoint data_config.dataset_root."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--mode", choices=SHAPLEY_MODES, default="exact")
    parser.add_argument("--permutations", default=16, type=int)
    parser.add_argument("--subset-batch-size", default=128, type=int)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--min-candidate-recall", default=None, type=float)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-validation-samples", default=None, type=int)
    args = parser.parse_args()
    args.configured_dataset_root = project_config.paths.dataset_root

    if args.permutations < 1:
        parser.error("--permutations must be >= 1")
    if args.subset_batch_size < 1:
        parser.error("--subset-batch-size must be >= 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        parser.error("--max-train-samples must be >= 1")
    if args.max_validation_samples is not None and args.max_validation_samples < 1:
        parser.error("--max-validation-samples must be >= 1")
    if args.min_candidate_recall is not None and not 0.0 <= args.min_candidate_recall <= 1.0:
        parser.error("--min-candidate-recall must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint or run_dir / "best_val_loss.pt"
    output_path = args.output or run_dir / "shapley_explainability.json"
    checkpoint = load_checkpoint(checkpoint_path)
    dataset = build_dataset(args, checkpoint)
    device = torch.device(args.device)
    model = instantiate_model(run_dir, checkpoint).to(device)
    node_dim = int(checkpoint["node_feature_dim"])
    edge_dim = int(checkpoint["edge_feature_dim"])
    groups = build_feature_groups(node_dim, edge_dim)
    data_config = dict(checkpoint.get("data_config") or {})
    min_candidate_recall = (
        args.min_candidate_recall
        if args.min_candidate_recall is not None
        else float(data_config.get("min_candidate_recall", 1.0))
    )

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset: {dataset.root}")
    print(f"device: {device}")
    print(f"target score: thresholded f1")
    print(f"threshold: {args.threshold}")
    print(f"shapley mode: {args.mode}")
    if args.mode == "exact":
        print(f"subset evaluations per sample: {2 ** len(groups)}")
        print(f"subset batch size: {args.subset_batch_size}")
    else:
        print(f"permutations: {args.permutations}")
    print(f"feature groups: {[group.name for group in groups]}")

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset.root),
        "node_feature_dim": node_dim,
        "edge_feature_dim": edge_dim,
        "feature_groups": [
            {
                "name": group.name,
                "kind": group.kind,
                "indices": list(group.indices),
            }
            for group in groups
        ],
        "splits": {
            "train": explain_split(
                model=model,
                dataset=dataset,
                split="train",
                groups=groups,
                mode=args.mode,
                permutations=args.permutations,
                subset_batch_size=args.subset_batch_size,
                seed=args.seed,
                device=device,
                threshold=args.threshold,
                min_candidate_recall=min_candidate_recall,
                limit=args.max_train_samples,
            ),
            "validation": explain_split(
                model=model,
                dataset=dataset,
                split="validation",
                groups=groups,
                mode=args.mode,
                permutations=args.permutations,
                subset_batch_size=args.subset_batch_size,
                seed=args.seed + 1,
                device=device,
                threshold=args.threshold,
                min_candidate_recall=min_candidate_recall,
                limit=args.max_validation_samples,
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"shapley output: {output_path}")
    print(json.dumps(json_ready(payload["splits"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
