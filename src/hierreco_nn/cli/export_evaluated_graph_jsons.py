#!/usr/bin/env python3

"""Export evaluated Hierreco graphs to one JSON file per input sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hierreco_nn.config import add_config_argument, load_project_config_from_cli

DECODER_CHOICES = ("greedy", "ilp")
torch = None
build_dataset = None
candidate_graph_recall = None
evaluate_sample = None
instantiate_model = None
load_checkpoint = None
undirected_edges = None


class MissingMLDependency(RuntimeError):
    """Raised when the active Python environment cannot run model evaluation."""


def parse_bin_spec(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated bin list, or return ``None`` for defaults."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ()

    return tuple(int(part) for part in parts)


def load_ml_dependencies() -> None:
    """Load PyTorch-dependent project helpers after argparse handles --help."""
    global torch
    global build_dataset
    global candidate_graph_recall
    global evaluate_sample
    global instantiate_model
    global load_checkpoint
    global undirected_edges

    try:
        import torch as torch_module
        from hierreco_nn.cli.visualize_run_predictions import (
            build_dataset as build_dataset_fn,
            candidate_graph_recall as candidate_graph_recall_fn,
            evaluate_sample as evaluate_sample_fn,
            instantiate_model as instantiate_model_fn,
            load_checkpoint as load_checkpoint_fn,
            undirected_edges as undirected_edges_fn,
        )
    except ModuleNotFoundError as error:
        raise MissingMLDependency(
            "Evaluation export requires the same Python environment as model "
            "training/visualization, including torch and torch_geometric."
        ) from error

    torch = torch_module
    build_dataset = build_dataset_fn
    candidate_graph_recall = candidate_graph_recall_fn
    evaluate_sample = evaluate_sample_fn
    instantiate_model = instantiate_model_fn
    load_checkpoint = load_checkpoint_fn
    undirected_edges = undirected_edges_fn


def selected_split_indices(dataset, split_name: str):
    """Return split/index pairs requested by the CLI."""
    if split_name == "train":
        return [("train", index) for index in dataset.train_indices]
    if split_name == "validation":
        return [("validation", index) for index in dataset.validation_indices]

    return (
        [("train", index) for index in dataset.train_indices]
        + [("validation", index) for index in dataset.validation_indices]
    )


def scalar_or_none(value: Any) -> Any:
    """Return a JSON-friendly scalar while preserving missing values."""
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        value = value.detach().cpu().item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return value


def float_or_none(value: Any) -> float | None:
    """Return ``value`` as float, or ``None`` when coordinates are unavailable."""
    value = scalar_or_none(value)
    if value is None:
        return None
    return float(value)


def sequence_num(value: Any) -> int | float | str | None:
    """Normalize sequence number values without inventing missing labels."""
    value = scalar_or_none(value)
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    return int(number) if number.is_integer() else number


def original_positions(data):
    """Return original node coordinates stored on a PyG data object."""
    fallback = [(None, None) for _ in range(int(data.num_nodes))]
    positions = getattr(data, "original_positions", fallback)

    if torch is not None and torch.is_tensor(positions):
        return [tuple(item) for item in positions.detach().cpu().tolist()]

    return list(positions)


def tensor_list(value, default=None):
    """Convert optional tensor-like metadata to a Python list."""
    if value is None:
        return default
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def load_source_sample(data) -> tuple[Path, dict[str, Any]]:
    """Load the source JSON sample for metadata not stored in the dataset."""
    sample_path = Path(getattr(data, "sample_path", ""))
    if not sample_path.exists():
        raise FileNotFoundError(f"Cannot load source sample JSON: {sample_path}")

    with sample_path.open("r", encoding="utf-8") as file:
        return sample_path, json.load(file)


def node_records(data, sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build shared node records for all exported graph edge sets."""
    node_ids = list(getattr(data, "node_ids", sample["nodes_array"]))
    blob_ids = list(getattr(data, "blob_ids", node_ids))
    points = data.pos.detach().cpu().tolist()
    source_positions = original_positions(data)
    features = sample.get("feature", {})
    sub_ids = tensor_list(getattr(data, "sub_id", None), default=[])
    boundary_mask = tensor_list(getattr(data, "subline_boundary_mask", None), default=[])
    nodes = []

    for index, node_id in enumerate(node_ids):
        feature = features.get(node_id, {})
        fallback_original_x, fallback_original_y = source_positions[index]
        original_x = feature.get("original_x", fallback_original_x)
        original_y = feature.get("original_y", fallback_original_y)

        node = {
            "index": index,
            "id": node_id,
            "blob_id": blob_ids[index],
            "sequence_num": sequence_num(
                feature.get("sequence_pos", feature.get("sequence_num"))
            ),
            "x": float(points[index][0]),
            "y": float(points[index][1]),
            "x_original": float_or_none(original_x),
            "y_original": float_or_none(original_y),
        }

        if index < len(sub_ids):
            node["sub_id"] = int(sub_ids[index])
        if index < len(boundary_mask):
            node["is_subline_boundary"] = bool(boundary_mask[index])

        nodes.append(node)

    return nodes


def edge_records(
    edges: set[tuple[int, int]],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build JSON-friendly undirected edge records."""
    records = []
    for source, target in sorted(edges):
        records.append(
            {
                "source": int(source),
                "target": int(target),
                "source_id": nodes[source]["id"],
                "target_id": nodes[target]["id"],
            }
        )
    return records


def substructure_records(data, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export substructure bboxes and boundary nodes used by the visualizer."""
    bboxes = tensor_list(getattr(data, "sub_bboxes", None), default=[])
    boundary_nodes = getattr(data, "sub_boundary_nodes", [])
    records = []

    for sub_id, bbox in enumerate(bboxes):
        member_nodes = [
            node["index"]
            for node in nodes
            if node.get("sub_id") == sub_id
        ]
        boundary = (
            list(boundary_nodes[sub_id])
            if sub_id < len(boundary_nodes)
            else []
        )
        records.append(
            {
                "sub_id": sub_id,
                "bbox": [float(value) for value in bbox],
                "nodes": member_nodes,
                "boundary_nodes": [int(node) for node in boundary],
            }
        )

    return records


def build_export_payload(
    *,
    data,
    sample_path: Path,
    sample: dict[str, Any],
    split: str,
    sample_index: int,
    predicted_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
    precision: float,
    recall: float,
    f1: float,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    candidate_recall: float,
) -> dict[str, Any]:
    """Build one per-sample offline graph JSON payload."""
    candidate_edges = undirected_edges(data.edge_index)
    nodes = node_records(data, sample)

    return {
        "sample_name": sample_path.name,
        "sample_path": str(sample_path),
        "split": split,
        "sample_index": int(sample_index),
        "component_id": getattr(data, "component_id", ""),
        "graph_type": getattr(data, "graph_type", sample.get("graph_type")),
        "metrics": {
            "candidate_recall": float(candidate_recall),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "candidate_count": len(candidate_edges),
            "gt_count": len(target_edges),
            "predicted_count": len(predicted_edges),
        },
        "nodes": nodes,
        "candidate_graph": {
            "edge_count": len(candidate_edges),
            "edges": edge_records(candidate_edges, nodes),
        },
        "gt_graph": {
            "edge_count": len(target_edges),
            "edges": edge_records(target_edges, nodes),
        },
        "predicted_graph": {
            "edge_count": len(predicted_edges),
            "edges": edge_records(predicted_edges, nodes),
        },
        "substructures": substructure_records(data, nodes),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def export_graph_jsons(
    *,
    model,
    dataset,
    split_name: str,
    device,
    output_dir: Path,
    min_candidate_recall: float,
    limit: int | None,
    decoder: str,
) -> dict[str, Any]:
    """Evaluate selected samples and export their graph JSON payloads."""
    split_indices = selected_split_indices(dataset, split_name)
    if limit is not None:
        split_indices = split_indices[:limit]

    written_names: dict[str, Path] = {}
    evaluated_samples = 0
    skipped_samples = 0
    written_samples = 0

    for split, sample_index in split_indices:
        data = dataset[sample_index]
        sample_candidate_recall = candidate_graph_recall(data)
        if sample_candidate_recall + 1e-12 < min_candidate_recall:
            skipped_samples += 1
            continue

        (
            predicted_edges,
            target_edges,
            precision,
            recall,
            f1,
            true_positive,
            false_positive,
            false_negative,
        ) = evaluate_sample(model, data, device, decoder=decoder)
        data = data.cpu()
        sample_path, sample = load_source_sample(data)
        output_path = output_dir / sample_path.name

        if output_path.resolve() == sample_path.resolve():
            raise ValueError(
                "Refusing to overwrite the source sample. Choose a different "
                f"--output-dir than {sample_path.parent}."
            )
        if output_path.name in written_names:
            raise ValueError(
                f"Two selected samples would write {output_path.name}. "
                f"First source: {written_names[output_path.name]}, "
                f"second source: {sample_path}"
            )

        payload = build_export_payload(
            data=data,
            sample_path=sample_path,
            sample=sample,
            split=split,
            sample_index=sample_index,
            predicted_edges=predicted_edges,
            target_edges=target_edges,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            candidate_recall=sample_candidate_recall,
        )
        write_json(output_path, payload)

        written_names[output_path.name] = sample_path
        evaluated_samples += 1
        written_samples += 1

        if written_samples % 50 == 0:
            print(f"wrote {written_samples} JSON files...")

    return {
        "requested_samples": len(split_indices),
        "evaluated_samples": evaluated_samples,
        "skipped_samples": skipped_samples,
        "written_samples": written_samples,
    }


def parse_args():
    """Parse command-line arguments."""
    project_config = load_project_config_from_cli()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Hierreco run once and export per-sample JSONs "
            "with candidate, GT, and decoded predicted graph edges."
        )
    )
    add_config_argument(parser, project_config)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing model.py and checkpoint.",
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        help=(
            "Dataset root. Defaults to config paths.dataset_root, then "
            "checkpoint data_config.dataset_root."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Default: <run_dir>/best_val_loss.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run_dir>/evaluated_graph_jsons.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Dataset split to evaluate. Default: all.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, ...")
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="greedy",
        help="Path decoder used for decoded predictions. Default: greedy.",
    )
    parser.add_argument(
        "--min-candidate-recall",
        type=float,
        default=None,
        help=(
            "Skip samples whose candidate graph recall is lower than this. "
            "Default: checkpoint data_config.min_candidate_recall, usually 1.0."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N requested split samples.",
    )
    parser.add_argument("--bin-count", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-bins", type=parse_bin_spec)
    parser.add_argument("--validation-bins", type=parse_bin_spec)
    args = parser.parse_args()
    args.configured_dataset_root = project_config.paths.dataset_root
    return args


def main() -> None:
    """Load a run, evaluate decoded predictions, and write graph JSON files."""
    args = parse_args()
    load_ml_dependencies()

    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint or run_dir / "best_val_loss.pt"
    output_dir = args.output_dir or run_dir / "evaluated_graph_jsons"

    checkpoint = load_checkpoint(checkpoint_path)
    dataset = build_dataset(args, checkpoint)
    device = torch.device(args.device)
    model = instantiate_model(run_dir, checkpoint).to(device)
    data_config = dict(checkpoint.get("data_config") or {})
    min_candidate_recall = (
        args.min_candidate_recall
        if args.min_candidate_recall is not None
        else float(data_config.get("min_candidate_recall", 1.0))
    )

    summary = export_graph_jsons(
        model=model,
        dataset=dataset,
        split_name=args.split,
        device=device,
        output_dir=output_dir,
        min_candidate_recall=min_candidate_recall,
        limit=args.limit,
        decoder=args.decoder,
    )

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset: {dataset.root}")
    print(f"split: {args.split}")
    print(f"decoder: {args.decoder}")
    print(f"min_candidate_recall: {min_candidate_recall}")
    print(f"requested samples: {summary['requested_samples']}")
    print(f"evaluated samples: {summary['evaluated_samples']}")
    print(f"skipped samples: {summary['skipped_samples']}")
    print(f"wrote JSON files: {summary['written_samples']} -> {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except MissingMLDependency as error:
        raise SystemExit(f"error: {error}") from None
