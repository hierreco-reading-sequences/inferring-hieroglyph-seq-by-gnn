#!/usr/bin/env python3

"""Export decoded TP/FP/FN edge rows for a trained Hierreco run."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from hierreco_nn.cli.visualize_run_predictions import (
    build_dataset,
    candidate_graph_recall,
    evaluate_sample,
    instantiate_model,
    load_checkpoint,
    parse_bin_spec,
)
from hierreco_nn.config import add_config_argument, load_project_config_from_cli
from hierreco_nn.path_decoder import DECODER_CHOICES

import torch


def parse_edge_types(value):
    """Parse requested model edge result types from a comma-separated string."""
    edge_types = tuple(
        part.strip().upper()
        for part in value.split(",")
        if part.strip()
    )
    allowed = {"TP", "FP", "FN"}
    invalid = sorted(set(edge_types) - allowed)

    if invalid:
        raise argparse.ArgumentTypeError(
            f"edge types must be selected from {sorted(allowed)}, got {invalid}"
        )

    if not edge_types:
        raise argparse.ArgumentTypeError("at least one edge type must be selected")

    return edge_types


def csv_value(value):
    """Format an optional scalar for CSV output."""
    return "" if value is None else value


def node_labels(data):
    """Return human-readable node labels, preferring blob IDs when available."""
    if hasattr(data, "blob_ids"):
        return list(data.blob_ids)
    if hasattr(data, "node_ids"):
        return list(data.node_ids)
    return [str(index) for index in range(int(data.num_nodes))]


def original_positions(data):
    """Return original node coordinates used only for error identification."""
    fallback = [(None, None) for _ in range(int(data.num_nodes))]
    positions = getattr(data, "original_positions", fallback)

    if torch.is_tensor(positions):
        return [
            tuple(item)
            for item in positions.detach().cpu().tolist()
        ]

    return list(positions)


def edge_rows_for_sample(
    data,
    *,
    predicted_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
    edge_types: tuple[str, ...],
):
    """Yield CSV rows for requested model result edge classes in one sample."""
    sample_name = Path(getattr(data, "sample_path", "")).name
    component_id = getattr(data, "component_id", "")
    labels = node_labels(data)
    positions = original_positions(data)

    typed_edges = {
        "TP": predicted_edges & target_edges,
        "FP": predicted_edges - target_edges,
        "FN": target_edges - predicted_edges,
    }

    for edge_type in edge_types:
        for source, target in sorted(typed_edges[edge_type]):
            source_original_x, source_original_y = positions[source]
            target_original_x, target_original_y = positions[target]
            yield {
                "sample_name": sample_name,
                "component_id": component_id,
                "node1": labels[source],
                "node1_original_x": csv_value(source_original_x),
                "node1_original_y": csv_value(source_original_y),
                "node2": labels[target],
                "node2_original_x": csv_value(target_original_x),
                "node2_original_y": csv_value(target_original_y),
                "type": edge_type,
            }


def selected_split_indices(dataset, split_name):
    """Return split/index pairs requested by the CLI."""
    if split_name == "train":
        return [("train", index) for index in dataset.train_indices]
    if split_name == "validation":
        return [("validation", index) for index in dataset.validation_indices]

    return (
        [("train", index) for index in dataset.train_indices]
        + [("validation", index) for index in dataset.validation_indices]
    )


def export_rows(
    *,
    model,
    dataset,
    split_name: str,
    device: torch.device,
    output_path: Path,
    edge_types: tuple[str, ...],
    min_candidate_recall: float,
    limit: int | None,
    decoder: str,
):
    """Evaluate samples and write requested decoded prediction errors to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_name",
        "component_id",
        "node1",
        "node1_original_x",
        "node1_original_y",
        "node2",
        "node2_original_x",
        "node2_original_y",
        "type",
    ]
    counts = Counter()
    row_count = 0
    evaluated_samples = 0
    skipped_samples = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    split_indices = selected_split_indices(dataset, split_name)
    if limit is not None:
        split_indices = split_indices[:limit]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for split, sample_index in split_indices:
            data = dataset[sample_index]
            candidate_recall = candidate_graph_recall(data)
            if candidate_recall + 1e-12 < min_candidate_recall:
                skipped_samples += 1
                continue

            (
                predicted_edges,
                target_edges,
                _precision,
                _recall,
                _f1,
                true_positive,
                false_positive,
                false_negative,
            ) = evaluate_sample(model, data, device, decoder=decoder)
            total_tp += true_positive
            total_fp += false_positive
            total_fn += false_negative
            evaluated_samples += 1

            for row in edge_rows_for_sample(
                data,
                predicted_edges=predicted_edges,
                target_edges=target_edges,
                edge_types=edge_types,
            ):
                writer.writerow(row)
                counts[row["type"]] += 1
                row_count += 1

    if total_tp + total_fp:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 1.0
    if total_tp + total_fn:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "row_count": row_count,
        "counts": counts,
        "requested_samples": len(split_indices),
        "evaluated_samples": evaluated_samples,
        "skipped_samples": skipped_samples,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args():
    """Parse command-line arguments."""
    project_config = load_project_config_from_cli()
    parser = argparse.ArgumentParser(
        description=(
            "Export model prediction edge errors to CSV for a trained Hierreco "
            "run. By default only FP and FN rows are written."
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
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <run_dir>/prediction_edge_errors.csv.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="Dataset split to evaluate. Default: all.",
    )
    parser.add_argument(
        "--edge-types",
        default=("FP", "FN"),
        type=parse_edge_types,
        help="Comma-separated model result types to export: TP, FP, FN.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, ...")
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="ilp",
        help="Path decoder used for decoded predictions. Default: ilp.",
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
        help="Evaluate only the first N requested split samples, useful for quick checks.",
    )
    parser.add_argument("--bin-count", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-bins", type=parse_bin_spec)
    parser.add_argument("--validation-bins", type=parse_bin_spec)
    args = parser.parse_args()
    args.configured_dataset_root = project_config.paths.dataset_root
    return args


def main():
    """Load a run, evaluate decoded predictions, and write the CSV export."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint or run_dir / "best_val_loss.pt"
    output_path = args.output or run_dir / "prediction_edge_errors.csv"

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

    summary = export_rows(
        model=model,
        dataset=dataset,
        split_name=args.split,
        device=device,
        output_path=output_path,
        edge_types=args.edge_types,
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
    print(f"evaluated samples: {summary['evaluated_samples']}/{summary['requested_samples']}")
    print(f"skipped samples: {summary['skipped_samples']}")
    print(f"wrote rows: {summary['row_count']} -> {output_path}")
    print(
        "rows by type: "
        + ", ".join(
            f"{edge_type}={summary['counts'][edge_type]}"
            for edge_type in args.edge_types
        )
    )
    print(
        "model decoded totals: "
        f"TP={summary['true_positive']} "
        f"FP={summary['false_positive']} "
        f"FN={summary['false_negative']} "
        f"precision={summary['precision']:.4f} "
        f"recall={summary['recall']:.4f} "
        f"f1={summary['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
