#!/usr/bin/env python3

"""Compute Hierreco dataset split and candidate-graph statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from hierreco_nn.config import add_config_argument, load_project_config_from_cli
from hierreco_nn.dataset import HierrecoDataset


@dataclass(frozen=True)
class SampleStats:
    """Statistics collected for one loaded PyG sample."""

    index: int
    path: str
    component_id: str
    bin_id: int
    split_name: str
    node_count: int
    candidate_count: int
    target_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    target_is_path: bool
    target_edge_count_error: int
    target_endpoint_count: int
    target_max_degree: int
    target_connected: bool
    missing_edges: tuple[tuple[int, int, str, str, str, str], ...]


def parse_bin_spec(value):
    """Parse a comma-separated bin list, or return ``None`` for defaults."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ()

    return tuple(int(part) for part in parts)


def undirected_edges(edge_index):
    """Collapse a PyG ``edge_index`` tensor into undirected edge tuples."""
    edges = set()

    for source, target in edge_index.t().tolist():
        if source != target:
            edges.add(HierrecoDataset.normalize_edge(source, target))

    return edges


def target_path_stats(node_count, target_edges):
    """Return topology checks for the GT path assumption."""
    degrees = [0 for _ in range(node_count)]
    adjacency = [[] for _ in range(node_count)]

    for source, target in target_edges:
        degrees[source] += 1
        degrees[target] += 1
        adjacency[source].append(target)
        adjacency[target].append(source)

    endpoint_count = sum(1 for degree in degrees if degree == 1)
    max_degree = max(degrees, default=0)
    edge_count_error = len(target_edges) - max(node_count - 1, 0)

    if node_count == 0:
        connected = True
    else:
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        connected = len(seen) == node_count

    is_path = (
        edge_count_error == 0
        and endpoint_count == 2
        and max_degree <= 2
        and connected
    )
    return is_path, edge_count_error, endpoint_count, max_degree, connected


def collect_sample_stats(dataset):
    """Iterate through a dataset and collect per-sample coverage statistics."""
    records = []

    for index in range(len(dataset)):
        data = dataset[index]
        candidate_edges = undirected_edges(data.edge_index)
        target_edges = undirected_edges(data.target_edge_index)
        missing_edges = sorted(target_edges - candidate_edges)
        extra_edges = candidate_edges - target_edges
        true_edges = target_edges & candidate_edges
        path_stats = target_path_stats(int(data.num_nodes), target_edges)
        blob_ids = getattr(data, "blob_ids", data.node_ids)

        records.append(
            SampleStats(
                index=index,
                path=data.sample_path,
                component_id=getattr(data, "component_id", ""),
                bin_id=int(data.bin_id),
                split_name=str(data.split_name),
                node_count=int(data.num_nodes),
                candidate_count=len(candidate_edges),
                target_count=len(target_edges),
                true_positive=len(true_edges),
                false_positive=len(extra_edges),
                false_negative=len(missing_edges),
                target_is_path=path_stats[0],
                target_edge_count_error=path_stats[1],
                target_endpoint_count=path_stats[2],
                target_max_degree=path_stats[3],
                target_connected=path_stats[4],
                missing_edges=tuple(
                    (
                        source,
                        target,
                        data.node_ids[source],
                        data.node_ids[target],
                        blob_ids[source],
                        blob_ids[target],
                    )
                    for source, target in missing_edges
                ),
            )
        )

    return records


def safe_divide(numerator, denominator):
    """Return ``numerator / denominator`` guarded against zero."""
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(true_positive, false_positive, false_negative):
    """Return precision, recall, and F1 for edge coverage counts."""
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def summarize_records(records):
    """Summarize a list of ``SampleStats`` records."""
    if not records:
        return {
            "graphs": 0,
            "nodes_total": 0,
            "nodes_min": 0,
            "nodes_median": 0.0,
            "nodes_mean": 0.0,
            "nodes_max": 0,
            "candidate_total": 0,
            "target_total": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "candidate_per_target": 0.0,
            "bad_samples": 0,
            "missing_total": 0,
            "missing_max": 0,
            "missing_hist": Counter(),
            "target_path_bad": 0,
            "target_edge_count_error_hist": Counter(),
            "target_endpoint_count_hist": Counter(),
            "target_max_degree_hist": Counter(),
            "target_disconnected": 0,
        }

    node_counts = [record.node_count for record in records]
    candidate_total = sum(record.candidate_count for record in records)
    target_total = sum(record.target_count for record in records)
    true_positive = sum(record.true_positive for record in records)
    false_positive = sum(record.false_positive for record in records)
    false_negative = sum(record.false_negative for record in records)
    precision, recall, f1 = precision_recall_f1(
        true_positive,
        false_positive,
        false_negative,
    )
    missing_counts = [
        record.false_negative
        for record in records
        if record.false_negative
    ]
    target_path_bad = [
        record
        for record in records
        if not record.target_is_path
    ]

    return {
        "graphs": len(records),
        "nodes_total": sum(node_counts),
        "nodes_min": min(node_counts),
        "nodes_median": median(node_counts),
        "nodes_mean": mean(node_counts),
        "nodes_max": max(node_counts),
        "candidate_total": candidate_total,
        "target_total": target_total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_per_target": safe_divide(candidate_total, target_total),
        "bad_samples": len(missing_counts),
        "missing_total": false_negative,
        "missing_max": max(missing_counts, default=0),
        "missing_hist": Counter(missing_counts),
        "target_path_bad": len(target_path_bad),
        "target_edge_count_error_hist": Counter(
            record.target_edge_count_error
            for record in records
        ),
        "target_endpoint_count_hist": Counter(
            record.target_endpoint_count
            for record in records
        ),
        "target_max_degree_hist": Counter(
            record.target_max_degree
            for record in records
        ),
        "target_disconnected": sum(
            1
            for record in records
            if not record.target_connected
        ),
    }


def print_profile(dataset):
    """Print the fitted train-bin geometry profile."""
    profile = dataset.geometry_profile
    print("Configuration:")
    print(f"  root: {dataset.root}")
    print(f"  samples after min_nodes filter: {len(dataset)}")
    print(f"  min_nodes: {dataset.min_nodes}")
    print(f"  bin_count: {dataset.bin_count}")
    print(f"  split_seed: {dataset.split_seed}")
    print(f"  train_bins: {dataset.train_bins}")
    print(f"  validation_bins: {dataset.validation_bins}")
    print(f"  train samples used for geometry profile: {profile.sample_count}")
    print("  profile:")
    print(
        "    sqrt_nodes "
        f"median={profile.median_sqrt_nodes:.3f}, "
        f"q75={profile.q75_sqrt_nodes:.3f}, "
        f"q90={profile.q90_sqrt_nodes:.3f}"
    )
    print(
        "    spacing_spread "
        f"median={profile.median_spacing_spread:.3f}, "
        f"q75={profile.q75_spacing_spread:.3f}, "
        f"q90={profile.q90_spacing_spread:.3f}"
    )
    print(
        "    empty_box_length_ratio "
        f"median={profile.median_empty_box_length_ratio:.3f}, "
        f"q75={profile.q75_empty_box_length_ratio:.3f}, "
        f"q90={profile.q90_empty_box_length_ratio:.3f}"
    )
    print()


def print_summary(name, summary):
    """Print a readable multi-line summary."""
    print(f"{name}:")
    print(f"  graphs: {summary['graphs']}")
    print(
        "  nodes: "
        f"total={summary['nodes_total']}, "
        f"min={summary['nodes_min']}, "
        f"median={summary['nodes_median']:.1f}, "
        f"mean={summary['nodes_mean']:.2f}, "
        f"max={summary['nodes_max']}"
    )
    print(
        "  edges undirected: "
        f"candidate={summary['candidate_total']}, "
        f"target={summary['target_total']}, "
        f"candidate/target={summary['candidate_per_target']:.4f}"
    )
    print(
        "  edges directed for PyG: "
        f"candidate={summary['candidate_total'] * 2}, "
        f"target={summary['target_total'] * 2}"
    )
    print(
        "  classification: "
        f"tp={summary['true_positive']}, "
        f"fp={summary['false_positive']}, "
        f"fn={summary['false_negative']}, "
        f"precision={summary['precision']:.6f}, "
        f"recall={summary['recall']:.6f}, "
        f"f1={summary['f1']:.6f}"
    )
    print(
        "  missing: "
        f"bad_samples={summary['bad_samples']}, "
        f"missing_total={summary['missing_total']}, "
        f"missing_max_per_sample={summary['missing_max']}, "
        f"hist={sorted(summary['missing_hist'].items())}"
    )
    print(
        "  GT path topology: "
        f"bad_samples={summary['target_path_bad']}, "
        f"edge_count_error_hist={sorted(summary['target_edge_count_error_hist'].items())}, "
        f"endpoint_count_hist={sorted(summary['target_endpoint_count_hist'].items())}, "
        f"max_degree_hist={sorted(summary['target_max_degree_hist'].items())}, "
        f"disconnected={summary['target_disconnected']}"
    )
    print()


def print_bin_table(dataset, records):
    """Print one compact row per deterministic dataset bin."""
    by_bin = defaultdict(list)
    for record in records:
        by_bin[record.bin_id].append(record)

    print("Bins:")
    print(
        "  bin  split       samples  nodes  cand  target  cand/target  "
        "bad  missing  recall    precision"
    )
    for bin_id in range(dataset.bin_count):
        bin_records = by_bin.get(bin_id, [])
        summary = summarize_records(bin_records)
        split_name = dataset.split_name_for_index(
            dataset.sample_bins.index(bin_id)
        ) if bin_records else "unused"
        print(
            f"  {bin_id:<4} {split_name:<10} "
            f"{summary['graphs']:>7} "
            f"{summary['nodes_total']:>6} "
            f"{summary['candidate_total']:>5} "
            f"{summary['target_total']:>7} "
            f"{summary['candidate_per_target']:>11.4f} "
            f"{summary['bad_samples']:>4} "
            f"{summary['missing_total']:>8} "
            f"{summary['recall']:>8.6f} "
            f"{summary['precision']:>10.6f}"
        )
    print()


def print_bad_samples(records, limit):
    """Print samples whose target edges are not fully covered."""
    bad_records = [
        record
        for record in records
        if record.false_negative
    ]
    bad_records.sort(
        key=lambda record: (
            -record.false_negative,
            record.index,
        )
    )

    if not bad_records:
        print("Bad samples: none")
        return

    shown = bad_records if limit is None else bad_records[:limit]
    print(f"Bad samples: showing {len(shown)} of {len(bad_records)}")
    for record in shown:
        missing = ", ".join(
            (
                f"{source}-{target}"
                f"({source_id}/{source_blob_id}-{target_id}/{target_blob_id})"
            )
            for (
                source,
                target,
                source_id,
                target_id,
                source_blob_id,
                target_blob_id,
            ) in record.missing_edges
        )
        print(
            f"  index={record.index} bin={record.bin_id} split={record.split_name} "
            f"missing={record.false_negative} cand={record.candidate_count} "
            f"target={record.target_count} component_id={record.component_id} "
            f"path={record.path}"
        )
        print(f"    missing_edges: {missing}")


def parse_args():
    """Parse command-line arguments."""
    project_config = load_project_config_from_cli()
    parser = argparse.ArgumentParser(
        description="Compute Hierreco dataset split and candidate-graph statistics.",
    )
    add_config_argument(parser, project_config)
    parser.add_argument(
        "--dataset-root",
        default=project_config.paths.dataset_root,
        type=Path,
    )
    parser.add_argument("--bin-count", default=5, type=int)
    parser.add_argument(
        "--split-seed",
        default=13,
        type=int,
        help="Seed for deterministic sample shuffling before bin assignment.",
    )
    parser.add_argument("--train-bins", type=parse_bin_spec)
    parser.add_argument(
        "--validation-bins",
        type=parse_bin_spec,
        help="Comma-separated validation bin ids. Default: final two bins when possible.",
    )
    parser.add_argument("--candidate-density", default=1.0, type=float)
    parser.add_argument("--split-sensitivity", default=1.0, type=float)
    parser.add_argument("--edge-length-strictness", default=1.0, type=float)
    parser.add_argument("--triangle-min-angle-degrees", default=30.0, type=float)
    parser.add_argument("--min-sub-nodes", default=2, type=int)
    parser.add_argument("--min-nodes", default=5, type=int)
    parser.add_argument(
        "--bad-limit",
        default=20,
        type=int,
        help="How many bad samples to print. Use -1 to print all, 0 to hide.",
    )
    return parser.parse_args()


def main():
    """Build the dataset and print statistics."""
    args = parse_args()
    dataset = HierrecoDataset(
        args.dataset_root,
        candidate_density=args.candidate_density,
        split_sensitivity=args.split_sensitivity,
        edge_length_strictness=args.edge_length_strictness,
        triangle_min_angle_degrees=args.triangle_min_angle_degrees,
        min_sub_nodes=args.min_sub_nodes,
        min_nodes=args.min_nodes,
        bin_count=args.bin_count,
        split_seed=args.split_seed,
        train_bins=args.train_bins,
        validation_bins=args.validation_bins,
        require_target_coverage=False,
    )
    records = collect_sample_stats(dataset)

    print_profile(dataset)
    print_summary("All data", summarize_records(records))
    print_summary(
        "Train data",
        summarize_records(
            [
                record
                for record in records
                if record.split_name == "train"
            ]
        ),
    )
    print_summary(
        "Validation data",
        summarize_records(
            [
                record
                for record in records
                if record.split_name == "validation"
            ]
        ),
    )
    unused_records = [
        record
        for record in records
        if record.split_name == "unused"
    ]
    if unused_records:
        print_summary("Unused data", summarize_records(unused_records))

    print_bin_table(dataset, records)

    if args.bad_limit != 0:
        bad_limit = None if args.bad_limit < 0 else args.bad_limit
        print_bad_samples(records, bad_limit)


if __name__ == "__main__":
    main()
