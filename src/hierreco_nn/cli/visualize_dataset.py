#!/usr/bin/env python3

"""CLI runner for Hierreco dataset visualizations."""

import argparse
from pathlib import Path

from hierreco_nn.config import (
    add_config_argument,
    configure_matplotlib,
    load_project_config_from_cli,
)
from hierreco_nn.dataset import HierrecoDataset
from hierreco_nn.dataset.visualization import HierrecoVisualizer


def parse_bin_spec(value):
    """Parse a comma-separated bin list, or return ``None`` for defaults."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ()

    return tuple(int(part) for part in parts)


def parse_args():
    """Parse visualization CLI arguments."""
    project_config = load_project_config_from_cli()
    configure_matplotlib(project_config)
    parser = argparse.ArgumentParser(
        description="Build and visualize Hierreco sequence-aware candidate graphs.",
        epilog=(
            "Adaptive parameters: candidate_density controls how many local "
            "candidate edges are produced; split_sensitivity controls how easy "
            "it is to split substructures; edge_length_strictness controls how "
            "short length-limited internal edges must be. Concrete candidate "
            "parameters are calibrated from train-bin coordinate statistics."
        ),
    )
    add_config_argument(parser, project_config)
    parser.add_argument("--root", default=project_config.paths.dataset_root, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-density", default=1.0, type=float)
    parser.add_argument("--split-sensitivity", default=1.0, type=float)
    parser.add_argument("--edge-length-strictness", default=1.0, type=float)
    parser.add_argument("--triangle-min-angle-degrees", default=30.0, type=float)
    parser.add_argument("--min-sub-nodes", default=2, type=int)
    parser.add_argument("--min-nodes", default=5, type=int)
    parser.add_argument("--bin-count", default=5, type=int)
    parser.add_argument(
        "--split-seed",
        default=13,
        type=int,
        help="Seed for deterministic sample shuffling before bin assignment.",
    )
    parser.add_argument("--train-bins", type=parse_bin_spec)
    parser.add_argument("--validation-bins", type=parse_bin_spec)
    parser.add_argument("--hide-boundaries", action="store_true")
    parser.add_argument("--hide-bboxes", action="store_true")
    parser.add_argument("--require-target-coverage", action="store_true")
    return parser.parse_args()


def run_visualization(args):
    """Build the dataset and render requested visualization output."""
    dataset = HierrecoDataset(
        args.root,
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
        require_target_coverage=args.require_target_coverage,
    )
    visualizer = HierrecoVisualizer(dataset)

    render_all = args.all or args.index is None
    if render_all:
        if args.index is not None:
            raise ValueError("--index cannot be used together with --all")

        records = visualizer.render_all(
            output_dir=args.output,
            limit=args.limit,
            show_boundaries=not args.hide_boundaries,
            show_bboxes=not args.hide_bboxes,
        )

        if args.output is not None:
            bad_count = sum(1 for _, _, missing_edges in records if missing_edges)
            print(
                f"Saved {len(records)} visualizations to: {args.output} "
                f"(bad first, bad={bad_count})"
            )
        return

    selected_index, data, missing_edges = visualizer.choose_sample(args.index)
    visualizer.render(
        data,
        selected_index,
        missing_edges,
        output_path=args.output,
        show_boundaries=not args.hide_boundaries,
        show_bboxes=not args.hide_bboxes,
    )

    if missing_edges:
        print(
            f"Visualizing sample {selected_index} with missing target edges: "
            f"{missing_edges}"
        )
    else:
        print(
            f"Visualizing sample {selected_index}; candidate graph covers "
            "all target edges."
        )

    if args.output:
        print(f"Saved visualization to: {args.output}")


def main():
    """Parse CLI arguments and run dataset visualization."""
    run_visualization(parse_args())


if __name__ == "__main__":
    main()
