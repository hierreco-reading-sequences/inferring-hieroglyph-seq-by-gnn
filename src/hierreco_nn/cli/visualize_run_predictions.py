#!/usr/bin/env python3

"""Visualize decoded path predictions for a trained Hierreco run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hierreco_nn.config import (
    add_config_argument,
    configure_matplotlib,
    load_project_config_from_cli,
)
from hierreco_nn.dataset import HierrecoDataset
from hierreco_nn.path_decoder import DECODER_CHOICES, decode_path_edges


@dataclass(frozen=True)
class SampleResult:
    """Evaluation result for one dataset sample."""

    split: str
    sample_index: int
    f1: float
    precision: float
    recall: float
    true_positive: int
    false_positive: int
    false_negative: int
    predicted_count: int
    target_count: int
    candidate_count: int
    data: Any
    predicted_edges: set[tuple[int, int]]
    target_edges: set[tuple[int, int]]
    output_path: Path | None = None


def parse_bin_spec(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated bin list, or return ``None`` for defaults."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ()

    return tuple(int(part) for part in parts)


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

    module_name = f"hierreco_run_model_{run_dir.name}"
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
        bin_count=int(args.bin_count or config_value(data_config, "bin_count", 5)),
        split_seed=(
            args.split_seed
            if args.split_seed is not None
            else config_value(data_config, "split_seed", 13)
        ),
        train_bins=(
            args.train_bins
            if args.train_bins is not None
            else tuple(data_config["train_bins"])
            if data_config.get("train_bins") is not None
            else None
        ),
        validation_bins=(
            args.validation_bins
            if args.validation_bins is not None
            else tuple(data_config["validation_bins"])
            if data_config.get("validation_bins") is not None
            else None
        ),
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


def candidate_graph_recall(data) -> float:
    """Return candidate graph recall against target undirected edges."""
    target_edges = undirected_edges(data.target_edge_index)
    if not target_edges:
        return 1.0
    candidate_edges = undirected_edges(data.edge_index)
    return len(target_edges & candidate_edges) / len(target_edges)


def prf(
    predicted_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
) -> tuple[float, float, float, int, int, int]:
    """Return precision, recall, F1, TP, FP, FN."""
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
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, true_positive, false_positive, false_negative


def draw_prediction(
    ax,
    data,
    *,
    sample_index: int,
    split: str,
    predicted_edges: set[tuple[int, int]],
    precision: float,
    recall: float,
    f1: float,
    show_bboxes: bool,
    show_boundaries: bool,
) -> None:
    """Draw candidate, target, and decoded prediction edges on an axes."""
    import matplotlib.patches as patches

    ax.clear()
    points = data.pos.detach().cpu().numpy()
    candidate_edges = undirected_edges(data.edge_index)
    target_edges = undirected_edges(data.target_edge_index)

    for source, target in sorted(candidate_edges):
        ax.plot(
            [points[source, 0], points[target, 0]],
            [points[source, 1], points[target, 1]],
            color="0.82",
            linewidth=0.8,
            linestyle="-",
            zorder=1,
        )

    for source, target in sorted(target_edges):
        ax.plot(
            [points[source, 0], points[target, 0]],
            [points[source, 1], points[target, 1]],
            color="#ef4444",
            linewidth=2.2,
            linestyle="-",
            zorder=2,
        )

    for source, target in sorted(predicted_edges):
        color = "#16a34a" if (source, target) in target_edges else "#2563eb"
        ax.plot(
            [points[source, 0], points[target, 0]],
            [points[source, 1], points[target, 1]],
            color=color,
            linewidth=2.0,
            linestyle="--",
            zorder=3,
        )

    missed_edges = target_edges - predicted_edges
    for source, target in sorted(missed_edges):
        ax.plot(
            [points[source, 0], points[target, 0]],
            [points[source, 1], points[target, 1]],
            color="#f97316",
            linewidth=3.2,
            linestyle=":",
            zorder=4,
        )

    if show_bboxes and hasattr(data, "sub_bboxes"):
        for sub_id, bbox in enumerate(data.sub_bboxes.tolist()):
            min_x, min_y, max_x, max_y = bbox
            rect = patches.Rectangle(
                (min_x, min_y),
                max(max_x - min_x, 1e-9),
                max(max_y - min_y, 1e-9),
                fill=False,
                edgecolor="0.35",
                linewidth=0.9,
                linestyle=":",
                zorder=2.5,
            )
            ax.add_patch(rect)
            ax.text(min_x, min_y, f"sub {sub_id}", fontsize=7, color="0.35")

    ax.scatter(points[:, 0], points[:, 1], color="black", s=26, zorder=5)
    if show_boundaries and hasattr(data, "subline_boundary_mask"):
        boundary_mask = data.subline_boundary_mask.detach().cpu().numpy()
        if boundary_mask.any():
            ax.scatter(
                points[boundary_mask, 0],
                points[boundary_mask, 1],
                facecolors="none",
                edgecolors="#2563eb",
                s=92,
                linewidth=1.5,
                zorder=6,
            )

    label_ids = getattr(data, "blob_ids", data.node_ids)
    for node_index, label_id in enumerate(label_ids):
        ax.annotate(
            label_id,
            (points[node_index, 0], points[node_index, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    true_positive = len(predicted_edges & target_edges)
    false_positive = len(predicted_edges - target_edges)
    false_negative = len(target_edges - predicted_edges)
    sample_name = Path(getattr(data, "sample_path", "")).name
    component_id = getattr(data, "component_id", "")
    ax.set_title(
        f"{split} index={sample_index} f1={f1:.4f} "
        f"precision={precision:.4f} recall={recall:.4f} "
        f"tp={true_positive} fp={false_positive} fn={false_negative}\n"
        f"{sample_name} | component_id={component_id}"
    )
    ax.plot([], [], color="0.82", linewidth=1.0, label="candidate")
    ax.plot([], [], color="#ef4444", linewidth=2.2, label="GT")
    ax.plot([], [], color="#16a34a", linewidth=2.0, linestyle="--", label="decoded TP")
    ax.plot([], [], color="#2563eb", linewidth=2.0, linestyle="--", label="decoded FP")
    ax.plot([], [], color="#f97316", linewidth=3.2, linestyle=":", label="missed GT")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9", linewidth=0.8)


def render_prediction(
    data,
    *,
    sample_index: int,
    split: str,
    predicted_edges: set[tuple[int, int]],
    output_path: Path | None,
    precision: float,
    recall: float,
    f1: float,
    show_bboxes: bool,
    show_boundaries: bool,
) -> None:
    """Render one prediction to a PNG or a blocking matplotlib window."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    draw_prediction(
        ax,
        data,
        sample_index=sample_index,
        split=split,
        predicted_edges=predicted_edges,
        precision=precision,
        recall=recall,
        f1=f1,
        show_bboxes=show_bboxes,
        show_boundaries=show_boundaries,
    )
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def evaluate_sample(model, data, device: torch.device, *, decoder: str = "ilp"):
    """Return decoded prediction metrics and edge sets for one sample."""
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu()

    data = data.cpu()
    predicted_edges = decode_path_edges(
        logits,
        data.edge_index,
        node_count=int(data.num_nodes),
        decoder=decoder,
    )
    target_edges = undirected_edges(data.target_edge_index)
    precision, recall, f1, tp, fp, fn = prf(predicted_edges, target_edges)
    return predicted_edges, target_edges, precision, recall, f1, tp, fp, fn


def evaluate_split(
    *,
    model,
    dataset: HierrecoDataset,
    indices: list[int],
    split: str,
    device: torch.device,
    output_dir: Path | None,
    min_candidate_recall: float,
    render_limit: int | None,
    save_images: bool,
    show_bboxes: bool,
    show_boundaries: bool,
    decoder: str,
) -> list[SampleResult]:
    """Evaluate one dataset split and optionally save worst-first images."""
    records = []
    evaluated = []

    for sample_index in indices:
        data = dataset[sample_index]
        recall = candidate_graph_recall(data)
        if recall + 1e-12 < min_candidate_recall:
            continue

        predicted_edges, target_edges, precision, recall_value, f1, tp, fp, fn = evaluate_sample(
            model,
            data,
            device,
            decoder=decoder,
        )
        evaluated.append(
            (
                f1,
                precision,
                recall_value,
                tp,
                fp,
                fn,
                sample_index,
                data,
                predicted_edges,
                target_edges,
            )
        )

    evaluated.sort(key=lambda item: (item[0], item[1], item[2], item[6]))
    if render_limit is not None:
        evaluated = evaluated[:render_limit]

    for order, item in enumerate(evaluated, start=1):
        (
            f1,
            precision,
            recall_value,
            tp,
            fp,
            fn,
            sample_index,
            data,
            predicted_edges,
            target_edges,
        ) = item
        output_path = None
        if save_images:
            if output_dir is None:
                raise ValueError("output_dir is required when save_images=True")
            sample_name = Path(
                getattr(data, "sample_path", f"sample_{sample_index}")
            ).stem
            output_path = (
                output_dir
                / split
                / f"{order:05d}_{split}_f1_{f1:.4f}_idx_{sample_index:05d}_{sample_name}.png"
            )
            render_prediction(
                data,
                sample_index=sample_index,
                split=split,
                predicted_edges=predicted_edges,
                output_path=output_path,
                precision=precision,
                recall=recall_value,
                f1=f1,
                show_bboxes=show_bboxes,
                show_boundaries=show_boundaries,
            )
        records.append(
            SampleResult(
                split=split,
                sample_index=sample_index,
                f1=f1,
                precision=precision,
                recall=recall_value,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                predicted_count=len(predicted_edges),
                target_count=len(target_edges),
                candidate_count=len(undirected_edges(data.edge_index)),
                data=data,
                predicted_edges=predicted_edges,
                target_edges=target_edges,
                output_path=output_path,
            )
        )

    return records


def write_summary(output_dir: Path, records: list[SampleResult]) -> None:
    """Write CSV and JSON summaries for rendered samples."""
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    rows = [
        {
            "split": record.split,
            "sample_index": record.sample_index,
            "f1": record.f1,
            "precision": record.precision,
            "recall": record.recall,
            "true_positive": record.true_positive,
            "false_positive": record.false_positive,
            "false_negative": record.false_negative,
            "predicted_count": record.predicted_count,
            "target_count": record.target_count,
            "candidate_count": record.candidate_count,
            "output_path": str(record.output_path) if record.output_path else "",
        }
        for record in records
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")


def browse_predictions(
    records: list[SampleResult],
    *,
    show_bboxes: bool,
    show_boundaries: bool,
) -> None:
    """Open an interactive matplotlib browser for evaluated predictions."""
    if not records:
        print("No samples to browse.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    state = {"index": 0}
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(bottom=0.16)

    previous_ax = fig.add_axes([0.36, 0.035, 0.12, 0.055])
    next_ax = fig.add_axes([0.52, 0.035, 0.12, 0.055])
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def redraw() -> None:
        record = records[state["index"]]
        draw_prediction(
            ax,
            record.data,
            sample_index=record.sample_index,
            split=record.split,
            predicted_edges=record.predicted_edges,
            precision=record.precision,
            recall=record.recall,
            f1=record.f1,
            show_bboxes=show_bboxes,
            show_boundaries=show_boundaries,
        )
        fig.suptitle(
            f"{state['index'] + 1}/{len(records)} | "
            f"{record.split} | idx={record.sample_index} | "
            f"component_id={getattr(record.data, 'component_id', '')} | "
            f"F1={record.f1:.4f} P={record.precision:.4f} R={record.recall:.4f}",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def step(delta: int) -> None:
        state["index"] = (state["index"] + delta) % len(records)
        redraw()

    def on_previous(_event) -> None:
        step(-1)

    def on_next(_event) -> None:
        step(1)

    def on_key(event) -> None:
        if event.key in {"right", "n", " "}:
            step(1)
        elif event.key in {"left", "p", "backspace"}:
            step(-1)
        elif event.key in {"escape", "q"}:
            plt.close(fig)

    previous_button.on_clicked(on_previous)
    next_button.on_clicked(on_next)
    fig._prediction_browser_buttons = (previous_button, next_button)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Interactive browser controls: right/n/space = next, left/p = previous, q/esc = close")
    redraw()
    plt.show()


def parse_args():
    """Parse command-line arguments."""
    project_config = load_project_config_from_cli()
    configure_matplotlib(project_config)
    parser = argparse.ArgumentParser(
        description=(
            "Visualize worst decoded-path predictions for a trained Hierreco run. "
            "Samples are sorted worst-first by decoded F1, train split first and "
            "validation split second."
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
        help=(
            "Output directory used with --save. "
            "Default: <run_dir>/decoded_sample_visualizations."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save PNG files and summary instead of opening the interactive browser.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, ...")
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="ilp",
        help="Path decoder used for decoded predictions. Default: ilp.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Show/save only this many worst train samples after sorting.",
    )
    parser.add_argument(
        "--limit-validation",
        type=int,
        default=None,
        help="Show/save only this many worst validation samples after sorting.",
    )
    parser.add_argument("--min-candidate-recall", type=float, default=None)
    parser.add_argument("--bin-count", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-bins", type=parse_bin_spec)
    parser.add_argument("--validation-bins", type=parse_bin_spec)
    parser.add_argument("--hide-bboxes", action="store_true")
    parser.add_argument("--hide-boundaries", action="store_true")
    args = parser.parse_args()
    args.configured_dataset_root = project_config.paths.dataset_root
    return args


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint or run_dir / "best_val_loss.pt"
    output_dir = (
        args.output_dir or run_dir / "decoded_sample_visualizations"
        if args.save
        else None
    )

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

    train_indices = list(dataset.train_indices)
    validation_indices = list(dataset.validation_indices)

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset: {dataset.root}")
    print(f"mode: {'save' if args.save else 'interactive'}")
    if output_dir is not None:
        print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(f"decoder: {args.decoder}")
    print(f"min_candidate_recall: {min_candidate_recall}")
    print(f"train samples before recall filter: {len(train_indices)}")
    print(f"validation samples before recall filter: {len(validation_indices)}")

    train_records = evaluate_split(
        model=model,
        dataset=dataset,
        indices=train_indices,
        split="train",
        device=device,
        output_dir=output_dir,
        min_candidate_recall=min_candidate_recall,
        render_limit=args.limit_train,
        save_images=args.save,
        show_bboxes=not args.hide_bboxes,
        show_boundaries=not args.hide_boundaries,
        decoder=args.decoder,
    )
    validation_records = evaluate_split(
        model=model,
        dataset=dataset,
        indices=validation_indices,
        split="validation",
        device=device,
        output_dir=output_dir,
        min_candidate_recall=min_candidate_recall,
        render_limit=args.limit_validation,
        save_images=args.save,
        show_bboxes=not args.hide_bboxes,
        show_boundaries=not args.hide_boundaries,
        decoder=args.decoder,
    )
    records = train_records + validation_records
    if args.save:
        if output_dir is None:
            raise ValueError("output_dir is required in save mode")
        write_summary(output_dir, records)

    def print_split_summary(name: str, split_records: list[SampleResult]) -> None:
        if not split_records:
            print(f"{name}: no rendered samples")
            return
        worst = split_records[0]
        best = max(split_records, key=lambda record: record.f1)
        mean_f1 = sum(record.f1 for record in split_records) / len(split_records)
        print(
            f"{name}: samples={len(split_records)} "
            f"worst_f1={worst.f1:.4f} idx={worst.sample_index} "
            f"best_f1={best.f1:.4f} mean_f1={mean_f1:.4f}"
        )

    print_split_summary("train", train_records)
    print_split_summary("validation", validation_records)
    if args.save:
        print(f"summary: {output_dir / 'summary.csv'}")
    else:
        browse_predictions(
            records,
            show_bboxes=not args.hide_bboxes,
            show_boundaries=not args.hide_boundaries,
        )


if __name__ == "__main__":
    main()
