#!/usr/bin/env python3

"""Visualize exported Hierreco graph JSON files without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hierreco_nn.config import (
    add_config_argument,
    configure_matplotlib,
    load_project_config_from_cli,
)


def normalized_edge(source: int, target: int) -> tuple[int, int]:
    """Return a canonical undirected edge tuple."""
    return (source, target) if source < target else (target, source)


def graph_edges(record: dict[str, Any], graph_key: str) -> set[tuple[int, int]]:
    """Read an exported graph edge set."""
    edges = set()
    graph = record.get(graph_key, {})
    for edge in graph.get("edges", []):
        source = int(edge["source"])
        target = int(edge["target"])
        if source != target:
            edges.add(normalized_edge(source, target))
    return edges


def is_graph_export(record: dict[str, Any]) -> bool:
    """Return true when a JSON object looks like the exported graph schema."""
    return (
        isinstance(record.get("nodes"), list)
        and isinstance(record.get("candidate_graph"), dict)
        and isinstance(record.get("gt_graph"), dict)
        and isinstance(record.get("predicted_graph"), dict)
    )


def graph_json_paths(input_path: Path) -> list[Path]:
    """Return exported JSON paths from a file or directory."""
    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*.json")
        if not any(part.startswith(".") for part in path.relative_to(input_path).parts)
    )


def load_records(input_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load all valid exported graph records under ``input_path``."""
    records = []
    for path in graph_json_paths(input_path):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if is_graph_export(record):
            records.append((path, record))

    if not records:
        raise FileNotFoundError(f"No exported graph JSON files found in {input_path}")

    return records


def sort_records(
    records: list[tuple[Path, dict[str, Any]]],
    order: str,
) -> list[tuple[Path, dict[str, Any]]]:
    """Sort records for browsing or batch rendering."""
    if order == "name":
        return sorted(records, key=lambda item: item[0].name)

    split_rank = {"train": 0, "validation": 1, "val": 1}

    def f1_value(record: dict[str, Any]) -> float:
        f1 = record.get("metrics", {}).get("f1")
        return float("inf") if f1 is None else float(f1)

    if order == "f1":
        return sorted(records, key=lambda item: (f1_value(item[1]), item[0].name))

    return sorted(
        records,
        key=lambda item: (
            split_rank.get(item[1].get("split"), 2),
            f1_value(item[1]),
            item[0].name,
        ),
    )


def fmt_float(value: Any, digits: int = 2) -> str:
    """Format optional numeric values compactly for plot titles."""
    if value is None:
        return "?"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def is_sequence_zero(value: Any) -> bool:
    """Return true when a sequence label represents zero."""
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return str(value) == "0"


def sequence_zero_text(record: dict[str, Any]) -> str:
    """Build the title fragment for the sequence-zero original coordinates."""
    for node in record.get("nodes", []):
        if is_sequence_zero(node.get("sequence_num")):
            label = node.get("blob_id") or node.get("id") or node.get("index")
            return (
                f"seq0={label} "
                f"original=({fmt_float(node.get('x_original'))}, "
                f"{fmt_float(node.get('y_original'))})"
            )
    return "seq0 original=(?, ?)"


def points_from_nodes(nodes: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return normalized plot points from exported node records."""
    return [(float(node["x"]), float(node["y"])) for node in nodes]


def edge_points(
    points: list[tuple[float, float]],
    source: int,
    target: int,
) -> tuple[list[float], list[float]]:
    """Return x/y coordinate lists for one edge."""
    return (
        [points[source][0], points[target][0]],
        [points[source][1], points[target][1]],
    )


def boundary_node_indices(record: dict[str, Any]) -> set[int]:
    """Return all exported boundary node indices."""
    boundary_nodes = {
        int(node["index"])
        for node in record.get("nodes", [])
        if node.get("is_subline_boundary")
    }
    for substructure in record.get("substructures", []):
        boundary_nodes.update(
            int(node_index)
            for node_index in substructure.get("boundary_nodes", [])
        )
    return boundary_nodes


def draw_record(
    ax,
    record: dict[str, Any],
    *,
    show_bboxes: bool,
    show_boundaries: bool,
    show_labels: bool,
) -> None:
    """Draw one exported graph record on a Matplotlib axes."""
    import matplotlib.patches as patches

    ax.clear()
    nodes = record["nodes"]
    points = points_from_nodes(nodes)
    candidate_edges = graph_edges(record, "candidate_graph")
    target_edges = graph_edges(record, "gt_graph")
    predicted_edges = graph_edges(record, "predicted_graph")

    for source, target in sorted(candidate_edges):
        ax.plot(
            *edge_points(points, source, target),
            color="0.82",
            linewidth=0.8,
            linestyle="-",
            zorder=1,
        )

    for source, target in sorted(target_edges):
        ax.plot(
            *edge_points(points, source, target),
            color="#ef4444",
            linewidth=2.2,
            linestyle="-",
            zorder=2,
        )

    for source, target in sorted(predicted_edges):
        color = "#16a34a" if (source, target) in target_edges else "#2563eb"
        ax.plot(
            *edge_points(points, source, target),
            color=color,
            linewidth=2.0,
            linestyle="--",
            zorder=3,
        )

    missed_edges = target_edges - predicted_edges
    for source, target in sorted(missed_edges):
        ax.plot(
            *edge_points(points, source, target),
            color="#f97316",
            linewidth=3.2,
            linestyle=":",
            zorder=4,
        )

    if show_bboxes:
        for substructure in record.get("substructures", []):
            bbox = substructure.get("bbox", [])
            if len(bbox) != 4:
                continue
            min_x, min_y, max_x, max_y = [float(value) for value in bbox]
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
            ax.text(
                min_x,
                min_y,
                f"sub {substructure.get('sub_id', '')}",
                fontsize=7,
                color="0.35",
            )

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.scatter(xs, ys, color="black", s=26, zorder=5)

    if show_boundaries:
        boundary_nodes = sorted(boundary_node_indices(record))
        if boundary_nodes:
            ax.scatter(
                [points[index][0] for index in boundary_nodes],
                [points[index][1] for index in boundary_nodes],
                facecolors="none",
                edgecolors="#2563eb",
                s=92,
                linewidth=1.5,
                zorder=6,
            )

    if show_labels:
        for node in nodes:
            index = int(node["index"])
            label = node.get("blob_id") or node.get("id") or index
            seq = node.get("sequence_num")
            text = f"{label} [{seq}]" if seq is not None else str(label)
            ax.annotate(
                text,
                points[index],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    metrics = record.get("metrics", {})
    ax.set_title(
        f"{record.get('split', '')} index={record.get('sample_index', '')} "
        f"f1={fmt_float(metrics.get('f1'), 4)} "
        f"precision={fmt_float(metrics.get('precision'), 4)} "
        f"recall={fmt_float(metrics.get('recall'), 4)} "
        f"tp={metrics.get('true_positive', '?')} "
        f"fp={metrics.get('false_positive', '?')} "
        f"fn={metrics.get('false_negative', '?')}\n"
        f"{record.get('sample_name', '')} | "
        f"component_id={record.get('component_id', '')} | "
        f"{sequence_zero_text(record)}"
    )
    ax.plot([], [], color="0.82", linewidth=1.0, label="candidate")
    ax.plot([], [], color="#ef4444", linewidth=2.2, label="GT")
    ax.plot([], [], color="#16a34a", linewidth=2.0, linestyle="--", label="decoded TP")
    ax.plot([], [], color="#2563eb", linewidth=2.0, linestyle="--", label="decoded FP")
    ax.plot([], [], color="#f97316", linewidth=3.2, linestyle=":", label="missed GT")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9", linewidth=0.8)


def render_record(
    record: dict[str, Any],
    *,
    output_path: Path | None,
    show_bboxes: bool,
    show_boundaries: bool,
    show_labels: bool,
) -> None:
    """Render one record to a PNG or a blocking Matplotlib window."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    draw_record(
        ax,
        record,
        show_bboxes=show_bboxes,
        show_boundaries=show_boundaries,
        show_labels=show_labels,
    )
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def visualization_filename(order_index: int, source_path: Path, record: dict[str, Any]) -> str:
    """Build a sortable PNG filename for one exported graph JSON."""
    metrics = record.get("metrics", {})
    split = record.get("split", "sample")
    f1 = fmt_float(metrics.get("f1"), 4)
    return f"{order_index:05d}_{split}_f1_{f1}_{source_path.stem}.png"


def render_all(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    output_dir: Path,
    limit: int | None,
    show_bboxes: bool,
    show_boundaries: bool,
    show_labels: bool,
) -> None:
    """Render all selected records to PNG files."""
    selected = records[:limit] if limit is not None else records
    output_dir.mkdir(parents=True, exist_ok=True)

    for order_index, (source_path, record) in enumerate(selected, start=1):
        output_path = output_dir / visualization_filename(order_index, source_path, record)
        render_record(
            record,
            output_path=output_path,
            show_bboxes=show_bboxes,
            show_boundaries=show_boundaries,
            show_labels=show_labels,
        )
        if order_index % 50 == 0:
            print(f"rendered {order_index} PNG files...")

    print(f"saved {len(selected)} visualizations to: {output_dir}")


def browse_records(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    start_index: int,
    show_bboxes: bool,
    show_boundaries: bool,
    show_labels: bool,
) -> None:
    """Open an interactive browser for exported graph records."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    state = {"index": start_index}
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(bottom=0.16)

    previous_ax = fig.add_axes([0.36, 0.035, 0.12, 0.055])
    next_ax = fig.add_axes([0.52, 0.035, 0.12, 0.055])
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def redraw() -> None:
        source_path, record = records[state["index"]]
        draw_record(
            ax,
            record,
            show_bboxes=show_bboxes,
            show_boundaries=show_boundaries,
            show_labels=show_labels,
        )
        fig.suptitle(
            f"{state['index'] + 1}/{len(records)} | {source_path.name} | "
            f"component_id={record.get('component_id', '')} | "
            f"{sequence_zero_text(record)}",
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
    fig._evaluated_graph_browser_buttons = (previous_button, next_button)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Interactive browser controls: right/n/space = next, left/p = previous, q/esc = close")
    redraw()
    plt.show()


def default_output_dir(input_path: Path) -> Path:
    """Return the default directory for batch-rendered PNG files."""
    if input_path.is_dir():
        return input_path / "visualizations"
    return input_path.parent / "visualizations"


def parse_args():
    """Parse command-line arguments."""
    project_config = load_project_config_from_cli()
    configure_matplotlib(project_config)
    parser = argparse.ArgumentParser(
        description=(
            "Visualize JSONs exported by export_evaluated_graph_jsons.py "
            "without loading the neural network."
        )
    )
    add_config_argument(parser, project_config)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Exported graph JSON file or directory.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Open one concrete exported graph JSON file.",
    )
    parser.add_argument("--index", type=int, default=None, help="Record index after sorting.")
    parser.add_argument(
        "--order",
        choices=("split-f1", "f1", "name"),
        default="split-f1",
        help="Browsing order. Default: split-f1.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all selected JSON records to PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG directory for --all. Default: <input>/visualizations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path for a single selected record.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit records for --all.")
    parser.add_argument("--hide-bboxes", action="store_true")
    parser.add_argument("--hide-boundaries", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    args = parser.parse_args()

    if args.input is None and args.json_path is None:
        parser.error("one of input or --json is required")
    if args.input is not None and args.json_path is not None:
        parser.error("use either positional input or --json, not both")
    if args.json_path is not None and not args.json_path.is_file():
        parser.error("--json must point to one exported graph JSON file")

    args.input_path = args.json_path or args.input
    return args


def main() -> None:
    """Load exported graph JSON files and visualize them offline."""
    args = parse_args()
    records = sort_records(load_records(args.input_path), args.order)

    if args.index is not None and not (0 <= args.index < len(records)):
        raise IndexError(f"--index must be in 0..{len(records) - 1}, got {args.index}")

    show_bboxes = not args.hide_bboxes
    show_boundaries = not args.hide_boundaries
    show_labels = not args.hide_labels

    if args.all:
        output_dir = args.output_dir or default_output_dir(args.input_path)
        render_all(
            records,
            output_dir=output_dir,
            limit=args.limit,
            show_bboxes=show_bboxes,
            show_boundaries=show_boundaries,
            show_labels=show_labels,
        )
        return

    if args.output is not None:
        selected_index = args.index if args.index is not None else 0
        _, record = records[selected_index]
        render_record(
            record,
            output_path=args.output,
            show_bboxes=show_bboxes,
            show_boundaries=show_boundaries,
            show_labels=show_labels,
        )
        print(f"saved visualization to: {args.output}")
        return

    browse_records(
        records,
        start_index=args.index or 0,
        show_bboxes=show_bboxes,
        show_boundaries=show_boundaries,
        show_labels=show_labels,
    )


if __name__ == "__main__":
    main()
