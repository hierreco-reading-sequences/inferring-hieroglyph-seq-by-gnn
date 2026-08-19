#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from .config import (
    add_config_argument,
    load_project_config,
    load_project_config_from_cli,
)
from .dataset import HierrecoDataset
from .losses import EdgeClassificationLoss, compute_pos_weight, undirected_edge_values
from .logging_utils import tee_std_streams
from .model import EdgeNodeGNN
from .path_decoder import (
    DECODER_CHOICES,
    decode_path_edges_with_diagnostics,
    validate_decoder,
)


LOADER_EXCLUDE_KEYS = [
    "pos",
    "target_edge_index",
    "node_ids",
    "blob_ids",
    "original_positions",
    "component_id",
    "graph_type",
    "sample_path",
    "sample_index",
    "bin_id",
    "split_name",
    "sub_id",
    "subline_boundary_mask",
    "sub_bboxes",
    "sub_boundary_nodes",
    "x_diagonal_edge_index",
]


def default_cache_dir() -> Path:
    """Return the configured dataset cache directory."""

    return load_project_config().paths.cache_dir


def default_run_file(name: str) -> Path:
    """Return a file path under the configured default run directory."""

    return load_project_config().paths.runs_root / "run_0" / name


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading, splitting, and batching."""

    dataset_root: Path
    batch_size: int = 8
    bin_count: int = 5
    split_seed: int | None = 13
    train_bins: tuple[int, ...] | None = None
    validation_bins: tuple[int, ...] | None = None
    min_candidate_recall: float = 1.0
    candidate_density: float = 1.0
    split_sensitivity: float = 1.0
    edge_length_strictness: float = 1.0
    triangle_min_angle_degrees: float = 30.0
    min_sub_nodes: int = 2
    min_nodes: int = 5
    num_workers: int = 0
    cache_dir: Path | None = field(default_factory=default_cache_dir)
    rebuild_cache: bool = False
    preload_cache: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the edge classifier architecture."""

    hidden_dim: int = 48
    layers: int = 2
    attention_heads: int = 4
    gnn_branch: str = "gine"
    edge_competition_layers: int = 0
    incident_selector: bool = False
    dropout: float = 0.15


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for optimization and reporting."""

    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    threshold: float = 0.5
    seed: int = 13
    device: str = "mps"
    run_dir: Path | None = None
    checkpoint_path: Path = field(
        default_factory=lambda: default_run_file("best_val_loss.pt")
    )
    resume_from: Path | None = None
    metrics_json: Path | None = None
    log_file: Path | None = field(default_factory=lambda: default_run_file("training.log"))
    log_every: int = 1
    live_plot: bool = True
    plot_html: Path = field(default_factory=lambda: default_run_file("training_plot.html"))
    plot_refresh_ms: int = 2000
    use_pos_weight: bool = True
    max_pos_weight: float | None = None
    bce_loss_weight: float = 1.0
    focal_gamma: float = 0.0
    label_smoothing: float = 0.0
    degree_loss_weight: float = 0.02
    edge_count_loss_weight: float = 0.01
    endpoint_loss_weight: float = 0.0
    incident_ranking_loss_weight: float = 0.0
    symmetry_loss_weight: float = 0.0
    endpoint_sigma: float = 0.35
    train_decoded_metrics: bool = False
    decoder: str = "ilp"
    shapley_enabled: bool = True
    shapley_mode: str = "exact"
    shapley_permutations: int = 16
    shapley_subset_batch_size: int = 128
    shapley_max_train_samples: int | None = None
    shapley_max_validation_samples: int | None = None
    shapley_seed: int = 13
    shapley_output: Path | None = None


class PlotlyMetricPlot:
    """Live Plotly HTML plot for train/validation metrics."""

    metrics = (
        "loss",
        "f1",
        "decoded_f1",
        "recall",
        "precision",
        "quadrat_f1",
        "quadrat_decoded_f1",
        "quadrat_recall",
        "quadrat_precision",
    )
    metric_labels = {
        "loss": "Loss",
        "f1": "F1",
        "decoded_f1": "Decoded F1",
        "recall": "Recall",
        "precision": "Precision",
        "quadrat_f1": "Quadrat F1",
        "quadrat_decoded_f1": "Quadrat Decoded F1",
        "quadrat_recall": "Quadrat Recall",
        "quadrat_precision": "Quadrat Precision",
    }

    def __init__(
        self,
        *,
        enabled: bool,
        html_path: Path,
        refresh_ms: int,
        initial_history: list[dict[str, Any]] | None = None,
    ):
        self.enabled = bool(enabled)
        self.html_path = Path(html_path)
        self.refresh_ms = int(refresh_ms)
        self.history: list[dict[str, Any]] = list(initial_history or [])
        self.available = False
        self.plotly_source = "cdn"

        if self.enabled:
            self._setup()

    def _setup(self) -> None:
        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        self.available = True
        self._write_html(auto_refresh=True)
        print(f"plotly live plot: {self.html_path}")

    def update(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        validation_metrics: dict[str, float],
    ) -> None:
        record = {
            "epoch": int(epoch),
            "train": {
                metric: float(train_metrics[metric])
                for metric in self.metrics
            },
            "validation": {
                metric: float(validation_metrics[metric])
                for metric in self.metrics
            },
        }
        self.history.append(record)

        if self.available:
            self._write_html(auto_refresh=True)

    def finish(self) -> None:
        if not self.available:
            return
        self._write_html(auto_refresh=False)
        print(f"plotly final plot: {self.html_path}")

    def _write_html(self, *, auto_refresh: bool) -> None:
        html = self._make_html(auto_refresh=auto_refresh)
        tmp_path = self.html_path.with_suffix(self.html_path.suffix + ".tmp")
        tmp_path.write_text(html, encoding="utf-8")
        tmp_path.replace(self.html_path)

    def _make_html(self, *, auto_refresh: bool) -> str:
        payload = {
            "history": self.history,
            "metrics": list(self.metrics),
            "metricLabels": self.metric_labels,
            "refreshMs": self.refresh_ms,
            "autoRefresh": bool(auto_refresh),
            "updatedAt": datetime.now().isoformat(timespec="seconds"),
        }
        payload_json = json.dumps(json_ready(payload), ensure_ascii=False)
        plotly_script = self._plotly_script()
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hierreco Training Metrics</title>
  {plotly_script}
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fb;
      color: #1f2937;
    }}
    header {{
      padding: 16px 20px 8px;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 650;
    }}
    .meta {{
      color: #6b7280;
      font-size: 13px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 16px;
      padding: 8px 20px 20px;
    }}
    #chart {{
      min-height: 620px;
      background: white;
      border: 1px solid #e5e7eb;
    }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      padding: 14px;
      align-self: start;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 14px;
      font-weight: 650;
    }}
    label {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
      font-size: 14px;
    }}
    input[type="checkbox"] {{
      width: 16px;
      height: 16px;
    }}
    .hint {{
      margin-top: 14px;
      color: #6b7280;
      font-size: 12px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Hierreco Training Metrics</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main class="layout">
    <div id="chart"></div>
    <aside class="panel">
      <h2>Visible Metrics</h2>
      <div id="metric-controls"></div>
      <div class="hint">
        Training curves are solid. Validation curves are dashed. Hover a point
        to inspect epoch, metric, split, and value.
      </div>
    </aside>
  </main>
  <script>
    const payload = {payload_json};
    const colors = {{
      loss: "#2563eb",
      f1: "#16a34a",
      decoded_f1: "#7c3aed",
      recall: "#f97316",
      precision: "#dc2626",
      quadrat_f1: "#0891b2",
      quadrat_decoded_f1: "#9333ea",
      quadrat_recall: "#ea580c",
      quadrat_precision: "#be123c"
    }};
    const storagePrefix = "hierreco-training-metric-";

    function metricVisible(metric) {{
      const saved = window.localStorage.getItem(storagePrefix + metric);
      return saved === null ? true : saved === "true";
    }}

    function setMetricVisible(metric, visible) {{
      window.localStorage.setItem(storagePrefix + metric, String(visible));
    }}

    function valuesFor(split, metric) {{
      return payload.history.map(row => row[split][metric]);
    }}

    function makeTraces() {{
      const epochs = payload.history.map(row => row.epoch);
      const traces = [];
      for (const metric of payload.metrics) {{
        const visible = metricVisible(metric) ? true : "legendonly";
        traces.push({{
          x: epochs,
          y: valuesFor("train", metric),
          mode: "lines+markers",
          name: `train ${{metric}}`,
          legendgroup: metric,
          metric,
          split: "train",
          visible,
          line: {{color: colors[metric], width: 2.5, dash: "solid"}},
          marker: {{size: 6}},
          hovertemplate:
            "train " + metric +
            "<br>epoch=%{{x}}" +
            "<br>value=%{{y:.6f}}" +
            "<extra></extra>"
        }});
        traces.push({{
          x: epochs,
          y: valuesFor("validation", metric),
          mode: "lines+markers",
          name: `validation ${{metric}}`,
          legendgroup: metric,
          metric,
          split: "validation",
          visible,
          line: {{color: colors[metric], width: 2.5, dash: "dash"}},
          marker: {{size: 6, symbol: "circle-open"}},
          hovertemplate:
            "validation " + metric +
            "<br>epoch=%{{x}}" +
            "<br>value=%{{y:.6f}}" +
            "<extra></extra>"
        }});
      }}
      return traces;
    }}

    function makeControls() {{
      const container = document.getElementById("metric-controls");
      for (const metric of payload.metrics) {{
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = metricVisible(metric);
        input.addEventListener("change", () => {{
          setMetricVisible(metric, input.checked);
          const update = {{visible: input.checked ? true : "legendonly"}};
          const indices = [];
          chart.data.forEach((trace, index) => {{
            if (trace.metric === metric) {{
              indices.push(index);
            }}
          }});
          Plotly.restyle("chart", update, indices);
        }});
        const text = document.createElement("span");
        text.textContent = payload.metricLabels[metric] || metric;
        label.appendChild(input);
        label.appendChild(text);
        container.appendChild(label);
      }}
    }}

    const chart = document.getElementById("chart");
    document.getElementById("meta").textContent =
      `epochs=${{payload.history.length}} | updated=${{payload.updatedAt}}`;

    Plotly.newPlot(chart, makeTraces(), {{
      template: "plotly_white",
      hovermode: "closest",
      margin: {{l: 60, r: 24, t: 24, b: 55}},
      xaxis: {{title: "Epoch", rangemode: "tozero", dtick: 1}},
      yaxis: {{title: "Metric value"}},
      legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0}}
    }}, {{
      responsive: true,
      displaylogo: false
    }});
    makeControls();

    if (payload.autoRefresh) {{
      window.setTimeout(() => window.location.reload(), payload.refreshMs);
    }}
  </script>
</body>
</html>
"""

    def _plotly_script(self) -> str:
        try:
            from plotly.offline.offline import get_plotlyjs
        except Exception:
            self.plotly_source = "cdn"
            return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

        self.plotly_source = "embedded"
        return f"<script>{get_plotlyjs()}</script>"


EPOCH_LINE_RE = re.compile(r"\bepoch=(?P<epoch>\d+)\b(?P<body>.*)")
METRIC_TOKEN_RE = re.compile(
    r"\b(?P<prefix>train|val)_(?P<metric>loss|f1|decoded_f1|recall|precision|"
    r"quadrat_f1|quadrat_decoded_f1|quadrat_recall|quadrat_precision)="
    r"(?P<value>[-+]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def load_epoch_history_from_log(
    log_file: Path | None,
    *,
    max_epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct Plotly epoch history from terminal-style training logs."""
    if log_file is None or not log_file.exists():
        return []

    records_by_epoch: dict[int, dict[str, Any]] = {}
    with log_file.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line_match = EPOCH_LINE_RE.search(line)
            if line_match is None:
                continue

            epoch = int(line_match.group("epoch"))
            if max_epoch is not None and epoch > max_epoch:
                continue

            record = {"epoch": epoch, "train": {}, "validation": {}}
            for metric_match in METRIC_TOKEN_RE.finditer(line_match.group("body")):
                split = (
                    "train"
                    if metric_match.group("prefix") == "train"
                    else "validation"
                )
                metric = metric_match.group("metric")
                record[split][metric] = float(metric_match.group("value"))

            base_metrics = ("loss", "f1", "decoded_f1", "recall", "precision")
            if all(
                metric in record[split]
                for split in ("train", "validation")
                for metric in base_metrics
            ):
                for split in ("train", "validation"):
                    for metric in PlotlyMetricPlot.metrics:
                        record[split].setdefault(metric, float("nan"))
                records_by_epoch[epoch] = record

    return [
        records_by_epoch[epoch]
        for epoch in sorted(records_by_epoch)
    ]


def build_dataset(config: DataConfig) -> HierrecoDataset:
    """Load the filtered Hierreco dataset from JSON sample files."""
    dataset = HierrecoDataset(
        config.dataset_root,
        candidate_density=config.candidate_density,
        split_sensitivity=config.split_sensitivity,
        edge_length_strictness=config.edge_length_strictness,
        triangle_min_angle_degrees=config.triangle_min_angle_degrees,
        min_sub_nodes=config.min_sub_nodes,
        min_nodes=config.min_nodes,
        bin_count=config.bin_count,
        split_seed=config.split_seed,
        train_bins=config.train_bins,
        validation_bins=config.validation_bins,
        require_target_coverage=False,
    )
    if config.cache_dir is None:
        print("dataset cache: disabled")
        return dataset

    return CachedHierrecoDataset.from_source(
        dataset,
        config=config,
        cache_root=config.cache_dir,
        rebuild=config.rebuild_cache,
        preload=config.preload_cache,
    )


def load_cached_data(path: Path):
    """Load one cached PyG Data object across PyTorch default variants."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cache_config_payload(config: DataConfig) -> dict[str, Any]:
    """Return the dataset-construction fields that define cached Data objects."""
    return {
        "schema_version": 7,
        "dataset_root": str(config.dataset_root.resolve()),
        "bin_count": config.bin_count,
        "split_seed": config.split_seed,
        "train_bins": config.train_bins,
        "validation_bins": config.validation_bins,
        "candidate_density": config.candidate_density,
        "split_sensitivity": config.split_sensitivity,
        "edge_length_strictness": config.edge_length_strictness,
        "triangle_min_angle_degrees": config.triangle_min_angle_degrees,
        "min_sub_nodes": config.min_sub_nodes,
        "min_nodes": config.min_nodes,
    }


def cache_key_for_config(config: DataConfig) -> str:
    """Return a stable short hash for a dataset cache configuration."""
    payload = cache_config_payload(config)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class CachedHierrecoDataset(Dataset):
    """Dataset wrapper backed by precomputed PyG Data files."""

    def __init__(
        self,
        source: HierrecoDataset,
        *,
        cache_dir: Path,
        manifest: dict[str, Any],
        preload: bool,
    ):
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.preload = bool(preload)
        self.data_dir = self.cache_dir / "samples"
        self.data_paths = [
            self.data_dir / file_name
            for file_name in self.manifest["files"]
        ]
        self._preloaded_data = (
            [load_cached_data(path) for path in self.data_paths]
            if self.preload
            else None
        )

        self.root = source.root
        self.min_nodes = source.min_nodes
        self.bin_count = source.bin_count
        self.split_seed = source.split_seed
        self.train_bins = source.train_bins
        self.validation_bins = source.validation_bins
        self.sample_bins = source.sample_bins
        self.train_indices = source.train_indices
        self.validation_indices = source.validation_indices
        self.split_indices = source.split_indices
        self.geometry_profile = source.geometry_profile
        self.sample_paths = source.sample_paths

    @classmethod
    def from_source(
        cls,
        source: HierrecoDataset,
        *,
        config: DataConfig,
        cache_root: Path,
        rebuild: bool,
        preload: bool,
    ) -> "CachedHierrecoDataset":
        cache_key = cache_key_for_config(config)
        cache_dir = Path(cache_root) / cache_key
        manifest_path = cache_dir / "manifest.json"
        expected_payload = cache_config_payload(config)
        expected_files = [
            f"data_{index:06d}.pt"
            for index in range(len(source))
        ]

        manifest = None
        if not rebuild and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = None

        cache_valid = (
            manifest is not None
            and manifest.get("config") == expected_payload
            and manifest.get("files") == expected_files
            and all((cache_dir / "samples" / file_name).exists() for file_name in expected_files)
        )

        if not cache_valid:
            manifest = build_dataset_cache(
                source,
                cache_dir=cache_dir,
                config_payload=expected_payload,
                files=expected_files,
            )
        else:
            print(
                "dataset cache: hit "
                f"{cache_dir} ({len(expected_files)} samples)"
            )

        dataset = cls(
            source,
            cache_dir=cache_dir,
            manifest=manifest,
            preload=preload,
        )
        if preload:
            print(f"dataset cache: preloaded {len(dataset)} samples in memory")
        return dataset

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        if self._preloaded_data is not None:
            return self._preloaded_data[index]
        return load_cached_data(self.data_paths[index])

    @staticmethod
    def normalize_edge(source, target):
        return HierrecoDataset.normalize_edge(source, target)

    def split_name_for_index(self, index):
        return self.source.split_name_for_index(index)


def build_dataset_cache(
    source: HierrecoDataset,
    *,
    cache_dir: Path,
    config_payload: dict[str, Any],
    files: list[str],
) -> dict[str, Any]:
    """Materialize every source sample as a cached PyG Data object."""
    data_dir = cache_dir / "samples"
    tmp_dir = cache_dir / "samples.tmp"
    tmp_manifest = cache_dir / "manifest.json.tmp"
    manifest_path = cache_dir / "manifest.json"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset cache: building {cache_dir} ({len(source)} samples)")
    for index, file_name in enumerate(files):
        data = source[index]
        torch.save(data, tmp_dir / file_name)
        if (index + 1) % 50 == 0 or index + 1 == len(source):
            print(f"  cached {index + 1}/{len(source)} samples")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    tmp_dir.replace(data_dir)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config_payload,
        "sample_count": len(source),
        "files": files,
        "sample_paths": [str(path) for path in source.sample_paths],
    }
    tmp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_manifest.replace(manifest_path)
    print(f"dataset cache: ready {cache_dir}")
    return manifest


def undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Collapse a directed PyG edge index into canonical undirected edges."""
    edges: set[tuple[int, int]] = set()

    for source, target in edge_index.t().tolist():
        if source != target:
            edges.add(HierrecoDataset.normalize_edge(int(source), int(target)))

    return edges


def candidate_graph_recall(data) -> tuple[float, int, int]:
    """Return candidate graph recall, missing-target count, and target count."""
    candidate_edges = undirected_edges(data.edge_index)
    target_edges = undirected_edges(data.target_edge_index)

    if not target_edges:
        return 1.0, 0, 0

    missing_edges = target_edges - candidate_edges
    true_positive = len(target_edges) - len(missing_edges)
    return true_positive / len(target_edges), len(missing_edges), len(target_edges)


def filter_indices_by_candidate_recall(
    dataset: HierrecoDataset,
    indices: tuple[int, ...],
    *,
    split_name: str,
    min_recall: float,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Keep only split samples whose candidate graph reaches ``min_recall``."""
    kept_indices: list[int] = []
    rejected: list[dict[str, Any]] = []
    tolerance = 1e-12

    for index in indices:
        data = dataset[index]
        recall, missing_count, target_count = candidate_graph_recall(data)

        if recall + tolerance >= min_recall:
            kept_indices.append(index)
            continue

        rejected.append(
            {
                "index": int(index),
                "bin_id": int(getattr(data, "bin_id", -1)),
                "recall": float(recall),
                "missing_edges": int(missing_count),
                "target_edges": int(target_count),
                "sample_path": str(getattr(data, "sample_path", "")),
            }
        )

    summary = {
        "split": split_name,
        "min_recall": float(min_recall),
        "before": len(indices),
        "after": len(kept_indices),
        "rejected": len(rejected),
        "retained_fraction": safe_divide(len(kept_indices), len(indices)),
        "rejected_fraction": safe_divide(len(rejected), len(indices)),
        "rejected_samples": rejected,
    }

    if not kept_indices:
        raise ValueError(
            f"No {split_name} samples remain after candidate-recall filtering "
            f"with min_recall={min_recall}"
        )

    return tuple(kept_indices), summary


def split_dataset(dataset: HierrecoDataset, config: DataConfig):
    """Return train/validation subsets from dataset bins and recall filtering."""
    if not dataset.validation_indices:
        raise ValueError("validation_bins must select at least one sample")

    train_indices, train_filter = filter_indices_by_candidate_recall(
        dataset,
        dataset.train_indices,
        split_name="train",
        min_recall=config.min_candidate_recall,
    )
    validation_indices, validation_filter = filter_indices_by_candidate_recall(
        dataset,
        dataset.validation_indices,
        split_name="validation",
        min_recall=config.min_candidate_recall,
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
        {
            "train": train_filter,
            "validation": validation_filter,
        },
    )


def build_loaders(train_dataset, validation_dataset, config: DataConfig):
    """Create PyTorch Geometric dataloaders for graph batches."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        exclude_keys=LOADER_EXCLUDE_KEYS,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        exclude_keys=LOADER_EXCLUDE_KEYS,
    )
    return train_loader, validation_loader


def summarize_dataset(dataset):
    """Collect simple graph and edge-label statistics over a dataset."""
    node_counts = []
    edge_counts = []
    positive_edges = []

    for data in dataset:
        node_counts.append(data.num_nodes)
        edge_counts.append(data.edge_index.size(1))
        positive_edges.append(int(data.edge_y.sum().item()))

    total_edges = sum(edge_counts)
    total_positive = sum(positive_edges)

    return {
        "graphs": len(dataset),
        "nodes_min": min(node_counts),
        "nodes_max": max(node_counts),
        "nodes_total": sum(node_counts),
        "candidate_edges_total": total_edges,
        "positive_edges_total": total_positive,
        "negative_edges_total": total_edges - total_positive,
        "positive_edge_ratio": total_positive / total_edges if total_edges else 0.0,
    }


def print_summary(name: str, summary: dict[str, Any]) -> None:
    """Print a compact human-readable dataset summary."""
    print(f"{name}:")
    print(f"  graphs: {summary['graphs']}")
    print(f"  nodes: {summary['nodes_total']} total")
    print(f"  nodes per graph: {summary['nodes_min']}..{summary['nodes_max']}")
    print(f"  candidate edges: {summary['candidate_edges_total']}")
    print(f"  positive edges: {summary['positive_edges_total']}")
    print(f"  negative edges: {summary['negative_edges_total']}")
    print(f"  positive edge ratio: {summary['positive_edge_ratio']:.4f}")


def print_recall_filter_summary(summary: dict[str, Any], *, limit: int = 10) -> None:
    """Print how many split samples survived the candidate-recall filter."""
    print(
        f"{summary['split']} candidate-recall filter: "
        f"kept {summary['after']}/{summary['before']} "
        f"({100.0 * summary['retained_fraction']:.2f}%, "
        f"rejected={100.0 * summary['rejected_fraction']:.2f}%, "
        f"min_recall={summary['min_recall']:.6f})"
    )
    if summary["rejected"] == 0:
        return

    print(f"  rejected samples: {summary['rejected']}")
    for sample in summary["rejected_samples"][:limit]:
        print(
            "  "
            f"index={sample['index']} bin={sample['bin_id']} "
            f"recall={sample['recall']:.6f} "
            f"missing={sample['missing_edges']}/{sample['target_edges']} "
            f"path={sample['sample_path']}"
        )
    if summary["rejected"] > limit:
        print(f"  ... {summary['rejected'] - limit} more rejected samples")


def inspect_first_batch(loader) -> None:
    """Print tensor shapes from one batched PyG graph batch."""
    batch = next(iter(loader))
    print("first train batch:")
    print(f"  x: {tuple(batch.x.shape)}")
    print(f"  edge_index: {tuple(batch.edge_index.shape)}")
    print(f"  edge_attr: {tuple(batch.edge_attr.shape)}")
    print(f"  edge_y: {tuple(batch.edge_y.shape)}")
    if hasattr(batch, "quadrat"):
        print(f"  quadrat: {tuple(batch.quadrat.shape)}")
    print(f"  batch vector: {tuple(batch.batch.shape)}")


def count_parameters(module: torch.nn.Module | None) -> int:
    """Return the number of trainable parameters in one module."""
    if module is None:
        return 0

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def model_parameter_summary(model: EdgeNodeGNN) -> dict[str, int]:
    """Return trainable parameter counts for the model and its main parts."""
    return {
        "total": count_parameters(model),
        "node_encoder": count_parameters(model.node_encoder),
        "edge_encoder": count_parameters(model.edge_encoder),
        "gine_branch": count_parameters(model.gine_branch),
        "gatv2_branch": count_parameters(model.gatv2_branch),
        "edge_context_encoder": count_parameters(model.edge_context_encoder),
        "edge_head": count_parameters(model.edge_head),
    }


def print_model_parameter_summary(summary: dict[str, int]) -> None:
    """Print trainable parameter counts in a compact form."""
    print("model parameters:")
    for name, count in summary.items():
        print(f"  {name}: {count:,}")


def prepare_data(config: DataConfig):
    """Build dataset, recall-filtered split, and dataloaders."""
    dataset = build_dataset(config)
    train_dataset, validation_dataset, recall_filter = split_dataset(dataset, config)
    train_loader, validation_loader = build_loaders(
        train_dataset,
        validation_dataset,
        config,
    )
    return (
        dataset,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
        recall_filter,
    )


def new_metric_state() -> dict[str, float]:
    """Create an accumulator for edge-classification metrics."""
    return {
        "loss_sum": 0.0,
        "edges": 0.0,
        "positive_edges": 0.0,
        "negative_edges": 0.0,
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "true_negative": 0.0,
        "quadrat_edges": 0.0,
        "quadrat_positive_edges": 0.0,
        "quadrat_negative_edges": 0.0,
        "quadrat_true_positive": 0.0,
        "quadrat_false_positive": 0.0,
        "quadrat_false_negative": 0.0,
        "quadrat_true_negative": 0.0,
        "degree_abs_error_sum": 0.0,
        "degree_nodes": 0.0,
        "edge_count_abs_error_sum": 0.0,
        "endpoint_count_abs_error_sum": 0.0,
        "topology_graphs": 0.0,
        "decoded_true_positive": 0.0,
        "decoded_false_positive": 0.0,
        "decoded_false_negative": 0.0,
        "quadrat_decoded_true_positive": 0.0,
        "quadrat_decoded_false_positive": 0.0,
        "quadrat_decoded_false_negative": 0.0,
        "decoded_metrics_enabled": 0.0,
        "decoded_graphs": 0.0,
        "decoder_fallbacks": 0.0,
        "grad_norm_sum": 0.0,
        "grad_norm_steps": 0.0,
    }


def edge_quadrat_mask(batch) -> torch.Tensor:
    """Return directed-edge mask for edges incident to any quadrat>0 node."""
    edge_index = batch.edge_index
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=edge_index.device)

    quadrat = getattr(batch, "quadrat", None)
    if quadrat is None:
        return torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    quadrat = quadrat.to(device=edge_index.device)
    source, target = edge_index
    return (quadrat[source] > 0) | (quadrat[target] > 0)


def add_binary_counts(
    state: dict[str, float],
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = "",
) -> None:
    """Accumulate TP/FP/FN/TN counts for a boolean prediction slice."""
    if prefix:
        prefix = f"{prefix}_"
    state[f"{prefix}true_positive"] += float((predictions & targets).sum().item())
    state[f"{prefix}false_positive"] += float((predictions & ~targets).sum().item())
    state[f"{prefix}false_negative"] += float((~predictions & targets).sum().item())
    state[f"{prefix}true_negative"] += float((~predictions & ~targets).sum().item())


def update_metric_state(
    state: dict[str, float],
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    batch,
    threshold: float,
    include_decoded: bool,
    decoder: str,
) -> None:
    """Add one batch of logits/labels to a metric accumulator."""
    labels = labels.detach().float()
    logits = logits.detach()
    edge_count = int(labels.numel())

    if edge_count == 0:
        return

    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets = labels >= 0.5

    state["loss_sum"] += float(loss.detach().item()) * edge_count
    state["edges"] += float(edge_count)
    state["positive_edges"] += float(targets.sum().item())
    state["negative_edges"] += float((~targets).sum().item())
    add_binary_counts(state, predictions=predictions, targets=targets)

    quadrat_mask = edge_quadrat_mask(batch)
    quadrat_edge_count = int(quadrat_mask.sum().item())
    if quadrat_edge_count:
        quadrat_predictions = predictions[quadrat_mask]
        quadrat_targets = targets[quadrat_mask]
        state["quadrat_edges"] += float(quadrat_edge_count)
        state["quadrat_positive_edges"] += float(quadrat_targets.sum().item())
        state["quadrat_negative_edges"] += float((~quadrat_targets).sum().item())
        add_binary_counts(
            state,
            predictions=quadrat_predictions,
            targets=quadrat_targets,
            prefix="quadrat",
        )

    update_topology_metric_state(
        state,
        logits=logits,
        batch=batch,
        threshold=threshold,
        include_decoded=include_decoded,
        decoder=decoder,
    )


def update_topology_metric_state(
    state: dict[str, float],
    *,
    logits: torch.Tensor,
    batch,
    threshold: float,
    include_decoded: bool,
    decoder: str,
) -> None:
    """Add path-topology errors for one batched prediction."""
    undirected = undirected_edge_values(
        logits.detach(),
        batch.edge_y.detach(),
        batch.edge_index.detach(),
        num_nodes=int(batch.num_nodes),
    )
    if undirected is None:
        return

    source, target, probability, target_value = undirected
    node_count = int(batch.num_nodes)
    predicted_value = (probability >= threshold).to(probability.dtype)
    predicted_degree = probability.new_zeros(node_count)
    target_degree = probability.new_zeros(node_count)
    predicted_degree.scatter_add_(0, source, predicted_value)
    predicted_degree.scatter_add_(0, target, predicted_value)
    target_degree.scatter_add_(0, source, target_value)
    target_degree.scatter_add_(0, target, target_value)

    node_batch = getattr(batch, "batch", None)
    if node_batch is None:
        node_batch = torch.zeros(node_count, dtype=torch.long, device=logits.device)
    graph_count = int(node_batch.max().item()) + 1 if node_count else 0
    graph_index = node_batch[source]
    predicted_edge_counts = probability.new_zeros(graph_count)
    target_edge_counts = probability.new_zeros(graph_count)
    predicted_edge_counts.scatter_add_(0, graph_index, predicted_value)
    target_edge_counts.scatter_add_(0, graph_index, target_value)

    predicted_endpoints = (predicted_degree == 1.0).to(probability.dtype)
    endpoint_counts = probability.new_zeros(graph_count)
    endpoint_counts.scatter_add_(0, node_batch, predicted_endpoints)

    state["degree_abs_error_sum"] += float(
        (predicted_degree - target_degree).abs().sum().item()
    )
    state["degree_nodes"] += float(node_count)
    state["edge_count_abs_error_sum"] += float(
        (predicted_edge_counts - target_edge_counts).abs().sum().item()
    )
    state["endpoint_count_abs_error_sum"] += float(
        (endpoint_counts - 2.0).abs().sum().item()
    )
    state["topology_graphs"] += float(graph_count)
    if include_decoded:
        state["decoded_metrics_enabled"] = 1.0
        update_decoded_path_metric_state(
            state,
            logits=logits,
            batch=batch,
            decoder=decoder,
        )


def update_decoded_path_metric_state(
    state: dict[str, float],
    *,
    logits: torch.Tensor,
    batch,
    decoder: str,
) -> None:
    """Add path-decoded edge metrics for one batched prediction."""
    node_batch = getattr(batch, "batch", None)
    if node_batch is None:
        graph_count = 1
    else:
        graph_count = int(node_batch.max().item()) + 1 if int(batch.num_nodes) else 0

    for graph_id in range(graph_count):
        if node_batch is None:
            node_mask = torch.ones(
                int(batch.num_nodes),
                dtype=torch.bool,
                device=logits.device,
            )
        else:
            node_mask = node_batch == graph_id

        node_ids = torch.nonzero(node_mask, as_tuple=False).view(-1)
        if node_ids.numel() == 0:
            continue

        node_start = int(node_ids.min().item())
        node_end = int(node_ids.max().item()) + 1
        edge_mask = (
            (batch.edge_index[0] >= node_start)
            & (batch.edge_index[0] < node_end)
            & (batch.edge_index[1] >= node_start)
            & (batch.edge_index[1] < node_end)
        )
        edge_ids = torch.nonzero(edge_mask, as_tuple=False).view(-1)
        if edge_ids.numel() == 0:
            continue

        local_edge_index = batch.edge_index[:, edge_ids] - node_start
        local_logits = logits[edge_ids]
        local_labels = batch.edge_y[edge_ids]
        quadrat = getattr(batch, "quadrat", None)
        local_quadrat = (
            quadrat[node_start:node_end].detach().cpu()
            if quadrat is not None
            else None
        )
        decode_result = decode_path_edges_with_diagnostics(
            local_logits,
            local_edge_index,
            node_count=int(node_ids.numel()),
            decoder=decoder,
        )
        decoded_edges = decode_result.edges
        state["decoded_graphs"] += 1.0
        state["decoder_fallbacks"] += float(decode_result.used_fallback)
        target_edges = target_undirected_edges(local_labels, local_edge_index)

        true_positive = len(decoded_edges & target_edges)
        false_positive = len(decoded_edges - target_edges)
        false_negative = len(target_edges - decoded_edges)
        state["decoded_true_positive"] += float(true_positive)
        state["decoded_false_positive"] += float(false_positive)
        state["decoded_false_negative"] += float(false_negative)

        quadrat_decoded_edges = filter_quadrat_edges(decoded_edges, local_quadrat)
        quadrat_target_edges = filter_quadrat_edges(target_edges, local_quadrat)
        state["quadrat_decoded_true_positive"] += float(
            len(quadrat_decoded_edges & quadrat_target_edges)
        )
        state["quadrat_decoded_false_positive"] += float(
            len(quadrat_decoded_edges - quadrat_target_edges)
        )
        state["quadrat_decoded_false_negative"] += float(
            len(quadrat_target_edges - quadrat_decoded_edges)
        )


def filter_quadrat_edges(
    edges: set[tuple[int, int]],
    quadrat: torch.Tensor | None,
) -> set[tuple[int, int]]:
    """Keep undirected edges incident to at least one node with quadrat > 0."""
    if quadrat is None:
        return set()

    values = quadrat.detach().cpu().tolist()
    return {
        (source, target)
        for source, target in edges
        if values[source] > 0 or values[target] > 0
    }


def target_undirected_edges(
    labels: torch.Tensor,
    edge_index: torch.Tensor,
) -> set[tuple[int, int]]:
    """Return undirected GT edges from a local directed edge list."""
    edges: set[tuple[int, int]] = set()
    for edge_id, label in enumerate(labels.detach().cpu().tolist()):
        if label < 0.5:
            continue
        source = int(edge_index[0, edge_id].detach().cpu().item())
        target = int(edge_index[1, edge_id].detach().cpu().item())
        if source != target:
            edges.add((min(source, target), max(source, target)))
    return edges


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return ``numerator / denominator`` with a configurable zero fallback."""
    return numerator / denominator if denominator else default


def precision_recall_f1_from_counts(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 from aggregate edge counts."""
    precision = safe_divide(true_positive, true_positive + false_positive, default=1.0)
    recall = safe_divide(true_positive, true_positive + false_negative, default=1.0)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def finalize_metric_state(state: dict[str, float]) -> dict[str, float]:
    """Compute scalar metrics from an accumulated metric state."""
    tp = state["true_positive"]
    fp = state["false_positive"]
    fn = state["false_negative"]
    tn = state["true_negative"]
    edges = state["edges"]
    topology_graphs = state["topology_graphs"]
    precision, recall, f1 = precision_recall_f1_from_counts(tp, fp, fn)
    specificity = safe_divide(tn, tn + fp, default=1.0)
    quadrat_tp = state["quadrat_true_positive"]
    quadrat_fp = state["quadrat_false_positive"]
    quadrat_fn = state["quadrat_false_negative"]
    quadrat_tn = state["quadrat_true_negative"]
    quadrat_precision, quadrat_recall, quadrat_f1 = precision_recall_f1_from_counts(
        quadrat_tp,
        quadrat_fp,
        quadrat_fn,
    )
    decoded_tp = state["decoded_true_positive"]
    decoded_fp = state["decoded_false_positive"]
    decoded_fn = state["decoded_false_negative"]
    if not state["decoded_metrics_enabled"]:
        decoded_precision = float("nan")
        decoded_recall = float("nan")
        decoded_f1 = float("nan")
        quadrat_decoded_precision = float("nan")
        quadrat_decoded_recall = float("nan")
        quadrat_decoded_f1 = float("nan")
    elif decoded_tp + decoded_fp + decoded_fn:
        decoded_precision, decoded_recall, decoded_f1 = precision_recall_f1_from_counts(
            decoded_tp,
            decoded_fp,
            decoded_fn,
        )
    else:
        decoded_precision = 0.0
        decoded_recall = 0.0
        decoded_f1 = 0.0
    if state["decoded_metrics_enabled"]:
        quadrat_decoded_tp = state["quadrat_decoded_true_positive"]
        quadrat_decoded_fp = state["quadrat_decoded_false_positive"]
        quadrat_decoded_fn = state["quadrat_decoded_false_negative"]
        if quadrat_decoded_tp + quadrat_decoded_fp + quadrat_decoded_fn:
            (
                quadrat_decoded_precision,
                quadrat_decoded_recall,
                quadrat_decoded_f1,
            ) = precision_recall_f1_from_counts(
                quadrat_decoded_tp,
                quadrat_decoded_fp,
                quadrat_decoded_fn,
            )
        else:
            quadrat_decoded_precision = 0.0
            quadrat_decoded_recall = 0.0
            quadrat_decoded_f1 = 0.0

    return {
        "loss": safe_divide(state["loss_sum"], edges),
        "edges": int(edges),
        "positive_edges": int(state["positive_edges"]),
        "negative_edges": int(state["negative_edges"]),
        "positive_edge_ratio": safe_divide(state["positive_edges"], edges),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "quadrat_edges": int(state["quadrat_edges"]),
        "quadrat_positive_edges": int(state["quadrat_positive_edges"]),
        "quadrat_negative_edges": int(state["quadrat_negative_edges"]),
        "quadrat_true_positive": int(quadrat_tp),
        "quadrat_false_positive": int(quadrat_fp),
        "quadrat_false_negative": int(quadrat_fn),
        "quadrat_true_negative": int(quadrat_tn),
        "accuracy": safe_divide(tp + tn, edges),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "quadrat_accuracy": safe_divide(
            quadrat_tp + quadrat_tn,
            state["quadrat_edges"],
        ),
        "quadrat_precision": quadrat_precision,
        "quadrat_recall": quadrat_recall,
        "quadrat_specificity": safe_divide(
            quadrat_tn,
            quadrat_tn + quadrat_fp,
            default=1.0,
        ),
        "quadrat_f1": quadrat_f1,
        "decoded_precision": decoded_precision,
        "decoded_recall": decoded_recall,
        "decoded_f1": decoded_f1,
        "quadrat_decoded_precision": quadrat_decoded_precision,
        "quadrat_decoded_recall": quadrat_decoded_recall,
        "quadrat_decoded_f1": quadrat_decoded_f1,
        "decoded_graphs": int(state["decoded_graphs"]),
        "decoder_fallbacks": int(state["decoder_fallbacks"]),
        "decoder_fallback_rate": safe_divide(
            state["decoder_fallbacks"],
            state["decoded_graphs"],
        ),
        "degree_mae": safe_divide(
            state["degree_abs_error_sum"],
            state["degree_nodes"],
        ),
        "edge_count_mae": safe_divide(
            state["edge_count_abs_error_sum"],
            topology_graphs,
        ),
        "endpoint_count_mae": safe_divide(
            state["endpoint_count_abs_error_sum"],
            topology_graphs,
        ),
        "grad_norm": safe_divide(
            state["grad_norm_sum"],
            state["grad_norm_steps"],
            default=float("nan"),
        ),
    }


def total_gradient_norm(parameters) -> float:
    """Return the global L2 norm of currently accumulated gradients."""
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().data.norm(2)
        squared_norm += float(parameter_norm.detach().cpu().item()) ** 2
    return squared_norm ** 0.5


def train_one_epoch(
    model: EdgeNodeGNN,
    loader: DataLoader,
    criterion: EdgeClassificationLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    threshold: float,
    grad_clip_norm: float,
    include_decoded: bool,
    decoder: str,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    state = new_metric_state()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch)
        loss.backward()

        if grad_clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )
            state["grad_norm_sum"] += float(grad_norm.detach().cpu().item())
        else:
            state["grad_norm_sum"] += total_gradient_norm(model.parameters())
        state["grad_norm_steps"] += 1.0

        optimizer.step()
        update_metric_state(
            state,
            logits=logits,
            labels=batch.edge_y,
            loss=loss,
            batch=batch,
            threshold=threshold,
            include_decoded=include_decoded,
            decoder=decoder,
        )

    return finalize_metric_state(state)


@torch.no_grad()
def evaluate(
    model: EdgeNodeGNN,
    loader: DataLoader,
    criterion: EdgeClassificationLoss,
    *,
    device: torch.device,
    threshold: float,
    include_decoded: bool = True,
    decoder: str = "ilp",
) -> dict[str, float]:
    """Run validation over a dataloader."""
    model.eval()
    state = new_metric_state()

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch)
        update_metric_state(
            state,
            logits=logits,
            labels=batch.edge_y,
            loss=loss,
            batch=batch,
            threshold=threshold,
            include_decoded=include_decoded,
            decoder=decoder,
        )

    return finalize_metric_state(state)


def print_epoch_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """Print one compact train/validation progress line."""
    print(
        f"epoch={epoch:04d} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"train_f1={train_metrics['f1']:.4f} "
        f"train_recall={train_metrics['recall']:.4f} "
        f"train_precision={train_metrics['precision']:.4f} "
        f"train_decoded_f1={train_metrics['decoded_f1']:.4f} "
        f"train_quadrat_f1={train_metrics['quadrat_f1']:.4f} "
        f"train_quadrat_recall={train_metrics['quadrat_recall']:.4f} "
        f"train_quadrat_precision={train_metrics['quadrat_precision']:.4f} "
        f"train_quadrat_decoded_f1={train_metrics['quadrat_decoded_f1']:.4f} "
        f"train_decoder_fallback_rate={train_metrics['decoder_fallback_rate']:.4f} "
        f"train_grad_norm={train_metrics['grad_norm']:.4f} "
        f"train_degree_mae={train_metrics['degree_mae']:.4f} "
        f"train_edge_count_mae={train_metrics['edge_count_mae']:.4f} "
        f"train_endpoint_mae={train_metrics['endpoint_count_mae']:.4f} "
        f"val_loss={validation_metrics['loss']:.6f} "
        f"val_f1={validation_metrics['f1']:.4f} "
        f"val_recall={validation_metrics['recall']:.4f} "
        f"val_precision={validation_metrics['precision']:.4f} "
        f"val_decoded_f1={validation_metrics['decoded_f1']:.4f} "
        f"val_quadrat_f1={validation_metrics['quadrat_f1']:.4f} "
        f"val_quadrat_recall={validation_metrics['quadrat_recall']:.4f} "
        f"val_quadrat_precision={validation_metrics['quadrat_precision']:.4f} "
        f"val_quadrat_decoded_f1={validation_metrics['quadrat_decoded_f1']:.4f} "
        f"val_decoder_fallback_rate={validation_metrics['decoder_fallback_rate']:.4f} "
        f"val_degree_mae={validation_metrics['degree_mae']:.4f} "
        f"val_edge_count_mae={validation_metrics['edge_count_mae']:.4f} "
        f"val_endpoint_mae={validation_metrics['endpoint_count_mae']:.4f}"
    )


def save_checkpoint(
    path: Path,
    *,
    model: EdgeNodeGNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    node_feature_dim: int,
    edge_feature_dim: int,
    pos_weight: float | None,
    recall_filter: dict[str, Any],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """Persist the best validation checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "pos_weight": pos_weight,
            "data_config": dataclass_to_dict(data_config),
            "model_config": dataclass_to_dict(model_config),
            "train_config": dataclass_to_dict(train_config),
            "recall_filter": recall_filter,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a training checkpoint across PyTorch default variants."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move restored optimizer tensors to the active training device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def dataclass_to_dict(value) -> dict[str, Any]:
    """Convert dataclass configs to JSON/checkpoint friendly dictionaries."""
    result = asdict(value)
    for key, item in list(result.items()):
        if isinstance(item, Path):
            result[key] = str(item)
    return result


def json_ready(value: Any) -> Any:
    """Convert paths and non-finite floats to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_ready(item)
            for item in value
        ]
    return value


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    """Write final training metadata and metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_checkpoint_payload(path: Path, checkpoint: dict[str, Any]) -> None:
    """Persist an already materialized checkpoint dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def evaluate_best_checkpoint_with_decoded_metrics(
    *,
    model: EdgeNodeGNN,
    checkpoint_path: Path,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: EdgeClassificationLoss,
    device: torch.device,
    threshold: float,
    decoder: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load the best checkpoint and compute expensive decoded metrics once."""
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("evaluating best checkpoint with decoded path metrics:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  decoder: {decoder}")

    train_metrics = evaluate(
        model,
        train_loader,
        criterion,
        device=device,
        threshold=threshold,
        include_decoded=True,
        decoder=decoder,
    )
    validation_metrics = evaluate(
        model,
        validation_loader,
        criterion,
        device=device,
        threshold=threshold,
        include_decoded=True,
        decoder=decoder,
    )

    checkpoint["selection_train_metrics"] = checkpoint.get("train_metrics", {})
    checkpoint["selection_validation_metrics"] = checkpoint.get(
        "validation_metrics",
        {},
    )
    checkpoint["train_metrics"] = train_metrics
    checkpoint["validation_metrics"] = validation_metrics
    checkpoint["decoded_metrics_evaluated_after_training"] = True
    save_checkpoint_payload(checkpoint_path, checkpoint)

    print(
        "best checkpoint decoded train: "
        f"f1={train_metrics['f1']:.4f} "
        f"decoded_f1={train_metrics['decoded_f1']:.4f} "
        f"quadrat_decoded_f1={train_metrics['quadrat_decoded_f1']:.4f} "
        f"fallback_rate={train_metrics['decoder_fallback_rate']:.4f}"
    )
    print(
        "best checkpoint decoded validation: "
        f"f1={validation_metrics['f1']:.4f} "
        f"decoded_f1={validation_metrics['decoded_f1']:.4f} "
        f"quadrat_decoded_f1={validation_metrics['quadrat_decoded_f1']:.4f} "
        f"fallback_rate={validation_metrics['decoder_fallback_rate']:.4f}"
    )

    return train_metrics, validation_metrics


def run_shapley_explainability(
    *,
    data_config: DataConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> None:
    """Run grouped Shapley explainability against the best checkpoint."""
    if not train_config.shapley_enabled:
        return

    if train_config.run_dir is None:
        raise ValueError("Shapley explainability requires train_config.run_dir")

    output_path = (
        train_config.shapley_output
        or train_config.run_dir / "shapley_explainability.json"
    )
    command = [
        sys.executable,
        "-m",
        "hierreco_nn.explainability.shapley",
        str(train_config.run_dir),
        str(data_config.dataset_root),
        "--checkpoint",
        str(train_config.checkpoint_path),
        "--output",
        str(output_path),
        "--device",
        str(device),
        "--threshold",
        str(train_config.threshold),
        "--mode",
        train_config.shapley_mode,
        "--permutations",
        str(train_config.shapley_permutations),
        "--subset-batch-size",
        str(train_config.shapley_subset_batch_size),
        "--seed",
        str(train_config.shapley_seed),
        "--min-candidate-recall",
        str(data_config.min_candidate_recall),
    ]
    if train_config.shapley_max_train_samples is not None:
        command.extend(
            ["--max-train-samples", str(train_config.shapley_max_train_samples)]
        )
    if train_config.shapley_max_validation_samples is not None:
        command.extend(
            [
                "--max-validation-samples",
                str(train_config.shapley_max_validation_samples),
            ]
        )

    print("running shapley explainability:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        print("shapley explainability result:")
        print(json.dumps(json_ready(payload.get("splits", {})), indent=2, ensure_ascii=False))


def next_run_directory(runs_root: Path) -> Path:
    """Create and return ``run_{last_idx + 1}`` under ``runs_root``."""
    runs_root.mkdir(parents=True, exist_ok=True)
    max_index = -1

    for path in runs_root.iterdir():
        if not path.is_dir() or not path.name.startswith("run_"):
            continue

        suffix = path.name.removeprefix("run_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))

    next_index = max_index + 1
    while True:
        run_dir = runs_root / f"run_{next_index}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            next_index += 1


def copy_model_snapshot(run_dir: Path) -> Path:
    """Copy the current model definition into the run directory."""
    source = Path(__file__).with_name("model.py")
    target = run_dir / "model.py"
    shutil.copy2(source, target)
    return target


def infer_feature_dims(train_dataset) -> tuple[int, int]:
    """Infer node and edge feature dimensions from the first train graph."""
    sample = train_dataset[0]
    return int(sample.x.size(-1)), int(sample.edge_attr.size(-1))


def resolve_device(device_name: str) -> torch.device:
    """Resolve and validate CLI device names."""
    if device_name != "auto":
        device = torch.device(device_name)

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested device {device}, but CUDA is not available."
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested CUDA device index {device.index}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available."
                )
        elif device.type == "mps":
            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
            if not mps_available:
                raise RuntimeError(
                    "Requested device mps, but PyTorch MPS is not available. "
                    "Use --device cpu or --cuda-device N on a CUDA machine."
                )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds used by Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    """Prepare data, train the model, and save the best validation checkpoint."""
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    if train_config.run_dir is not None:
        print(f"run directory: {train_config.run_dir}")
    print(f"training device: {device}")
    (
        dataset,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
        recall_filter,
    ) = prepare_data(data_config)

    print(f"loaded samples after min_nodes filtering: {len(dataset)}")
    print(
        f"bin split: train_bins={dataset.train_bins}, "
        f"validation_bins={dataset.validation_bins}, "
        f"split_seed={dataset.split_seed}, "
        f"profile_samples={dataset.geometry_profile.sample_count}"
    )
    print_recall_filter_summary(recall_filter["train"])
    print_recall_filter_summary(recall_filter["validation"])
    print_summary("train data", summarize_dataset(train_dataset))
    print_summary("validation data", summarize_dataset(validation_dataset))
    inspect_first_batch(train_loader)

    node_feature_dim, edge_feature_dim = infer_feature_dims(train_dataset)
    model = EdgeNodeGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.layers,
        attention_heads=model_config.attention_heads,
        gnn_branch=model_config.gnn_branch,
        edge_competition_layers=model_config.edge_competition_layers,
        incident_selector=model_config.incident_selector,
        dropout=model_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    pos_weight = (
        compute_pos_weight(train_dataset, max_value=train_config.max_pos_weight)
        if train_config.use_pos_weight
        else None
    )
    criterion = EdgeClassificationLoss(
        pos_weight=pos_weight,
        bce_loss_weight=train_config.bce_loss_weight,
        focal_gamma=train_config.focal_gamma,
        label_smoothing=train_config.label_smoothing,
        degree_loss_weight=train_config.degree_loss_weight,
        edge_count_loss_weight=train_config.edge_count_loss_weight,
        endpoint_loss_weight=train_config.endpoint_loss_weight,
        incident_ranking_loss_weight=train_config.incident_ranking_loss_weight,
        symmetry_loss_weight=train_config.symmetry_loss_weight,
        endpoint_sigma=train_config.endpoint_sigma,
    ).to(device)

    print(
        "model: "
        f"node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, "
        f"hidden_dim={model_config.hidden_dim}, layers={model_config.layers}, "
        f"attention_heads={model_config.attention_heads}, "
        f"gnn_branch={model_config.gnn_branch}, "
        f"dropout={model_config.dropout}"
    )
    parameter_summary = model_parameter_summary(model)
    print_model_parameter_summary(parameter_summary)
    if pos_weight is None:
        print("criterion: BCEWithLogitsLoss without positive weighting")
    else:
        print(
            "criterion: weighted BCEWithLogitsLoss "
            f"pos_weight={pos_weight:.6f}"
        )
    print(
        "path regularization: "
        f"bce={train_config.bce_loss_weight}, "
        f"degree={train_config.degree_loss_weight}, "
        f"edge_count={train_config.edge_count_loss_weight}"
    )
    print(
        "decoded path decoder: "
        f"{train_config.decoder} (final best-checkpoint evaluation only)"
    )
    if train_config.train_decoded_metrics:
        print("per-epoch train decoded metrics: enabled")
    else:
        print("per-epoch train decoded metrics: disabled")
    print("per-epoch validation decoded metrics: disabled")

    start_epoch = 1
    best_epoch = 0
    best_validation_loss = float("inf")
    final_train_metrics: dict[str, float] = {}
    final_validation_metrics: dict[str, float] = {}

    if train_config.resume_from is not None:
        checkpoint = load_checkpoint(train_config.resume_from)
        checkpoint_node_dim = int(checkpoint.get("node_feature_dim", node_feature_dim))
        checkpoint_edge_dim = int(checkpoint.get("edge_feature_dim", edge_feature_dim))
        if checkpoint_node_dim != node_feature_dim or checkpoint_edge_dim != edge_feature_dim:
            raise ValueError(
                "Resume checkpoint feature dimensions do not match current dataset: "
                f"checkpoint node/edge={checkpoint_node_dim}/{checkpoint_edge_dim}, "
                f"current node/edge={node_feature_dim}/{edge_feature_dim}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)

        resumed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        best_epoch = resumed_epoch
        checkpoint_validation = checkpoint.get("validation_metrics") or {}
        best_validation_loss = float(
            checkpoint_validation.get("loss", float("inf"))
        )
        print(
            f"resumed checkpoint: {train_config.resume_from} "
            f"epoch={resumed_epoch} "
            f"best_validation_loss={best_validation_loss:.6f}"
        )

    initial_plot_history = []
    if train_config.resume_from is not None:
        initial_plot_history = load_epoch_history_from_log(
            train_config.log_file,
            max_epoch=start_epoch - 1,
        )
        if initial_plot_history:
            print(
                "plotly resume history: "
                f"loaded {len(initial_plot_history)} epochs from {train_config.log_file}"
            )

    live_plot = PlotlyMetricPlot(
        enabled=train_config.live_plot,
        html_path=train_config.plot_html,
        refresh_ms=train_config.plot_refresh_ms,
        initial_history=initial_plot_history,
    )

    if start_epoch > train_config.epochs:
        raise ValueError(
            f"resume checkpoint epoch is {start_epoch - 1}, but --epochs is "
            f"{train_config.epochs}. Increase --epochs to continue training."
        )

    for epoch in range(start_epoch, train_config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            threshold=train_config.threshold,
            grad_clip_norm=train_config.grad_clip_norm,
            include_decoded=train_config.train_decoded_metrics,
            decoder=train_config.decoder,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device=device,
            threshold=train_config.threshold,
            include_decoded=False,
            decoder=train_config.decoder,
        )

        final_train_metrics = train_metrics
        final_validation_metrics = validation_metrics
        live_plot.update(epoch, train_metrics, validation_metrics)

        if validation_metrics["loss"] < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                train_config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                data_config=data_config,
                model_config=model_config,
                train_config=train_config,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                pos_weight=pos_weight,
                recall_filter=recall_filter,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )

        if (
            epoch == 1
            or epoch == train_config.epochs
            or epoch % train_config.log_every == 0
        ):
            print_epoch_metrics(epoch, train_metrics, validation_metrics)

    best_train_metrics, best_validation_metrics = (
        evaluate_best_checkpoint_with_decoded_metrics(
            model=model,
            checkpoint_path=train_config.checkpoint_path,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=train_config.threshold,
            decoder=train_config.decoder,
        )
    )

    payload = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": str(train_config.checkpoint_path),
        "data_config": dataclass_to_dict(data_config),
        "model_config": dataclass_to_dict(model_config),
        "model_parameters": parameter_summary,
        "train_config": dataclass_to_dict(train_config),
        "recall_filter": recall_filter,
        "epoch_history": live_plot.history,
        "final_train_metrics": final_train_metrics,
        "final_validation_metrics": final_validation_metrics,
        "best_checkpoint_train_metrics": best_train_metrics,
        "best_checkpoint_validation_metrics": best_validation_metrics,
    }

    if train_config.metrics_json is not None:
        write_metrics_json(train_config.metrics_json, payload)

    print(
        f"best validation: epoch={best_epoch} "
        f"loss={best_validation_loss:.6f} "
        f"checkpoint={train_config.checkpoint_path}"
    )
    live_plot.finish()
    run_shapley_explainability(
        data_config=data_config,
        train_config=train_config,
        device=device,
    )
    return payload


def parse_bin_spec(value: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated bin list, or return ``None`` for defaults."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ()

    return tuple(int(part) for part in parts)


def resolve_cross_validation_bins(
    *,
    bin_count: int,
    cv_fold: tuple[int, ...] | None,
    train_bins: tuple[int, ...] | None,
    validation_bins: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Return explicit train/validation bins for requested validation bins."""
    if cv_fold is None:
        return train_bins, validation_bins

    validation = tuple(sorted(set(cv_fold)))
    validation_set = set(validation)
    train = tuple(
        bin_id
        for bin_id in range(bin_count)
        if bin_id not in validation_set
    )
    return train, validation


def parse_args():
    """Parse command-line arguments for model training."""
    project_config = load_project_config_from_cli()
    paths = project_config.paths
    parser = argparse.ArgumentParser(
        description="Train a Hierreco candidate-edge classifier."
    )
    add_config_argument(parser, project_config)
    parser.add_argument(
        "--dataset-root",
        default=paths.dataset_root,
        type=Path,
        help="Directory searched recursively for *sample_*.json files.",
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--bin-count", default=5, type=int)
    parser.add_argument(
        "--cv-val-folds",
        default=None,
        dest="cv_val_folds",
        nargs="+",
        type=int,
        help=(
            "Validation bin id(s) for cross-validation. Example: "
            "--cv-val-folds 2 4 uses bins 2 and 4 for validation and the rest for train."
        ),
    )
    parser.add_argument(
        "--cv-fold",
        dest="cv_val_folds",
        nargs="+",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--split-seed",
        default=13,
        type=int,
        help="Seed for deterministic sample shuffling before bin assignment.",
    )
    parser.add_argument(
        "--train-bins",
        type=parse_bin_spec,
        help="Comma-separated train bin ids. Default: all bins except validation.",
    )
    parser.add_argument(
        "--validation-bins",
        type=parse_bin_spec,
        help="Comma-separated validation bin ids. Default: final two bins when possible.",
    )
    parser.add_argument(
        "--min-candidate-recall",
        default=1.0,
        type=float,
        help="Keep only samples whose candidate graph reaches this GT-edge recall.",
    )
    parser.add_argument("--candidate-density", default=1.0, type=float)
    parser.add_argument("--split-sensitivity", default=1.0, type=float)
    parser.add_argument("--edge-length-strictness", default=1.0, type=float)
    parser.add_argument("--triangle-min-angle-degrees", default=30.0, type=float)
    parser.add_argument("--min-sub-nodes", default=2, type=int)
    parser.add_argument("--min-nodes", default=5, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--cache-dir",
        default=paths.cache_dir,
        type=Path,
        help="Directory for precomputed PyG Data cache. Default: config paths.cache_dir.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable precomputed dataset cache and build Data objects on the fly.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the dataset cache for the current dataset configuration.",
    )
    parser.add_argument(
        "--no-preload-cache",
        action="store_true",
        help="Skip in-memory preload after cache validation.",
    )

    parser.add_argument(
        "--hidden-dim",
        default=48,
        type=int,
        help="Latent node/edge dimension. Default: 48.",
    )
    parser.add_argument(
        "--layers",
        default=2,
        type=int,
        help="Number of GNN message-passing layers per branch. Default: 2.",
    )
    parser.add_argument("--attention-heads", default=4, type=int)
    parser.add_argument(
        "--gnn-branch",
        choices=("gine", "gatv2"),
        default="gine",
        help="Message-passing branch. Default: GINE.",
    )
    parser.add_argument(
        "--edge-competition-layers",
        default=0,
        type=int,
        help=(
            "Legacy option kept for compatibility. The simplified model "
            "does not use an edge-competition graph."
        ),
    )
    parser.add_argument(
        "--incident-selector",
        dest="incident_selector",
        action="store_true",
        default=False,
        help="Legacy option kept for compatibility. Disabled by default.",
    )
    parser.add_argument(
        "--no-incident-selector",
        dest="incident_selector",
        action="store_false",
        help="Legacy no-op; the simplified model has no incident selector.",
    )
    parser.add_argument("--dropout", default=0.15, type=float)

    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip-norm", default=1.0, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument(
        "--train-decoded-metrics",
        dest="train_decoded_metrics",
        action="store_true",
        default=False,
        help=(
            "Compute decoded_f1 for train batches during every epoch. "
            "Disabled by default; decoded metrics are computed once for the "
            "best checkpoint after training."
        ),
    )
    parser.add_argument(
        "--no-train-decoded-metrics",
        dest="train_decoded_metrics",
        action="store_false",
        help="Skip per-epoch train decoded_f1. This is the default.",
    )
    parser.add_argument(
        "--decoder",
        choices=DECODER_CHOICES,
        default="ilp",
        help="Path decoder used for decoded metrics. Default: ilp.",
    )
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument(
        "--device",
        default="mps",
        help=(
            "Training device. Default: mps for macOS Apple Silicon. "
            "Use cpu, auto, cuda, cuda:0, etc."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        default=None,
        type=int,
        help="CUDA device index shortcut. For example, --cuda-device 1 uses cuda:1.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=Path,
        help="Checkpoint path. Default: runs/run_{idx}/best_val_loss.pt.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        type=Path,
        help="Load model and optimizer state from this checkpoint before training.",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        type=Path,
        help=(
            "Resume directly inside this run directory. Shorthand for "
            "--run-dir RUN --resume-from RUN/best_val_loss.pt."
        ),
    )
    parser.add_argument("--metrics-json", default=None, type=Path)
    parser.add_argument(
        "--log-file",
        default=None,
        type=Path,
        help="Append terminal training output to this .log file. Default: run dir.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable writing terminal training output to a .log file.",
    )
    parser.add_argument("--log-every", default=1, type=int)
    parser.add_argument(
        "--no-live-plot",
        action="store_true",
        help="Disable the live Plotly HTML metric plot.",
    )
    parser.add_argument(
        "--rebuild-plot-from-log",
        action="store_true",
        help="Rebuild the Plotly HTML plot from the run training.log and exit.",
    )
    parser.add_argument(
        "--plot-html",
        default=None,
        type=Path,
        help="HTML file updated with the live Plotly metric plot. Default: run dir.",
    )
    parser.add_argument("--plot-refresh-ms", default=2000, type=int)
    parser.add_argument(
        "--runs-root",
        default=paths.runs_root,
        type=Path,
        help="Directory where run_{idx} folders are created.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        type=Path,
        help="Use this existing/new run directory instead of creating run_{idx}.",
    )
    parser.add_argument(
        "--no-shapley",
        action="store_true",
        help="Disable post-training grouped Shapley explainability.",
    )
    parser.add_argument(
        "--shapley-mode",
        choices=("exact", "permutation"),
        default="exact",
        help=(
            "Grouped Shapley computation mode. Exact evaluates all feature-group "
            "subsets and is the default."
        ),
    )
    parser.add_argument(
        "--shapley-permutations",
        default=16,
        type=int,
        help=(
            "Monte Carlo permutations per sample when --shapley-mode permutation "
            "is used. Ignored by exact mode."
        ),
    )
    parser.add_argument(
        "--shapley-subset-batch-size",
        default=128,
        type=int,
        help="Masked subset evaluation batch size for exact Shapley. Default: 128.",
    )
    parser.add_argument("--shapley-max-train-samples", default=None, type=int)
    parser.add_argument("--shapley-max-validation-samples", default=None, type=int)
    parser.add_argument("--shapley-seed", default=13, type=int)
    parser.add_argument(
        "--shapley-output",
        default=None,
        type=Path,
        help="Shapley JSON output. Default: run_dir/shapley_explainability.json.",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable dataset-level positive-class weighting.",
    )
    parser.add_argument("--max-pos-weight", default=None, type=float)
    parser.add_argument("--bce-loss-weight", default=1.0, type=float)
    parser.add_argument("--focal-gamma", default=0.0, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--degree-loss-weight", default=0.02, type=float)
    parser.add_argument("--edge-count-loss-weight", default=0.01, type=float)
    parser.add_argument("--endpoint-loss-weight", default=0.0, type=float)
    parser.add_argument(
        "--incident-ranking-loss-weight",
        default=0.0,
        type=float,
        help=(
            "Legacy option kept for compatibility. The simplified criterion "
            "does not use incident assignment by default."
        ),
    )
    parser.add_argument("--symmetry-loss-weight", default=0.0, type=float)
    parser.add_argument("--endpoint-sigma", default=0.35, type=float)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.bin_count < 1:
        parser.error("--bin-count must be >= 1")
    if args.cv_val_folds is not None:
        invalid_cv_bins = [
            bin_id
            for bin_id in args.cv_val_folds
            if bin_id < 0 or bin_id >= args.bin_count
        ]
        if invalid_cv_bins:
            parser.error(
                f"--cv-val-folds bin ids must be in 0..{args.bin_count - 1}, "
                f"got {invalid_cv_bins}"
            )
        if len(set(args.cv_val_folds)) >= args.bin_count:
            parser.error("--cv-val-folds must leave at least one train bin")
        if args.train_bins is not None or args.validation_bins is not None:
            parser.error(
                "--cv-val-folds cannot be combined with "
                "--train-bins/--validation-bins"
            )
    if not 0.0 <= args.min_candidate_recall <= 1.0:
        parser.error("--min-candidate-recall must be in [0, 1]")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be >= 1")
    if args.layers < 1:
        parser.error("--layers must be >= 1")
    if args.attention_heads < 1:
        parser.error("--attention-heads must be >= 1")
    if args.edge_competition_layers < 0:
        parser.error("--edge-competition-layers must be >= 0")
    if args.hidden_dim % args.attention_heads != 0:
        parser.error("--hidden-dim must be divisible by --attention-heads")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")
    if args.shapley_permutations < 1:
        parser.error("--shapley-permutations must be >= 1")
    if args.shapley_subset_batch_size < 1:
        parser.error("--shapley-subset-batch-size must be >= 1")
    if args.shapley_max_train_samples is not None and args.shapley_max_train_samples < 1:
        parser.error("--shapley-max-train-samples must be >= 1")
    if (
        args.shapley_max_validation_samples is not None
        and args.shapley_max_validation_samples < 1
    ):
        parser.error("--shapley-max-validation-samples must be >= 1")
    if args.max_pos_weight is not None and args.max_pos_weight <= 0.0:
        parser.error("--max-pos-weight must be positive")
    if args.bce_loss_weight < 0.0:
        parser.error("--bce-loss-weight must be non-negative")
    if args.degree_loss_weight < 0.0:
        parser.error("--degree-loss-weight must be non-negative")
    if args.edge_count_loss_weight < 0.0:
        parser.error("--edge-count-loss-weight must be non-negative")
    if args.endpoint_loss_weight < 0.0:
        parser.error("--endpoint-loss-weight must be non-negative")
    if args.incident_ranking_loss_weight < 0.0:
        parser.error("--incident-ranking-loss-weight must be non-negative")
    if args.symmetry_loss_weight < 0.0:
        parser.error("--symmetry-loss-weight must be non-negative")
    if args.endpoint_sigma <= 0.0:
        parser.error("--endpoint-sigma must be positive")
    if args.cuda_device is not None and args.cuda_device < 0:
        parser.error("--cuda-device must be >= 0")
    if args.plot_refresh_ms < 250:
        parser.error("--plot-refresh-ms must be >= 250")
    if args.resume_run is not None and args.run_dir is not None:
        parser.error("--resume-run already sets --run-dir; do not pass both")
    if args.resume_run is not None and not args.resume_run.is_dir():
        parser.error(f"--resume-run must be an existing run directory: {args.resume_run}")
    if args.resume_run is not None and args.resume_from is None:
        resume_checkpoint = args.resume_run / "best_val_loss.pt"
        if not resume_checkpoint.exists():
            parser.error(f"--resume-run checkpoint does not exist: {resume_checkpoint}")
    if args.resume_from is not None and not args.resume_from.exists():
        parser.error(f"--resume-from does not exist: {args.resume_from}")
    try:
        validate_decoder(args.decoder)
    except RuntimeError as error:
        parser.error(str(error))

    return args


def main() -> None:
    args = parse_args()
    run_dir = args.resume_run or args.run_dir or next_run_directory(args.runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_from = args.resume_from
    if args.resume_run is not None and resume_from is None:
        resume_from = args.resume_run / "best_val_loss.pt"
    checkpoint_path = args.checkpoint_path or run_dir / "best_val_loss.pt"
    metrics_json = args.metrics_json or run_dir / "metrics.json"
    log_file = None if args.no_log_file else (args.log_file or run_dir / "training.log")
    plot_html = args.plot_html or run_dir / "training_plot.html"

    if args.rebuild_plot_from_log:
        if log_file is None:
            raise ValueError("--rebuild-plot-from-log requires a log file")
        history = load_epoch_history_from_log(log_file)
        live_plot = PlotlyMetricPlot(
            enabled=True,
            html_path=plot_html,
            refresh_ms=args.plot_refresh_ms,
            initial_history=history,
        )
        live_plot.finish()
        print(
            f"rebuilt plot from log: epochs={len(history)} "
            f"log={log_file} html={plot_html}"
        )
        return

    train_bins, validation_bins = resolve_cross_validation_bins(
        bin_count=args.bin_count,
        cv_fold=args.cv_val_folds,
        train_bins=args.train_bins,
        validation_bins=args.validation_bins,
    )

    data_config = DataConfig(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        bin_count=args.bin_count,
        split_seed=args.split_seed,
        train_bins=train_bins,
        validation_bins=validation_bins,
        min_candidate_recall=args.min_candidate_recall,
        candidate_density=args.candidate_density,
        split_sensitivity=args.split_sensitivity,
        edge_length_strictness=args.edge_length_strictness,
        triangle_min_angle_degrees=args.triangle_min_angle_degrees,
        min_sub_nodes=args.min_sub_nodes,
        min_nodes=args.min_nodes,
        num_workers=args.num_workers,
        cache_dir=None if args.no_cache else args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        preload_cache=not args.no_preload_cache,
    )
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        gnn_branch=args.gnn_branch,
        edge_competition_layers=args.edge_competition_layers,
        incident_selector=args.incident_selector,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        threshold=args.threshold,
        seed=args.seed,
        run_dir=run_dir,
        device=(
            f"cuda:{args.cuda_device}"
            if args.cuda_device is not None
            else args.device
        ),
        checkpoint_path=checkpoint_path,
        resume_from=resume_from,
        metrics_json=metrics_json,
        log_file=log_file,
        log_every=args.log_every,
        live_plot=not args.no_live_plot,
        plot_html=plot_html,
        plot_refresh_ms=args.plot_refresh_ms,
        use_pos_weight=not args.no_pos_weight,
        max_pos_weight=args.max_pos_weight,
        bce_loss_weight=args.bce_loss_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        degree_loss_weight=args.degree_loss_weight,
        edge_count_loss_weight=args.edge_count_loss_weight,
        endpoint_loss_weight=args.endpoint_loss_weight,
        incident_ranking_loss_weight=args.incident_ranking_loss_weight,
        symmetry_loss_weight=args.symmetry_loss_weight,
        endpoint_sigma=args.endpoint_sigma,
        train_decoded_metrics=args.train_decoded_metrics,
        decoder=args.decoder,
        shapley_enabled=not args.no_shapley,
        shapley_mode=args.shapley_mode,
        shapley_permutations=args.shapley_permutations,
        shapley_subset_batch_size=args.shapley_subset_batch_size,
        shapley_max_train_samples=args.shapley_max_train_samples,
        shapley_max_validation_samples=args.shapley_max_validation_samples,
        shapley_seed=args.shapley_seed,
        shapley_output=args.shapley_output,
    )
    with tee_std_streams(train_config.log_file, run_label="Hierreco training run"):
        model_snapshot = copy_model_snapshot(run_dir)
        print(f"model snapshot: {model_snapshot}")
        run_training(data_config, model_config, train_config)


if __name__ == "__main__":
    main()
