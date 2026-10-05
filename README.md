# Inferring Hieroglyph Sequences by Graph Neural Networks

This repository contains the source code and preprocessed samples associated
with PUBLICATION. It provides the neural reconstruction pipeline used to infer
hieroglyph reading sequences from candidate graph edges.

Any use of the source code, included preprocessed samples, generated artifacts,
or derived data is conditional on proper citation of PUBLICATION.

## Citation

```bibtex
@article{PUBLICATION,
  title = {PUBLICATION},
  author = {PUBLICATION},
  journal = {PUBLICATION},
  year = {PUBLICATION},
  doi = {PUBLICATION},
  url = {PUBLICATION}
}
```

The same BibTeX entry is stored in `CITATION.bib`.

## Overview

The package supports the complete experimental pipeline:

- export connected GraphML components to normalized JSON samples
- build target-independent candidate graphs
- construct node and edge features
- train a graph neural candidate-edge classifier
- decode edge scores into simple path predictions
- export evaluation JSON and CSV artifacts
- render candidate graph and prediction visualizations
- compute grouped Shapley feature attributions

## Repository Contents

```text
.
|-- CITATION.bib
|-- config.toml
|-- json_data/
|-- src/hierreco_nn/
|   |-- cli/
|   |-- dataset/
|   |-- explainability/
|   |-- config.py
|   |-- logging_utils.py
|   |-- losses.py
|   |-- model.py
|   |-- path_decoder.py
|   `-- training.py
|-- Makefile
|-- MANIFEST.in
|-- pyproject.toml
`-- README.md
```

Main components:

| Path | Purpose |
| --- | --- |
| `src/hierreco_nn/dataset/` | JSON loading, candidate graph construction, features, splits, visualization |
| `src/hierreco_nn/cli/` | Dataset conversion, statistics, evaluation export, visualization tools |
| `src/hierreco_nn/explainability/` | Grouped Shapley explainability |
| `src/hierreco_nn/model.py` | Graph neural edge classifier |
| `src/hierreco_nn/losses.py` | BCE loss and path-topology regularizers |
| `src/hierreco_nn/path_decoder.py` | Greedy and ILP path decoders |
| `src/hierreco_nn/training.py` | Training loop, metrics, checkpoints, logs, plots |
| `json_data/` | Preprocessed JSON samples included with the repository |
| `config.toml` | Default path configuration |
| `CITATION.bib` | BibTeX citation placeholder for PUBLICATION |

## Installation

The project targets Python 3.11 or newer. The default setup creates a
repository-local virtual environment at `.venv/`.

```bash
make install
```

Manual equivalent:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -e ".[visualization]"
```

Development installation:

```bash
make install-dev
```

Environment check:

```bash
./.venv/bin/python -c "import sys, torch, scipy; print(sys.prefix); print(torch.__version__, scipy.__version__)"
```

## Configuration

Default filesystem locations are defined in `config.toml`:

```toml
[paths]
dataset_root = "json_data"
cache_dir = "cache"
runs_root = "runs"
matplotlib_config_dir = ".cache/matplotlib"
```

Relative paths are resolved from the directory containing `config.toml`. All
CLI entry points accept `--config`; command-specific path arguments override
configuration values. The same config path may be selected through the
environment:

```bash
export HIERRECO_NN_CONFIG=config.toml
```

## Data

`json_data/` contains preprocessed JSON samples extracted for this repository.
The original data source and download information are provided in PUBLICATION.

The training pipeline reads files named `*sample_*.json`. Each sample contains:

| Field | Description |
| --- | --- |
| `nodes_array` | Ordered node identifiers for one connected component |
| `edges_list` | Target undirected edges between node identifiers |
| `component_id` | Source component identifier |
| `graph_type` | Sequence orientation, `row` or `col` |
| `feature` | Per-node metadata and image-derived measurements |

Per-node metadata includes transformed sequence coordinates, original image
coordinates, sequence position, quadrat, blob id, transformed bounding-box
fields, and optional Zernike moment magnitudes.

Additional JSON samples are generated from GraphML components with
`hierreco-graphml2samples`.

Each GraphML component must carry its orientation metadata on exactly one node:
`seq_type`, `legs_point_to`, and `head_faces`.

```bash
./.venv/bin/hierreco-graphml2samples \
  path/to/component_or_graphml \
  json_data
```

With an explicit PGM image for Zernike features:

```bash
./.venv/bin/hierreco-graphml2samples \
  path/to/graphs.graphml \
  json_data \
  --pgm-path path/to/image.pgm
```

Relevant export options:

```text
--min-nodes N          Minimum component size, default 5
--zernike-degree N     Maximum Zernike moment degree
--zernike-size N       Square mask size used before Zernike computation
--no-zernike           Skip Zernike blob features
```

## Reproduction Workflow

Commands are intended to be run from the repository root.

### 1. Inspect Dataset Statistics

```bash
./.venv/bin/hierreco-compute-stats --config config.toml
```

Useful options:

```text
--bin-count N
--train-bins 0,1,2
--validation-bins 3,4
--candidate-density FLOAT
--split-sensitivity FLOAT
--edge-length-strictness FLOAT
--require-target-coverage
```

### 2. Visualize Candidate Graphs

Single sample:

```bash
./.venv/bin/hierreco-visualize-dataset \
  --config config.toml \
  --index 0
```

Batch rendering:

```bash
./.venv/bin/hierreco-visualize-dataset \
  --config config.toml \
  --all \
  --output visualizations/candidate_graphs
```

### 3. Train

```bash
./.venv/bin/hierreco-train \
  --config config.toml \
  --run-dir runs/fold_0_val_0_1 \
  --bin-count 5 \
  --cv-val-folds 0 1 \
  --device mps \
  --decoder ilp \
  --no-shapley
```

`--cv-val-folds 0 1` assigns bins `0` and `1` to validation and uses the
remaining bins for training. The default training setup filters samples by
candidate graph recall, writes run artifacts under `runs/`, and stores the best
validation-loss checkpoint as `best_val_loss.pt`.

Common training options:

```text
--dataset-root PATH
--cache-dir PATH
--no-cache
--rebuild-cache
--batch-size N
--epochs N
--lr FLOAT
--weight-decay FLOAT
--hidden-dim N
--layers N
--gnn-branch gine|gatv2
--dropout FLOAT
--threshold FLOAT
--min-candidate-recall FLOAT
--candidate-density FLOAT
--split-sensitivity FLOAT
--edge-length-strictness FLOAT
--decoder greedy|ilp
--device cpu|mps|auto|cuda|cuda:0
```

### 4. Resume Training

```bash
./.venv/bin/hierreco-train \
  --config config.toml \
  --resume-run runs/fold_0_val_0_1 \
  --device mps \
  --decoder ilp \
  --no-shapley
```

`--resume-run RUN_DIR` is equivalent to using `--run-dir RUN_DIR` and
`--resume-from RUN_DIR/best_val_loss.pt`.

### 5. Export Evaluation Artifacts

Per-sample graph JSON export:

```bash
./.venv/bin/hierreco-export-graphs \
  --config config.toml \
  runs/fold_0_val_0_1 \
  --output-dir exports/fold_0_val_0_1_graphs \
  --split all \
  --decoder ilp \
  --device mps
```

Decoded edge error CSV:

```bash
./.venv/bin/hierreco-export-edge-errors \
  --config config.toml \
  runs/fold_0_val_0_1 \
  --output exports/fold_0_val_0_1_edge_errors.csv \
  --edge-types FP,FN \
  --decoder ilp \
  --device mps
```

### 6. Visualize Predictions

Interactive worst-first browser:

```bash
./.venv/bin/hierreco-visualize-run \
  --config config.toml \
  runs/fold_0_val_0_1 \
  --decoder ilp \
  --device mps
```

PNG export:

```bash
./.venv/bin/hierreco-visualize-run \
  --config config.toml \
  runs/fold_0_val_0_1 \
  --save \
  --output-dir visualizations/fold_0_val_0_1_predictions \
  --limit-validation 25 \
  --decoder ilp \
  --device mps
```

Exported graph JSON visualization without model loading:

```bash
./.venv/bin/hierreco-visualize-evaluated \
  exports/fold_0_val_0_1_graphs \
  --all \
  --output-dir visualizations/fold_0_val_0_1_exported
```

### 7. Compute Grouped Shapley Explanations

Training runs grouped Shapley automatically unless `--no-shapley` is set.
Manual execution:

```bash
./.venv/bin/hierreco-shapley \
  runs/fold_0_val_0_1 \
  --checkpoint runs/fold_0_val_0_1/best_val_loss.pt \
  --output runs/fold_0_val_0_1/shapley_explainability.json \
  --mode exact \
  --device mps
```

Relevant options:

```text
--mode exact|permutation
--permutations N
--subset-batch-size N
--threshold FLOAT
--max-train-samples N
--max-validation-samples N
--min-candidate-recall FLOAT
```

The default target score is thresholded undirected edge F1 without path
decoding. Exact mode evaluates all feature-group subsets. Permutation mode uses
Monte Carlo permutations.

## Method Summary

### Data Normalization

GraphML connected components are exported as independent JSON samples. Node
coordinates are transformed according to component orientation, shifted so that
the sequence-origin node is at `(0, 0)`, and stored with original image
coordinates and metadata. Superblob groups are merged before export.

### Candidate Graph

Candidate edges are generated from node coordinates and graph type only. Target
annotations are used after graph construction for labels and coverage checks.

The candidate graph is built as an undirected graph and optionally expanded to
both directions for PyTorch Geometric message passing. Edge rules include:

- bounded radius-kNN edges within detected substructures
- sparse nearest neighbors in angular sectors
- local empty-triangle edges
- local diagonal edges across sparse axis-aligned boxes
- first-hit directional envelope sweeps along the reading direction
- short forward neighbors along the reading axis
- component bridges inside disconnected substructures
- boundary-node links between adjacent substructure bounding boxes

Graph-construction thresholds are adaptive. Per-sample geometry statistics are
combined with a coordinate profile fitted on the configured training bins.
Validation samples use the fitted profile but do not contribute to it.

### Model

`EdgeNodeGNN` embeds node and edge features with MLP encoders, propagates node
context over the directed candidate graph, and scores each candidate directed
edge. The default message-passing branch is edge-conditioned GINE. GATv2 is
available for ablations.

For each directed candidate edge, the classifier combines source node
embedding, target node embedding, encoded edge features, absolute source-target
embedding difference, and element-wise source-target embedding product. The
combined representation is passed through an MLP edge head that produces one
logit per directed edge.

### Loss

The primary objective is binary cross-entropy with logits on directed candidate
edge labels. Optional terms include positive-class weighting, focal weighting,
label smoothing, soft node-degree regularization on undirected probabilities,
and soft edge-count regularization toward `num_nodes - 1`.

Default topology regularization:

```text
bce_loss_weight = 1.0
degree_loss_weight = 0.02
edge_count_loss_weight = 0.01
```

### Decoding

Raw thresholded edge probabilities are evaluated directly as edge-classification
metrics. Decoded metrics first convert edge logits into an undirected path:

- `greedy`: high-scoring acyclic degree-2 path construction
- `ilp`: maximum-score Hamiltonian path ILP, with greedy fallback diagnostics

### Metrics

Training and export routines report loss, accuracy, precision, recall,
specificity, balanced accuracy, F1, positive/negative edge counts,
quadrat-restricted metrics, decoded path metrics, candidate graph recall,
degree MAE, edge-count MAE, endpoint-count MAE, decoder fallback rate, and
gradient norm.

## Feature Definitions

### Node Features

Node feature vectors are constructed in this order:

| Feature | Description |
| --- | --- |
| `x` | Raw transformed x-coordinate |
| `y` | Raw transformed y-coordinate |
| `x_norm` | Sample-local x-coordinate normalized to `[0, 1]` |
| `y_norm` | Sample-local y-coordinate normalized to `[0, 1]` |
| `main_axis_rank` | Rank along the reading axis, normalized to `[0, 1]` |
| `local_density` | Mean distance to the nearest local neighbors |
| `dist_to_prev_axis` | Normalized distance to the previous node on the reading axis |
| `dist_to_next_axis` | Normalized distance to the next node on the reading axis |
| `bbox_x_norm` | Sample-local normalized glyph bounding-box x-position |
| `bbox_y_norm` | Sample-local normalized glyph bounding-box y-position |
| `bbox_width_norm` | Glyph bounding-box width normalized by sample-local width scale |
| `bbox_height_norm` | Glyph bounding-box height normalized by sample-local height scale |
| `zernike_moments` | Optional rotation-invariant blob-shape descriptors |

The optional Zernike vector length depends on `--zernike-degree`. Samples in one
dataset must have consistent Zernike vector lengths.

### Edge Features

Edge feature vectors are constructed for every directed candidate edge in this
order:

| Feature | Description |
| --- | --- |
| `dx` | Target x minus source x |
| `dy` | Target y minus source y |
| `abs_dx` | Absolute x displacement |
| `abs_dy` | Absolute y displacement |
| `distance` | Euclidean source-target distance |
| `main_axis_delta` | Signed displacement on the reading axis |
| `cross_axis_delta` | Signed displacement on the cross axis |
| `abs_main_axis_delta` | Absolute reading-axis displacement |
| `abs_cross_axis_delta` | Absolute cross-axis displacement |
| `main_axis_rank_delta` | Normalized rank displacement on the reading axis |
| `points_between_axis` | Number of intermediate reading-axis ranks, normalized |
| `is_forward_axis` | `1.0` when the edge points forward on the reading axis |

### Labels and Metadata

`edge_y` is assigned after candidate generation by matching normalized
candidate edges against `edges_list`. Metadata such as node identifiers, blob
ids, original coordinates, quadrats, component ids, substructure bounding
boxes, and boundary nodes is preserved for diagnostics, exports, and
visualization.

## Outputs

A typical training run directory contains:

```text
runs/fold_0_val_0_1/
|-- best_val_loss.pt
|-- metrics.json
|-- model.py
|-- training.log
|-- training_plot.html
`-- shapley_explainability.json
```

`model.py` is copied into the run directory to keep the checkpoint tied to the
model definition used during training.

`hierreco-export-graphs` writes one JSON file per evaluated sample with sample
identity, split, candidate edges, target edges, decoded prediction edges,
TP/FP/FN edge classes, decoded metrics, and visualization metadata.

`hierreco-export-edge-errors` writes a compact CSV focused on decoded TP/FP/FN
edges and original node coordinates.

## Version-Controlled and Generated Files

Source files, configuration, citation metadata, and preprocessed JSON samples
are intended for version control. Local runtime artifacts are ignored by Git:

```text
.venv/
.cache/
cache/
runs/
exports/
visualizations/
*.pt
*.pth
*.ckpt
*.onnx
*.log
```

Trained weights and generated outputs belong in explicit release artifacts when
needed for archival use. Release notes record the corresponding `config.toml`,
dependency versions, run command, and checkpoint path.

## Development

```bash
make check
make lint
make stats
```

`make check` runs Python bytecode compilation and Ruff checks. `make lint`
runs Ruff only. `make stats` runs the dataset statistics entry point against
the configured dataset.
