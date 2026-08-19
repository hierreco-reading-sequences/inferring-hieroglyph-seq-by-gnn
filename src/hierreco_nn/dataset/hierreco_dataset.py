from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .candidate_graph import CandidateGraphMixin
from .features import FeatureMixin
from .geometry import GeometryMixin
from .io import SampleIOMixin
from .parameters import ParameterMixin
from .substructures import SubstructureMixin


class HierrecoDataset(
    FeatureMixin,
    SampleIOMixin,
    CandidateGraphMixin,
    ParameterMixin,
    SubstructureMixin,
    GeometryMixin,
    Dataset,
):
    """PyTorch Geometric dataset for normalized hieroglyph position samples."""

    def __init__(
        self,
        root,
        *,
        candidate_density: float = 1.0,
        split_sensitivity: float = 1.0,
        edge_length_strictness: float = 1.0,
        triangle_min_angle_degrees: float = 30.0,
        min_sub_nodes: int = 2,
        min_nodes: int = 5,
        bin_count: int = 5,
        split_seed: int | None = 13,
        train_bins=None,
        validation_bins=None,
        add_reverse_edges: bool = True,
        require_target_coverage: bool = True,
    ):
        """Collect usable JSON samples and configure graph construction.

        Args:
            root: Directory searched recursively for ``*sample_*.json`` files.
            candidate_density: Global density knob for local candidates.
                Larger values increase k-NN seeds, angular sectors,
                empty-triangle neighborhood size, and axis-window width.
            split_sensitivity: Global split knob for substructure detection.
                Larger values lower split thresholds and make additional
                substructures easier to create.
            edge_length_strictness: Global internal-edge length knob. Larger
                values shorten length-limited internal candidates; smaller
                values allow longer internal edges.
            triangle_min_angle_degrees: Minimum internal angle for empty
                triangles. Raising it removes thin, long triangle candidates.
            min_sub_nodes: Minimum number of nodes needed for an independent
                substructure cluster. Smaller clusters are merged into the
                closest valid cluster.
            min_nodes: Minimum number of nodes required for a sample to be used.
            bin_count: Number of deterministic bins assigned by sorted sample
                order after seed-controlled shuffling. With the default ``5``,
                bins ``0..2`` are training and bins ``3..4`` are validation.
            split_seed: Seed used to deterministically shuffle sample order
                before assigning bins. Use ``None`` to keep sorted order.
            train_bins: Bin ids used to fit coordinate statistics. If omitted,
                all bins except the default validation bins are used.
            validation_bins: Bin ids reserved for validation. If omitted, the
                final two bins are used when possible while leaving at least
                one training bin.
            add_reverse_edges: If true, every undirected edge is emitted in both
                directions for PyG message passing.
            require_target_coverage: If true, raise an error when the generated
                candidate graph does not contain all target edges.

        Candidate-edge generation is target-independent: it uses only
        coordinates and graph type. The coordinate profile is fitted on training
        bins and reused for all samples.
        """
        self.root = Path(root)
        self.candidate_density = float(candidate_density)
        self.split_sensitivity = float(split_sensitivity)
        self.edge_length_strictness = float(edge_length_strictness)
        self.triangle_min_angle_degrees = float(triangle_min_angle_degrees)
        self.min_sub_nodes = int(min_sub_nodes)
        self.min_nodes = int(min_nodes)
        self.bin_count = self.validate_bin_count(bin_count)
        self.split_seed = None if split_seed is None else int(split_seed)
        self.add_reverse_edges = bool(add_reverse_edges)
        self.require_target_coverage = bool(require_target_coverage)

        self.sample_paths = [
            path
            for path in sorted(self.root.rglob("*sample_*.json"))
            if not self.is_hidden_path(path) and self.sample_has_min_nodes(path)
        ]

        if not self.sample_paths:
            raise FileNotFoundError(
                f"No *sample_*.json files with at least {self.min_nodes} nodes "
                f"found under {self.root}"
            )

        self.sample_paths = self.shuffle_sample_paths(
            self.sample_paths,
            self.split_seed,
        )
        self.train_bins, self.validation_bins = self.resolve_split_bins(
            self.bin_count,
            train_bins=train_bins,
            validation_bins=validation_bins,
        )
        self.sample_bins = [
            sample_index % self.bin_count
            for sample_index in range(len(self.sample_paths))
        ]
        self.train_indices = self.indices_for_bins(self.train_bins)
        self.validation_indices = self.indices_for_bins(self.validation_bins)
        self.split_indices = {
            "train": self.train_indices,
            "validation": self.validation_indices,
            "val": self.validation_indices,
            "all": tuple(range(len(self.sample_paths))),
        }

        if not self.train_indices:
            raise ValueError("train_bins must select at least one sample")

        self.geometry_profile = self.build_geometry_profile(
            [self.sample_paths[index] for index in self.train_indices]
        )

    def __len__(self):
        """Return the number of samples that passed the minimum-node filter."""
        return len(self.sample_paths)

    def __getitem__(self, index):
        """Load one JSON sample and convert it to a PyG ``Data`` object."""
        sample_path = self.sample_paths[index]

        with sample_path.open("r", encoding="utf-8") as file:
            sample = json.load(file)

        return self.sample_to_data(
            sample,
            sample_path=sample_path,
            sample_index=index,
        )

    def sample_to_data(self, sample, *, sample_path=None, sample_index=None):
        """Convert one parsed sample dictionary into a PyG ``Data`` object.

        Target information is used only after candidate edges are generated:
        it builds ``target_edge_index`` and ``edge_y``. Candidate generation is
        called with coordinates and graph_type only.
        """
        node_ids = list(sample["nodes_array"])
        node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
        graph_type = sample.get("graph_type")
        component_id = self.make_component_id(sample)
        blob_ids = self.make_blob_ids(sample, node_ids)
        original_positions = self.make_original_positions(sample, node_ids)
        quadrats = self.make_quadrats(sample, node_ids)
        points = self.make_points(sample, node_ids)
        node_features = self.make_node_features(
            points,
            graph_type=graph_type,
            sample=sample,
            node_ids=node_ids,
        )

        # Target edges are created only for supervision and evaluation.
        target_edges = self.make_target_edges(sample["edges_list"], node_to_index)

        candidate_result = self.make_candidate_edges(
            points,
            graph_type=graph_type,
            return_substructures=True,
            return_x_diagonal_edges=True,
        )
        candidate_edges, substructures, x_diagonal_edges = candidate_result

        self.validate_target_coverage(
            candidate_edges,
            target_edges,
            sample_path=sample_path,
        )

        target_set = set(target_edges)
        input_edges = self.make_graph_edges(candidate_edges)
        target_graph_edges = self.make_graph_edges(target_edges)
        edge_attr = self.make_edge_features(
            points,
            input_edges,
            graph_type=graph_type,
        )
        edge_y = torch.tensor(
            [
                1.0 if self.normalize_edge(source, target) in target_set else 0.0
                for source, target in input_edges
            ],
            dtype=torch.float32,
        )

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            pos=torch.tensor(points, dtype=torch.float32),
            edge_index=self.edges_to_index(input_edges),
            edge_attr=self.edge_features_to_tensor(edge_attr),
            target_edge_index=self.edges_to_index(target_graph_edges),
            edge_y=edge_y,
        )
        data.node_ids = node_ids
        data.blob_ids = blob_ids
        data.original_positions = original_positions
        data.quadrat = torch.tensor(quadrats, dtype=torch.long)
        data.component_id = component_id
        data.graph_type = graph_type

        sub_id_by_node = [-1 for _ in points]
        boundary_mask = [False for _ in points]
        sub_bboxes = []
        sub_boundary_nodes = []

        for sub in substructures:
            sub_bboxes.append(list(sub.bbox))
            sub_boundary_nodes.append(list(sub.boundary_nodes))
            for node in sub.nodes:
                sub_id_by_node[node] = sub.sub_id
            for node in sub.boundary_nodes:
                boundary_mask[node] = True

        data.sub_id = torch.tensor(sub_id_by_node, dtype=torch.long)
        data.subline_boundary_mask = torch.tensor(boundary_mask, dtype=torch.bool)
        data.sub_bboxes = (
            torch.tensor(sub_bboxes, dtype=torch.float32)
            if sub_bboxes
            else torch.empty((0, 4), dtype=torch.float32)
        )
        data.sub_boundary_nodes = sub_boundary_nodes
        data.x_diagonal_edge_index = self.edges_to_index(
            self.make_graph_edges(x_diagonal_edges)
        )

        if sample_path is not None:
            data.sample_path = str(sample_path)
        if sample_index is not None:
            data.sample_index = int(sample_index)
            data.bin_id = int(self.sample_bins[sample_index])
            data.split_name = self.split_name_for_index(sample_index)

        return data

    @staticmethod
    def validate_bin_count(bin_count):
        """Return a valid positive integer bin count."""
        bin_count = int(bin_count)
        if bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {bin_count}")
        return bin_count

    @staticmethod
    def shuffle_sample_paths(sample_paths, split_seed):
        """Return sample paths shuffled deterministically before binning."""
        sample_paths = list(sample_paths)
        if split_seed is None:
            return sample_paths

        rng = random.Random(split_seed)
        rng.shuffle(sample_paths)
        return sample_paths

    @classmethod
    def resolve_split_bins(cls, bin_count, *, train_bins=None, validation_bins=None):
        """Resolve training and validation bin ids for profile fitting."""
        if validation_bins is None:
            if bin_count <= 1:
                validation_bins = ()
            else:
                validation_bin_count = min(2, bin_count - 1)
                validation_bins = tuple(
                    range(bin_count - validation_bin_count, bin_count)
                )
        else:
            validation_bins = cls.normalize_bins(validation_bins, bin_count)

        if train_bins is None:
            validation_set = set(validation_bins)
            train_bins = tuple(
                bin_id
                for bin_id in range(bin_count)
                if bin_id not in validation_set
            )
            if not train_bins:
                train_bins = tuple(range(bin_count))
        else:
            train_bins = cls.normalize_bins(train_bins, bin_count)

        overlap = set(train_bins) & set(validation_bins)
        if overlap:
            raise ValueError(
                f"train_bins and validation_bins overlap: {sorted(overlap)}"
            )

        return tuple(train_bins), tuple(validation_bins)

    @staticmethod
    def normalize_bins(bins, bin_count):
        """Normalize a bin specification into sorted unique integer ids."""
        if isinstance(bins, int):
            raw_bins = [bins]
        elif isinstance(bins, str):
            raw_bins = [
                part.strip()
                for part in bins.split(",")
                if part.strip()
            ]
        else:
            raw_bins = list(bins)

        normalized = tuple(sorted({int(bin_id) for bin_id in raw_bins}))
        invalid = [
            bin_id
            for bin_id in normalized
            if bin_id < 0 or bin_id >= bin_count
        ]
        if invalid:
            raise ValueError(
                f"bin ids must be in 0..{bin_count - 1}, got {invalid}"
            )
        return normalized

    def indices_for_bins(self, bins):
        """Return sample indices whose deterministic bin id is selected."""
        selected_bins = set(bins)
        return tuple(
            index
            for index, bin_id in enumerate(self.sample_bins)
            if bin_id in selected_bins
        )

    def split_name_for_index(self, index):
        """Return the configured split label for one sample index."""
        bin_id = self.sample_bins[index]
        if bin_id in self.train_bins:
            return "train"
        if bin_id in self.validation_bins:
            return "validation"
        return "unused"
