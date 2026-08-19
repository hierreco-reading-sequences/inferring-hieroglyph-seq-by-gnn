import json

import torch


class SampleIOMixin:
    """Read JSON samples and convert edge lists to tensor formats."""

    def validate_target_coverage(self, candidate_edges, target_edges, *, sample_path):
        """Ensure every target edge is present among generated candidate edges."""
        missing_edges = sorted(set(target_edges) - set(candidate_edges))

        if not missing_edges or not self.require_target_coverage:
            return

        location = f" in {sample_path}" if sample_path is not None else ""
        raise ValueError(
            "Candidate graph does not contain all target edges"
            f"{location}. Missing target edges: {missing_edges}"
        )

    def sample_has_min_nodes(self, sample_path):
        """Return true when a JSON sample has at least ``self.min_nodes`` nodes."""
        with sample_path.open("r", encoding="utf-8") as file:
            sample = json.load(file)

        return len(sample["nodes_array"]) >= self.min_nodes

    def is_hidden_path(self, path):
        """Return true when a path or any relative parent starts with a dot."""
        relative_path = path.relative_to(self.root)
        return any(part.startswith(".") for part in relative_path.parts)

    @staticmethod
    def make_points(sample, node_ids):
        """Extract raw ``[x, y]`` coordinates in ``nodes_array`` order."""
        features = sample["feature"]
        return [
            [
                float(features[node_id]["x"]),
                float(features[node_id]["y"]),
            ]
            for node_id in node_ids
        ]

    @staticmethod
    def make_blob_ids(sample, node_ids):
        """Extract per-node blob identifiers as metadata, not model features."""
        features = sample.get("feature", {})
        return [
            str(features.get(node_id, {}).get("blob_id", node_id))
            for node_id in node_ids
        ]

    @staticmethod
    def make_original_positions(sample, node_ids):
        """Extract original image coordinates as metadata, not model features."""
        features = sample.get("feature", {})
        positions = []

        for node_id in node_ids:
            node_feature = features.get(node_id, {})
            original_x = node_feature.get("original_x")
            original_y = node_feature.get("original_y")
            positions.append(
                (
                    None if original_x is None else float(original_x),
                    None if original_y is None else float(original_y),
                )
            )

        return positions

    @staticmethod
    def make_quadrats(sample, node_ids):
        """Extract per-node quadrat ids as metadata, not model features."""
        features = sample.get("feature", {})
        return [
            int(features.get(node_id, {}).get("quadrat", 0) or 0)
            for node_id in node_ids
        ]

    @staticmethod
    def make_component_id(sample):
        """Extract the sample-level component identifier as metadata."""
        component_id = sample.get("component_id")
        return "" if component_id is None else str(component_id)

    @classmethod
    def make_target_edges(cls, edges_list, node_to_index):
        """Map JSON node-id edges to sorted zero-based undirected index edges."""
        edges = set()

        for source_id, target_id in edges_list:
            if source_id not in node_to_index or target_id not in node_to_index:
                continue

            source = node_to_index[source_id]
            target = node_to_index[target_id]
            if source != target:
                edges.add(cls.normalize_edge(source, target))

        return sorted(edges)

    def make_graph_edges(self, edges):
        """Convert undirected edges to the edge list used in PyG tensors."""
        if not self.add_reverse_edges:
            return edges

        directed_edges = []
        for source, target in edges:
            directed_edges.append((source, target))
            directed_edges.append((target, source))

        return directed_edges

    @staticmethod
    def edges_to_index(edges):
        """Convert edge list to PyG ``edge_index`` shape ``[2, num_edges]``."""
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()
