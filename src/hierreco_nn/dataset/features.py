import math

import torch


class FeatureMixin:
    """Build node and edge feature tensors from coordinates and JSON metadata."""

    def make_edge_features(self, points, edges, *, graph_type):
        """Build geometric edge features aligned with a directed edge list.

        Feature order:
        ``dx``, ``dy``, ``abs_dx``, ``abs_dy``, ``distance``,
        ``main_axis_delta``, ``cross_axis_delta``, ``abs_main_axis_delta``,
        ``abs_cross_axis_delta``, ``main_axis_rank_delta``,
        ``points_between_axis``, ``is_forward_axis``.
        """
        main_axis = self.main_axis(graph_type)
        cross_axis = 1 - main_axis
        main_ranks = self.integer_axis_ranks(points, main_axis)
        max_rank_delta = max(len(points) - 1, 1)
        edge_features = []

        for source, target in edges:
            dx = points[target][0] - points[source][0]
            dy = points[target][1] - points[source][1]
            distance = math.hypot(dx, dy)
            main_axis_delta = points[target][main_axis] - points[source][main_axis]
            cross_axis_delta = points[target][cross_axis] - points[source][cross_axis]
            main_axis_rank_delta = (
                main_ranks[target] - main_ranks[source]
            ) / max_rank_delta
            points_between_axis = max(
                abs(main_ranks[target] - main_ranks[source]) - 1,
                0,
            ) / max_rank_delta
            is_forward_axis = 1.0 if main_axis_delta > 0 else 0.0

            edge_features.append(
                [
                    dx,
                    dy,
                    abs(dx),
                    abs(dy),
                    distance,
                    main_axis_delta,
                    cross_axis_delta,
                    abs(main_axis_delta),
                    abs(cross_axis_delta),
                    main_axis_rank_delta,
                    points_between_axis,
                    is_forward_axis,
                ]
            )

        return edge_features

    @staticmethod
    def edge_features_to_tensor(edge_features):
        """Convert edge feature rows to a stable ``[num_edges, 12]`` tensor."""
        if not edge_features:
            return torch.empty((0, 12), dtype=torch.float32)

        return torch.tensor(edge_features, dtype=torch.float32)

    def make_node_features(self, points, *, graph_type, sample=None, node_ids=None):
        """Build node features from raw positions.

        Feature order:
        ``x``, ``y``, ``x_norm``, ``y_norm``, ``main_axis_rank``,
        ``local_density``, ``dist_to_prev_axis``, ``dist_to_next_axis``,
        ``bbox_x_norm``, ``bbox_y_norm``, ``bbox_width_norm``,
        ``bbox_height_norm``, then optional ``zernike_moments`` from the JSON
        sample.
        """
        if not points:
            return []

        x_norm, y_norm = self.normalized_coordinates(points)
        main_axis = self.main_axis(graph_type)
        main_axis_rank = self.axis_ranks(points, main_axis)
        local_density = self.local_density(points, k=3)
        dist_to_prev_axis, dist_to_next_axis = self.axis_neighbor_distances(
            points,
            main_axis,
        )
        bbox_features = self.normalized_node_bbox_features(
            sample,
            node_ids,
            len(points),
        )
        zernike_features = self.node_zernike_features(
            sample,
            node_ids,
            len(points),
        )

        return [
            [
                points[index][0],
                points[index][1],
                x_norm[index],
                y_norm[index],
                main_axis_rank[index],
                local_density[index],
                dist_to_prev_axis[index],
                dist_to_next_axis[index],
                *bbox_features[index],
                *zernike_features[index],
            ]
            for index in range(len(points))
        ]

    @staticmethod
    def node_zernike_features(sample, node_ids, node_count):
        """Return per-node Zernike moment vectors stored in the JSON sample."""
        if sample is None or node_ids is None:
            return [[] for _ in range(node_count)]

        feature_map = sample.get("feature", {})
        vectors = [
            [
                float(value)
                for value in feature_map.get(node_id, {}).get("zernike_moments", [])
            ]
            for node_id in node_ids
        ]

        lengths = {len(vector) for vector in vectors}
        if len(lengths) > 1:
            raise ValueError(
                "Inconsistent zernike_moments vector lengths in one sample: "
                f"{sorted(lengths)}"
            )

        return vectors

    @staticmethod
    def normalized_node_bbox_features(sample, node_ids, node_count):
        """Return normalized ``bbox_x/y/width/height`` features per node.

        Raw image coordinates are converted to sample-local scales before being
        mixed with position-derived features.
        """
        if sample is None or node_ids is None:
            return [[0.0, 0.0, 0.0, 0.0] for _ in range(node_count)]

        feature_map = sample.get("feature", {})
        raw_boxes = []

        for node_id in node_ids:
            node_feature = feature_map.get(node_id, {})
            bbox_x = FeatureMixin.safe_float(node_feature.get("bbox_x"))
            bbox_y = FeatureMixin.safe_float(node_feature.get("bbox_y"))
            bbox_width = FeatureMixin.safe_float(node_feature.get("bbox_width"))
            bbox_height = FeatureMixin.safe_float(node_feature.get("bbox_height"))
            raw_boxes.append((bbox_x, bbox_y, bbox_width, bbox_height))

        min_x = min((bbox_x for bbox_x, _, _, _ in raw_boxes), default=0.0)
        min_y = min((bbox_y for _, bbox_y, _, _ in raw_boxes), default=0.0)
        max_x = max(
            (bbox_x + bbox_width for bbox_x, _, bbox_width, _ in raw_boxes),
            default=0.0,
        )
        max_y = max(
            (bbox_y + bbox_height for _, bbox_y, _, bbox_height in raw_boxes),
            default=0.0,
        )
        max_width = max((bbox_width for _, _, bbox_width, _ in raw_boxes), default=0.0)
        max_height = max((bbox_height for _, _, _, bbox_height in raw_boxes), default=0.0)
        sample_width = max(max_x - min_x, 1e-9)
        sample_height = max(max_y - min_y, 1e-9)
        width_scale = max(max_width, 1e-9)
        height_scale = max(max_height, 1e-9)

        return [
            [
                (bbox_x - min_x) / sample_width,
                (bbox_y - min_y) / sample_height,
                bbox_width / width_scale,
                bbox_height / height_scale,
            ]
            for bbox_x, bbox_y, bbox_width, bbox_height in raw_boxes
        ]

    @staticmethod
    def safe_float(value):
        """Convert a JSON scalar to float, using zero for missing values."""
        if value is None:
            return 0.0
        return float(value)
