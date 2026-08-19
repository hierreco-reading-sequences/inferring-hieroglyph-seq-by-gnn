import json
import math

from .types import CandidateGraphParams, DatasetGeometryProfile, GraphGeometryStats


class ParameterMixin:
    """Derive per-sample candidate-graph parameters from geometry statistics."""

    def build_geometry_profile(self, sample_paths):
        """Fit coordinate statistics from training-bin samples only.

        The profile ignores target annotations and stores coordinate scales
        used for fixed train/validation candidate-graph calibration.
        """
        values = {
            "sqrt_nodes": [],
            "spacing_spread": [],
            "main_q75_ratio": [],
            "cross_q75_ratio": [],
            "cell_diagonal_ratio": [],
            "empty_box_length_ratio": [],
            "bbox_ratio": [],
        }

        for sample_path in sample_paths:
            with sample_path.open("r", encoding="utf-8") as file:
                sample = json.load(file)

            node_ids = list(sample["nodes_array"])
            points = self.make_points(sample, node_ids)
            graph_type = sample.get("graph_type")
            stats = self.graph_geometry_stats(
                points,
                main_axis=self.main_axis(graph_type),
            )
            median_nn = max(stats.median_nn, self.numeric_epsilon())
            cell_diagonal = self.local_cell_diagonal(stats)
            empty_box_length = self.local_empty_box_length(stats)

            values["sqrt_nodes"].append(math.sqrt(max(stats.node_count, 1)))
            values["spacing_spread"].append(stats.q90_nn / median_nn)
            values["main_q75_ratio"].append(stats.q75_main_gap / median_nn)
            values["cross_q75_ratio"].append(stats.q75_cross_gap / median_nn)
            values["cell_diagonal_ratio"].append(cell_diagonal / median_nn)
            values["empty_box_length_ratio"].append(empty_box_length / median_nn)
            values["bbox_ratio"].append(stats.bbox_diagonal / median_nn)

        return DatasetGeometryProfile(
            sample_count=len(sample_paths),
            median_sqrt_nodes=self.median(values["sqrt_nodes"]),
            q75_sqrt_nodes=self.quantile(values["sqrt_nodes"], 0.75),
            q90_sqrt_nodes=self.quantile(values["sqrt_nodes"], 0.90),
            median_spacing_spread=self.median(values["spacing_spread"]),
            q75_spacing_spread=self.quantile(values["spacing_spread"], 0.75),
            q90_spacing_spread=self.quantile(values["spacing_spread"], 0.90),
            median_main_q75_ratio=self.median(values["main_q75_ratio"]),
            q25_main_q75_ratio=self.quantile(values["main_q75_ratio"], 0.25),
            q75_main_q75_ratio=self.quantile(values["main_q75_ratio"], 0.75),
            median_cross_q75_ratio=self.median(values["cross_q75_ratio"]),
            q75_cross_q75_ratio=self.quantile(values["cross_q75_ratio"], 0.75),
            q90_cross_q75_ratio=self.quantile(values["cross_q75_ratio"], 0.90),
            median_cell_diagonal_ratio=self.median(values["cell_diagonal_ratio"]),
            q75_cell_diagonal_ratio=self.quantile(
                values["cell_diagonal_ratio"],
                0.75,
            ),
            q90_cell_diagonal_ratio=self.quantile(
                values["cell_diagonal_ratio"],
                0.90,
            ),
            median_empty_box_length_ratio=self.median(
                values["empty_box_length_ratio"]
            ),
            q75_empty_box_length_ratio=self.quantile(
                values["empty_box_length_ratio"],
                0.75,
            ),
            q90_empty_box_length_ratio=self.quantile(
                values["empty_box_length_ratio"],
                0.90,
            ),
            median_bbox_ratio=self.median(values["bbox_ratio"]),
        )

    def graph_geometry_stats(self, points, *, main_axis):
        """Compute per-graph geometry statistics for adaptive parameters."""
        cross_axis = 1 - main_axis
        node_count = len(points)
        bbox = self.bbox_for_points(points)
        bbox_diagonal = max(self.bbox_diagonal(bbox), 1e-9)
        nearest_distances = []

        for source in range(node_count):
            distances = self.neighbor_distances(points, source)
            if distances and distances[0][0] > 1e-9:
                nearest_distances.append(distances[0][0])

        if nearest_distances:
            median_nn = self.median(nearest_distances)
            q90_nn = self.quantile(nearest_distances, 0.90)
        else:
            median_nn = max(bbox_diagonal, 1.0)
            q90_nn = median_nn

        main_gaps = self.axis_positive_gaps(points, range(node_count), main_axis)
        cross_gaps = self.axis_positive_gaps(points, range(node_count), cross_axis)

        return GraphGeometryStats(
            node_count=node_count,
            main_axis=main_axis,
            cross_axis=cross_axis,
            bbox_diagonal=bbox_diagonal,
            median_nn=max(median_nn, 1e-9),
            q90_nn=max(q90_nn, median_nn, 1e-9),
            median_main_gap=(
                self.median(main_gaps) if main_gaps else max(median_nn, 1e-9)
            ),
            q75_main_gap=(
                self.quantile(main_gaps, 0.75)
                if main_gaps
                else max(median_nn, 1e-9)
            ),
            median_cross_gap=(
                self.median(cross_gaps) if cross_gaps else max(median_nn, 1e-9)
            ),
            q75_cross_gap=(
                self.quantile(cross_gaps, 0.75)
                if cross_gaps
                else max(median_nn, 1e-9)
            ),
        )

    def derive_candidate_graph_params(self, stats):
        """Derive concrete parameters for one graph from statistics and knobs.

        Thresholds combine current-sample statistics with the fitted training
        profile. ``candidate_density`` controls candidate counts,
        ``split_sensitivity`` controls substructure splits, and
        ``edge_length_strictness`` controls internal edge length limits.
        """
        profile = self.geometry_profile
        density = self.positive_knob(self.candidate_density)
        sensitivity = self.positive_knob(self.split_sensitivity)
        strictness = self.positive_knob(self.edge_length_strictness)
        spacing_spread = self.spacing_spread(stats)
        max_edge_length = self.derived_max_edge_length(stats, strictness)
        candidate_k = self.neighbor_count(density * spacing_spread, stats)
        triangle_neighbor_k = self.neighbor_count(
            density * profile.q75_sqrt_nodes,
            stats,
        )
        empty_box_neighbor_k = self.neighbor_count(
            density
            * profile.q90_sqrt_nodes
            * self.safe_ratio(spacing_spread, profile.median_spacing_spread),
            stats,
        )
        directional_sectors = self.neighbor_count(
            density * max(self.sqrt_node_count(stats), profile.q75_sqrt_nodes),
            stats,
        )
        directional_per_sector = self.neighbor_count(
            density * self.safe_ratio(spacing_spread, profile.q90_spacing_spread),
            stats,
        )
        axis_window = self.neighbor_count(density * spacing_spread, stats)

        return CandidateGraphParams(
            candidate_k=candidate_k,
            candidate_radius_factor=max(spacing_spread, self.numeric_epsilon()),
            directional_neighbor_sectors=directional_sectors,
            directional_neighbors_per_sector=directional_per_sector,
            empty_triangle_neighbor_k=triangle_neighbor_k,
            empty_triangle_min_angle_degrees=self.triangle_min_angle_degrees,
            empty_box_neighbor_k=empty_box_neighbor_k,
            empty_box_max_length=self.derived_empty_box_max_length(
                stats,
                strictness,
                max_edge_length,
            ),
            empty_box_min_axis_fraction=self.empty_box_min_axis_fraction(profile),
            empty_box_max_inside=self.empty_box_inside_count(
                stats,
                empty_box_neighbor_k,
            ),
            envelope_sweep_cross_window=(
                self.derived_envelope_sweep_cross_window(stats)
            ),
            envelope_sweep_front_window=(
                self.derived_envelope_sweep_front_window(stats)
            ),
            axis_window=axis_window,
            cross_cluster_gap_factor=self.scaled_by_sensitivity(
                profile.q90_empty_box_length_ratio,
                sensitivity,
            ),
            cross_cluster_min_abs_factor=self.scaled_by_sensitivity(
                profile.q25_main_q75_ratio,
                sensitivity,
            ),
            main_axis_subsplit_gap_factor=self.scaled_by_sensitivity(
                profile.q90_empty_box_length_ratio,
                sensitivity,
            ),
            main_axis_subsplit_min_abs_factor=self.scaled_by_sensitivity(
                profile.q25_main_q75_ratio,
                sensitivity,
            ),
            main_axis_subsplit_neighbor_gap_ratio=self.scaled_by_sensitivity(
                self.safe_ratio(
                    profile.q90_empty_box_length_ratio,
                    profile.median_spacing_spread,
                ),
                sensitivity,
            ),
            narrow_sub_merge_gap_factor=self.scaled_by_sensitivity(
                profile.q90_spacing_spread,
                sensitivity,
            ),
            narrow_sub_merge_width_ratio=self.safe_ratio(
                profile.q90_cross_q75_ratio,
                profile.median_main_q75_ratio,
            ),
            narrow_sub_merge_max_width_factor=self.scaled_by_sensitivity(
                profile.q25_main_q75_ratio,
                sensitivity,
            ),
            min_sub_main_overlap_ratio=self.safe_ratio(
                profile.q90_cross_q75_ratio,
                profile.median_main_q75_ratio,
            ),
            min_sub_main_overlap_abs_factor=self.scaled_by_sensitivity(
                profile.q25_main_q75_ratio,
                sensitivity,
            ),
            max_edge_length=max_edge_length,
        )

    def derived_max_edge_length(self, stats, strictness):
        """Return dynamic maximum length for internal candidate edges."""
        return max(
            stats.median_nn,
            max(
                self.local_cell_diagonal(stats),
                stats.median_nn * self.geometry_profile.q90_empty_box_length_ratio,
            )
            / math.sqrt(strictness),
        )

    def derived_empty_box_max_length(self, stats, strictness, max_edge_length):
        """Return dynamic maximum length for empty-box diagonal candidates."""
        profile = self.geometry_profile
        empty_box_limit = max(
            self.local_empty_box_length(stats) * profile.q90_spacing_spread,
            stats.median_nn * profile.q90_empty_box_length_ratio,
        ) / math.sqrt(strictness)
        return max(max_edge_length, empty_box_limit)

    def derived_envelope_sweep_cross_window(self, stats):
        """Return local cross-axis width for first-hit envelope sweeps."""
        return max(
            stats.q90_nn,
            stats.q75_cross_gap,
            stats.median_nn * self.geometry_profile.q90_empty_box_length_ratio,
        )

    def derived_envelope_sweep_front_window(self, stats):
        """Return main-axis tolerance for grouping one sweep hit-front."""
        return max(stats.median_main_gap, self.numeric_epsilon())

    def local_cell_diagonal(self, stats):
        """Return a local cell diagonal inferred from axis-gap statistics."""
        return math.hypot(
            max(stats.q75_main_gap, stats.median_nn),
            max(stats.q75_cross_gap, stats.median_nn),
        )

    def local_empty_box_length(self, stats):
        """Return a local diagonal reach for empty-box candidates."""
        return self.local_cell_diagonal(stats) + stats.q90_nn

    def spacing_spread(self, stats):
        """Return q90/median nearest-neighbor spread for one sample."""
        return self.safe_ratio(stats.q90_nn, stats.median_nn)

    def sample_geometry_ratios(self, stats):
        """Return scale-free ratios computed from the current sample only."""
        return {
            "spacing_spread": self.spacing_spread(stats),
            "main_gap_ratio": self.safe_ratio(stats.q75_main_gap, stats.median_nn),
            "cross_gap_ratio": self.safe_ratio(stats.q75_cross_gap, stats.median_nn),
            "cell_diagonal_ratio": self.safe_ratio(
                self.local_cell_diagonal(stats),
                stats.median_nn,
            ),
            "empty_box_length_ratio": self.safe_ratio(
                self.local_empty_box_length(stats),
                stats.median_nn,
            ),
        }

    @staticmethod
    def sqrt_node_count(stats):
        """Return a smooth sample-size term for neighborhood sizes."""
        return math.sqrt(max(stats.node_count, 1))

    @staticmethod
    def numeric_epsilon():
        """Return the small positive value used only for numeric stability."""
        return 1e-9

    def positive_knob(self, value):
        """Convert a public knob into a positive scale factor."""
        return max(float(value), self.numeric_epsilon())

    def safe_ratio(self, numerator, denominator):
        """Return a ratio guarded against zero denominators."""
        return float(numerator) / max(float(denominator), self.numeric_epsilon())

    def neighbor_count(self, value, stats):
        """Convert a data-derived continuous value to a valid neighbor count."""
        max_neighbors = max(stats.node_count - 1, 0)
        if max_neighbors == 0:
            return 0
        return min(max_neighbors, max(1, math.ceil(float(value))))

    def empty_box_inside_count(self, stats, neighbor_count):
        """Return allowed inner points for empty-box candidates."""
        empty_box_length = self.local_empty_box_length(stats)
        inside_fraction = self.safe_ratio(stats.q90_nn, empty_box_length)
        return max(1, math.ceil(neighbor_count * inside_fraction))

    def empty_box_min_axis_fraction(self, profile):
        """Return minimum diagonal balance from training-bin axis ratios."""
        return self.safe_ratio(
            profile.median_cross_q75_ratio,
            profile.median_main_q75_ratio,
        )

    def scaled_by_sensitivity(self, value, sensitivity):
        """Scale a data-derived threshold by the public split sensitivity."""
        return float(value) / math.sqrt(sensitivity)
