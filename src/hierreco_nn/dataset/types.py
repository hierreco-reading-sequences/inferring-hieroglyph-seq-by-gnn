from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Sequence[float]
Edge = Tuple[int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Substructure:
    """A detected sequence sub-row/sub-column."""

    sub_id: int
    nodes: Tuple[int, ...]
    bbox: BBox
    boundary_nodes: Tuple[int, ...]
    cross_center: float
    main_min: float
    main_max: float


@dataclass(frozen=True)
class GraphGeometryStats:
    """Per-sample coordinate statistics used to derive graph parameters."""

    node_count: int
    main_axis: int
    cross_axis: int
    bbox_diagonal: float
    median_nn: float
    q90_nn: float
    median_main_gap: float
    q75_main_gap: float
    median_cross_gap: float
    q75_cross_gap: float


@dataclass(frozen=True)
class DatasetGeometryProfile:
    """Coordinate profile fitted on the configured training bins only."""

    sample_count: int
    median_sqrt_nodes: float
    q75_sqrt_nodes: float
    q90_sqrt_nodes: float
    median_spacing_spread: float
    q75_spacing_spread: float
    q90_spacing_spread: float
    median_main_q75_ratio: float
    q25_main_q75_ratio: float
    q75_main_q75_ratio: float
    median_cross_q75_ratio: float
    q75_cross_q75_ratio: float
    q90_cross_q75_ratio: float
    median_cell_diagonal_ratio: float
    q75_cell_diagonal_ratio: float
    q90_cell_diagonal_ratio: float
    median_empty_box_length_ratio: float
    q75_empty_box_length_ratio: float
    q90_empty_box_length_ratio: float
    median_bbox_ratio: float


@dataclass(frozen=True)
class CandidateGraphParams:
    """Concrete graph-construction parameters derived for one sample."""

    candidate_k: int
    candidate_radius_factor: float
    directional_neighbor_sectors: int
    directional_neighbors_per_sector: int
    empty_triangle_neighbor_k: int
    empty_triangle_min_angle_degrees: float
    empty_box_neighbor_k: int
    empty_box_max_length: float
    empty_box_min_axis_fraction: float
    empty_box_max_inside: int
    envelope_sweep_cross_window: float
    envelope_sweep_front_window: float
    axis_window: int
    cross_cluster_gap_factor: float
    cross_cluster_min_abs_factor: float
    main_axis_subsplit_gap_factor: float
    main_axis_subsplit_min_abs_factor: float
    main_axis_subsplit_neighbor_gap_ratio: float
    narrow_sub_merge_gap_factor: float
    narrow_sub_merge_width_ratio: float
    narrow_sub_merge_max_width_factor: float
    min_sub_main_overlap_ratio: float
    min_sub_main_overlap_abs_factor: float
    max_edge_length: float
