from .types import Substructure


class SubstructureMixin:
    """Detect sequence-aware sub-row/sub-column structures from coordinates."""

    def build_single_substructure(self, points, graph_type):
            """Return one substructure covering all nodes."""
            if not points:
                return []

            graph_type = graph_type if graph_type in ("row", "col") else "col"
            main_axis = self.main_axis(graph_type)
            nodes = tuple(range(len(points)))
            bbox = self.bounding_box(points, nodes)
            boundary_nodes = self.boundary_nodes_for_bbox(points, nodes, bbox)
            return [
                Substructure(
                    sub_id=0,
                    nodes=nodes,
                    bbox=bbox,
                    boundary_nodes=boundary_nodes,
                    cross_center=self.bbox_center(bbox, 1 - main_axis),
                    main_min=bbox[main_axis],
                    main_max=bbox[main_axis + 2],
                )
            ]

    def detect_sequence_substructures(self, points, *, graph_type, stats, params):
            """Detect sequence-aware subcolumns/subrows.

            A substructure is first detected as a parallel lane on the cross axis
            and validated by overlap along the reading/main axis.  Each resulting
            lane is then optionally split by unusually large gaps on the main axis.
            """
            node_count = len(points)
            if node_count <= 1:
                return self.build_single_substructure(points, graph_type)

            main_axis = self.main_axis(graph_type)
            cross_axis = 1 - main_axis
            raw_clusters = self.cluster_by_cross_axis(
                points,
                cross_axis=cross_axis,
                stats=stats,
                params=params,
            )

            if len(raw_clusters) <= 1:
                clusters = [list(range(len(points)))]
            else:
                # Merge undersized clusters into the nearest supported cluster.
                clusters = self.merge_small_cross_clusters(
                    points,
                    raw_clusters,
                    cross_axis=cross_axis,
                    min_nodes=self.min_sub_nodes,
                )
                clusters = self.merge_narrow_adjacent_cross_clusters(
                    points,
                    clusters,
                    cross_axis=cross_axis,
                    stats=stats,
                    params=params,
                )

                if len(clusters) > 1:
                    overlap_pairs = self.valid_main_overlap_pairs(
                        points,
                        clusters,
                        main_axis=main_axis,
                        stats=stats,
                        params=params,
                    )

                    # Without overlapping lanes, keep one full-sequence substructure.
                    if not overlap_pairs:
                        clusters = [list(range(len(points)))]
                    else:
                        # Merge shifted continuations into the nearest overlapping lane.
                        overlapping_cluster_ids = set()
                        for first, second in overlap_pairs:
                            overlapping_cluster_ids.add(first)
                            overlapping_cluster_ids.add(second)

                        clusters = self.merge_non_overlapping_clusters(
                            points,
                            clusters,
                            overlapping_cluster_ids=overlapping_cluster_ids,
                            cross_axis=cross_axis,
                        )

            clusters = self.split_clusters_by_main_axis(
                points,
                clusters,
                main_axis=main_axis,
                stats=stats,
                params=params,
            )

            substructures = []
            ordered_clusters = sorted(
                clusters,
                key=lambda cluster: (
                    self.mean_axis_value(points, cluster, cross_axis),
                    min(cluster),
                ),
            )

            for sub_id, cluster in enumerate(ordered_clusters):
                nodes = tuple(self.sort_by_axis(points, cluster, main_axis))
                bbox = self.bounding_box(points, nodes)
                boundary_nodes = self.boundary_nodes_for_bbox(points, nodes, bbox)
                substructures.append(
                    Substructure(
                        sub_id=sub_id,
                        nodes=nodes,
                        bbox=bbox,
                        boundary_nodes=boundary_nodes,
                        cross_center=self.bbox_center(bbox, cross_axis),
                        main_min=bbox[main_axis],
                        main_max=bbox[main_axis + 2],
                    )
                )

            return substructures

    def cluster_by_cross_axis(self, points, *, cross_axis, stats, params):
            """Cluster possible parallel lanes along the cross axis only."""
            ordered = self.sort_by_axis(points, range(len(points)), cross_axis)
            if len(ordered) <= 1:
                return [ordered]

            values = [points[index][cross_axis] for index in ordered]
            gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            positive_gaps = [gap for gap in gaps if gap > 1e-9]

            if not positive_gaps:
                return [ordered]

            median_gap = self.median(positive_gaps)
            threshold = max(
                median_gap * params.cross_cluster_gap_factor,
                stats.median_nn * params.cross_cluster_min_abs_factor,
                1e-9,
            )

            clusters = []
            current = [ordered[0]]
            for gap, node_index in zip(gaps, ordered[1:]):
                if gap > threshold:
                    clusters.append(current)
                    current = []
                current.append(node_index)
            clusters.append(current)

            return [cluster for cluster in clusters if cluster]

    def split_clusters_by_main_axis(
            self,
            points,
            clusters,
            *,
            main_axis,
            stats,
            params,
        ):
            """Split detected lanes by large gaps on the main axis.

            Accepted splits are binary and keep at least ``min_sub_nodes`` nodes
            on both sides.
            """
            split_clusters = []

            for cluster in clusters:
                parts = self.split_one_cluster_by_main_axis(
                    points,
                    cluster,
                    main_axis=main_axis,
                    stats=stats,
                    params=params,
                )
                split_clusters.extend(parts)

            return [sorted(cluster) for cluster in split_clusters if cluster]

    def split_one_cluster_by_main_axis(
            self,
            points,
            cluster,
            *,
            main_axis,
            stats,
            params,
        ):
            """Return a binary main-axis split for one cluster, or the original."""
            ordered = self.sort_by_axis(points, cluster, main_axis)

            if len(ordered) < max(2 * self.min_sub_nodes, 3):
                return [sorted(cluster)]

            values = [points[index][main_axis] for index in ordered]
            gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            positive_gaps = [gap for gap in gaps if gap > 1e-9]

            if not positive_gaps:
                return [sorted(cluster)]

            median_gap = self.median(positive_gaps)
            threshold = max(
                median_gap * params.main_axis_subsplit_gap_factor,
                stats.median_nn * params.main_axis_subsplit_min_abs_factor,
                1e-9,
            )

            split_position = self.best_main_axis_subsplit_position(
                gaps,
                threshold=threshold,
                params=params,
            )

            if split_position is None:
                return [sorted(cluster)]

            first = ordered[:split_position]
            second = ordered[split_position:]

            if len(first) < self.min_sub_nodes or len(second) < self.min_sub_nodes:
                return [sorted(cluster)]

            return [sorted(first), sorted(second)]

    def best_main_axis_subsplit_position(self, gaps, *, threshold, params):
            """Return the best valid binary split position on the main axis."""
            best = None
            best_score = None
            ratio = max(params.main_axis_subsplit_neighbor_gap_ratio, 1.0)

            for index, gap in enumerate(gaps):
                position = index + 1
                left_count = position
                right_count = len(gaps) + 1 - position

                if left_count < self.min_sub_nodes or right_count < self.min_sub_nodes:
                    continue

                if gap <= threshold:
                    continue

                left_internal_max = max(gaps[:index], default=0.0)
                right_internal_max = max(gaps[index + 1 :], default=0.0)
                internal_max = max(left_internal_max, right_internal_max, 1e-9)

                if gap < internal_max * ratio:
                    continue

                score = gap / internal_max
                if best_score is None or score > best_score:
                    best = position
                    best_score = score

            return best

    def merge_small_cross_clusters(self, points, clusters, *, cross_axis, min_nodes):
            """Merge clusters smaller than min_nodes into nearest larger cluster."""
            large = [list(cluster) for cluster in clusters if len(cluster) >= min_nodes]
            small = [list(cluster) for cluster in clusters if len(cluster) < min_nodes]

            if not large:
                return [sorted(node for cluster in clusters for node in cluster)]

            for cluster in small:
                target_index = min(
                    range(len(large)),
                    key=lambda index: abs(
                        self.mean_axis_value(points, cluster, cross_axis)
                        - self.mean_axis_value(points, large[index], cross_axis)
                    ),
                )
                large[target_index].extend(cluster)

            return [sorted(cluster) for cluster in large if cluster]

    def merge_narrow_adjacent_cross_clusters(
            self,
            points,
            clusters,
            *,
            cross_axis,
            stats,
            params,
        ):
            """Merge narrow adjacent lanes using coordinate-only criteria."""
            ordered = [
                sorted(cluster)
                for cluster in sorted(
                    clusters,
                    key=lambda cluster: (
                        self.mean_axis_value(points, cluster, cross_axis),
                        min(cluster),
                    ),
                )
                if cluster
            ]

            if len(ordered) <= 1:
                return ordered

            max_gap = max(stats.median_nn * params.narrow_sub_merge_gap_factor, 1e-9)
            absolute_narrow_width = max(
                stats.median_nn * params.narrow_sub_merge_max_width_factor,
                1e-9,
            )

            changed = True
            while changed and len(ordered) > 1:
                changed = False
                best_pair = None
                best_gap = None

                for index in range(len(ordered) - 1):
                    first = ordered[index]
                    second = ordered[index + 1]
                    gap = self.axis_interval_gap(
                        self.axis_interval(points, first, cross_axis),
                        self.axis_interval(points, second, cross_axis),
                    )

                    if gap > max_gap:
                        continue

                    first_width = self.cluster_axis_width(points, first, cross_axis)
                    second_width = self.cluster_axis_width(points, second, cross_axis)
                    narrow_width = min(first_width, second_width)
                    wide_width = max(first_width, second_width, 1e-9)
                    is_narrow = (
                        narrow_width <= absolute_narrow_width
                        or narrow_width <= wide_width * params.narrow_sub_merge_width_ratio
                    )

                    if not is_narrow:
                        continue

                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_pair = index

                if best_pair is not None:
                    ordered[best_pair] = sorted(
                        ordered[best_pair] + ordered[best_pair + 1],
                    )
                    del ordered[best_pair + 1]
                    changed = True

            return ordered

    def valid_main_overlap_pairs(self, points, clusters, *, main_axis, stats, params):
            """Return cluster index pairs with sufficient main-axis interval overlap."""
            pairs = []
            intervals = [self.axis_interval(points, cluster, main_axis) for cluster in clusters]
            min_abs_overlap = max(stats.median_nn * params.min_sub_main_overlap_abs_factor, 1e-9)

            for first in range(len(clusters)):
                for second in range(first + 1, len(clusters)):
                    overlap = self.interval_overlap(intervals[first], intervals[second])
                    if overlap <= 0:
                        continue

                    first_len = max(intervals[first][1] - intervals[first][0], 1e-9)
                    second_len = max(intervals[second][1] - intervals[second][0], 1e-9)
                    relative_overlap = overlap / min(first_len, second_len)

                    if (
                        relative_overlap >= params.min_sub_main_overlap_ratio
                        or overlap >= min_abs_overlap
                    ):
                        pairs.append((first, second))

            return pairs

    def merge_non_overlapping_clusters(
            self,
            points,
            clusters,
            *,
            overlapping_cluster_ids,
            cross_axis,
        ):
            """Merge unsupported clusters into nearest supported parallel lane."""
            supported = [
                list(cluster)
                for index, cluster in enumerate(clusters)
                if index in overlapping_cluster_ids
            ]
            unsupported = [
                list(cluster)
                for index, cluster in enumerate(clusters)
                if index not in overlapping_cluster_ids
            ]

            if not supported:
                return [sorted(node for cluster in clusters for node in cluster)]

            for cluster in unsupported:
                target_index = min(
                    range(len(supported)),
                    key=lambda index: abs(
                        self.mean_axis_value(points, cluster, cross_axis)
                        - self.mean_axis_value(points, supported[index], cross_axis)
                    ),
                )
                supported[target_index].extend(cluster)

            return [sorted(cluster) for cluster in supported if cluster]
