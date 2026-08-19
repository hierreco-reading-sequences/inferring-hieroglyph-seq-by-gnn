import math


class RadiusKnnEdges:
    """Add bounded radius-kNN edges inside one substructure."""

    def build(self, ops, points, nodes, *, params, **_):
        edges = set()
        node_set = set(nodes)
        if params.candidate_k <= 0:
            return []
        for source in nodes:
            for target in self.radius_knn_neighbors(ops, points, source, node_set, params=params):
                edges.add(ops.normalize_edge(source, target))
        return sorted(edges)

    def radius_knn_neighbors(self, ops, points, source, node_set, *, params):
            """Return local neighbors selected by a bounded k-neighbor radius."""
            distances = [
                (distance, target)
                for distance, target in ops.neighbor_distances(points, source)
                if target in node_set and distance <= params.max_edge_length
            ]

            if not distances or params.candidate_k <= 0:
                return []

            seed_count = min(params.candidate_k, len(distances))
            farthest_distance = distances[seed_count - 1][0]
            radius_factor = max(params.candidate_radius_factor, 1.0)
            threshold = farthest_distance * radius_factor

            return [
                target
                for distance, target in distances
                if distance <= threshold
            ]


class DirectionalSectorEdges:
    """Add sparse nearest-neighbor edges in angular sectors."""

    def build(self, ops, points, nodes, *, params, **_):
            """Return sparse nearest-neighbor edges in angular sectors."""
            sector_count = max(params.directional_neighbor_sectors, 0)
            per_sector = max(params.directional_neighbors_per_sector, 0)

            if sector_count <= 0 or per_sector <= 0:
                return []

            node_set = set(nodes)
            sector_width = (2.0 * math.pi) / sector_count
            edges = set()

            for source in nodes:
                source_x, source_y = points[source]
                buckets = [[] for _ in range(sector_count)]

                for target in node_set:
                    if source == target:
                        continue

                    distance = ops.l2_distance(points[source], points[target])
                    if distance > params.max_edge_length:
                        continue

                    angle = math.atan2(
                        points[target][1] - source_y,
                        points[target][0] - source_x,
                    )
                    normalized_angle = (angle + 2.0 * math.pi) % (2.0 * math.pi)
                    sector = min(int(normalized_angle / sector_width), sector_count - 1)
                    buckets[sector].append((distance, target))

                for bucket in buckets:
                    bucket.sort(key=lambda item: (item[0], item[1]))
                    for _, target in bucket[:per_sector]:
                        edges.add(ops.normalize_edge(source, target))

            return sorted(edges)


class EmptyTriangleEdges:
    """Add edges from local empty triangles."""

    def build(self, ops, points, nodes, *, params, **_):
            """Return local empty-triangle edges inside one substructure.

            Triangles are built from each node and its nearest configured
            neighbors. Accepted triangles meet the minimum-angle threshold and
            contain no other node. Triangle edges are not length-filtered.
            """
            neighbor_k = max(params.empty_triangle_neighbor_k, 0)
            if neighbor_k <= 0 or len(nodes) < 3:
                return []

            node_set = set(nodes)
            triangles = set()

            for anchor in nodes:
                neighbors = [
                    target
                    for _, target in ops.neighbor_distances(points, anchor)
                    if target in node_set
                ][:neighbor_k]

                for first_index in range(len(neighbors)):
                    first = neighbors[first_index]
                    for second in neighbors[first_index + 1 :]:
                        triangle = tuple(sorted((anchor, first, second)))
                        if len(set(triangle)) == 3:
                            triangles.add(triangle)

            edges = set()
            for first, second, third in sorted(triangles):
                triangle = (first, second, third)
                if not ops.triangle_has_min_angle(points, triangle, params=params):
                    continue

                if ops.triangle_is_empty(points, triangle, nodes):
                    edges.add(ops.normalize_edge(first, second))
                    edges.add(ops.normalize_edge(first, third))
                    edges.add(ops.normalize_edge(second, third))

            return sorted(edges)


class EmptyBoxEdges:
    """Add local diagonal edges across sparse axis-aligned boxes."""

    def build(self, ops, points, nodes, *, params, **_):
            """Return local diagonal edges across sparse axis-aligned boxes.

            Candidate count and maximum diagonal length are derived per graph
            from density, nearest-neighbor spacing, and length strictness.
            """
            neighbor_k = max(params.empty_box_neighbor_k, 0)
            if neighbor_k <= 0 or len(nodes) < 2:
                return []

            node_set = set(nodes)
            pairs = set()

            for source in nodes:
                neighbors = [
                    (distance, target)
                    for distance, target in ops.neighbor_distances(points, source)
                    if target in node_set and distance <= params.empty_box_max_length
                ][:neighbor_k]

                for _, target in neighbors:
                    edge = ops.normalize_edge(source, target)
                    if edge[0] != edge[1]:
                        pairs.add(edge)

            edges = set()
            for source, target in sorted(pairs):
                if not ops.is_diagonal_box_pair(points[source], points[target], params=params):
                    continue

                inside_count = ops.axis_aligned_box_inside_count(
                    points,
                    source,
                    target,
                    nodes,
                )
                if inside_count <= params.empty_box_max_inside:
                    edges.add((source, target))

            return sorted(edges)


class ForwardEnvelopeSweepEdges:
    """Add first-hit/front-hit edges from a directional envelope sweep."""

    def build(self, ops, points, nodes, *, graph_type, params, **_):
            """Return first-hit edges from a directional substructure sweep.

            Row samples sweep right; column samples sweep downward. The sweep
            is bounded on the cross axis and unbounded along its direction.
            """
            if len(nodes) <= 1:
                return []

            bbox = ops.bounding_box(points, nodes)
            if graph_type == "row":
                axis = 0
                direction = 1.0
                cross_min, cross_max = bbox[1], bbox[3]
            else:
                axis = 1
                direction = -1.0
                cross_min, cross_max = bbox[0], bbox[2]

            cross_axis = 1 - axis
            edges = []
            epsilon = 1e-9

            for source in nodes:
                source_axis_value = points[source][axis]
                source_cross_value = points[source][cross_axis]
                candidates = []
                has_source_front_peers = any(
                    source != target
                    and abs(points[target][axis] - source_axis_value)
                    <= params.envelope_sweep_front_window
                    and abs(points[target][cross_axis] - source_cross_value)
                    <= params.envelope_sweep_cross_window
                    for target in nodes
                )

                for target in nodes:
                    if source == target:
                        continue

                    target_cross_value = points[target][cross_axis]
                    if not cross_min - epsilon <= target_cross_value <= cross_max + epsilon:
                        continue

                    forward_delta = (
                        points[target][axis] - source_axis_value
                    ) * direction
                    if forward_delta <= epsilon:
                        continue
                    if (
                        has_source_front_peers
                        and forward_delta <= params.envelope_sweep_front_window
                    ):
                        continue

                    cross_delta = abs(target_cross_value - source_cross_value)
                    if cross_delta > params.envelope_sweep_cross_window:
                        continue

                    candidates.append((forward_delta, cross_delta, target))

                if not candidates:
                    continue

                first_front = min(candidate[0] for candidate in candidates)
                front_limit = first_front + params.envelope_sweep_front_window

                for forward_delta, cross_delta, target in sorted(candidates):
                    if forward_delta <= front_limit:
                        edges.append((source, target))

            return edges


class AxisWindowEdges:
    """Add short forward neighbors along the reading axis."""

    def build(self, ops, points, nodes, *, main_axis, params, **_):
        if params.axis_window <= 0:
            return []
        edges = set()
        ordered = ops.sort_by_axis(points, nodes, main_axis)
        for order_index, source in enumerate(ordered):
            stop = min(order_index + params.axis_window + 1, len(ordered))
            for neighbor_index in range(order_index + 1, stop):
                target = ordered[neighbor_index]
                if ops.l2_distance(points[source], points[target]) <= params.max_edge_length:
                    edges.add(ops.normalize_edge(source, target))
        return sorted(edges)


class ComponentBridgeEdges:
    """Connect disconnected components inside one substructure."""

    def build(self, ops, points, nodes, *, edges, main_axis, **_):
        """Bridge disconnected components inside one substructure.

        Components are ordered by their minimum reading-axis coordinate. Each
        neighboring component pair is connected by the shortest Euclidean edge
        between their nodes.
        """
        components = self.edge_components(nodes, edges)

        if len(components) <= 1:
            return []

        ordered_components = sorted(
            components,
            key=lambda component: (
                min(points[node][main_axis] for node in component),
                min(component),
            ),
        )
        bridges = []

        for first, second in zip(ordered_components, ordered_components[1:]):
            bridges.append(self.shortest_component_bridge(ops, points, first, second))

        return bridges

    @staticmethod
    def edge_components(nodes, edges):
        """Return connected components induced by ``edges`` over ``nodes``."""
        node_set = set(nodes)
        adjacency = {node: set() for node in node_set}

        for source, target in edges:
            if source in node_set and target in node_set:
                adjacency[source].add(target)
                adjacency[target].add(source)

        components = []
        seen = set()

        for start in nodes:
            if start in seen:
                continue

            stack = [start]
            seen.add(start)
            component = []

            while stack:
                node = stack.pop()
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

            components.append(tuple(sorted(component)))

        return components

    @staticmethod
    def shortest_component_bridge(ops, points, first_component, second_component):
        """Return the shortest edge between two node components."""
        return min(
            (
                (source, target)
                for source in first_component
                for target in second_component
            ),
            key=lambda edge: (
                ops.l2_distance(points[edge[0]], points[edge[1]]),
                edge[0],
                edge[1],
            ),
        )


class BoundarySubstructureEdges:
    """Connect adjacent substructure bounding boxes through boundary nodes."""

    def build(self, ops, points, substructures, *, graph_type, sample_scale):
        """Connect adjacent sub-bboxes through boundary-node pairs.

        Boundary nodes are selected near bbox corners. Inter-substructure links
        are not length-filtered.
        """
        if len(substructures) <= 1:
            return []

        edges = set()

        for first, second in self.adjacent_substructure_pairs(substructures):
            for source, target in self.all_boundary_node_pairs(first, second):
                if source == target:
                    continue

                edges.add(ops.normalize_edge(source, target))

        return sorted(edges)

    @staticmethod
    def adjacent_substructure_pairs(substructures):
        """Return consecutive substructure pairs along the cross axis."""
        ordered = sorted(
            substructures,
            key=lambda sub: (sub.cross_center, sub.sub_id),
        )
        return list(zip(ordered, ordered[1:]))

    @staticmethod
    def all_boundary_node_pairs(first, second):
        """Return all boundary-node pairs between two adjacent substructures."""
        pairs = []

        for source in first.boundary_nodes:
            for target in second.boundary_nodes:
                pairs.append((source, target))

        return tuple(pairs)
