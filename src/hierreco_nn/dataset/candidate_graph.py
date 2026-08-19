from .edge_rules import (
    AxisWindowEdges,
    BoundarySubstructureEdges,
    ComponentBridgeEdges,
    DirectionalSectorEdges,
    EmptyBoxEdges,
    EmptyTriangleEdges,
    ForwardEnvelopeSweepEdges,
    RadiusKnnEdges,
)


class CandidateGraphMixin:
    """Build candidate graphs from coordinates using modular edge rules."""

    internal_edge_rules = (
        RadiusKnnEdges(),
        DirectionalSectorEdges(),
        EmptyTriangleEdges(),
        EmptyBoxEdges(),
        ForwardEnvelopeSweepEdges(),
        AxisWindowEdges(),
    )
    component_bridge_rule = ComponentBridgeEdges()
    boundary_edge_rule = BoundarySubstructureEdges()

    def make_candidate_edges(
            self,
            points,
            *,
            graph_type=None,
            return_substructures=False,
            return_x_diagonal_edges=False,
        ):
            """Generate target-independent candidate edges from geometry.

            Only coordinates and graph type are used. Internal substructure
            edges use length limits; boundary-node links between substructures
            are unbounded.
            """
            node_count = len(points)
            edges = set()

            if node_count <= 1:
                substructures = self.build_single_substructure(points, graph_type)
                if return_substructures and return_x_diagonal_edges:
                    return [], substructures, []
                if return_substructures:
                    return [], substructures
                return []

            graph_type = graph_type if graph_type in ("row", "col") else "col"
            main_axis = self.main_axis(graph_type)
            stats = self.graph_geometry_stats(points, main_axis=main_axis)
            params = self.derive_candidate_graph_params(stats)
            substructures = self.detect_sequence_substructures(
                points,
                graph_type=graph_type,
                stats=stats,
                params=params,
            )

            for sub in substructures:
                for source, target in self.internal_substructure_edges(
                    points,
                    sub.nodes,
                    graph_type=graph_type,
                    main_axis=main_axis,
                    params=params,
                ):
                    edges.add(self.normalize_edge(source, target))

            x_diagonal_edges = self.boundary_edge_rule.build(
                self,
                points,
                substructures,
                graph_type=graph_type,
                sample_scale=stats.median_nn,
            )

            x_diagonal_edges = self.expand_edges_across_duplicate_points(
                points,
                x_diagonal_edges,
            )

            for source, target in x_diagonal_edges:
                edges.add(self.normalize_edge(source, target))

            sorted_edges = sorted(edges)
            x_diagonal_edges = sorted(set(x_diagonal_edges))

            if return_substructures and return_x_diagonal_edges:
                return sorted_edges, substructures, x_diagonal_edges
            if return_substructures:
                return sorted_edges, substructures
            return sorted_edges


    def internal_substructure_edges(self, points, nodes, *, graph_type, main_axis, params):
        """Return local candidate edges inside one connected substructure."""
        edges = set()
        node_list = list(nodes)

        if len(node_list) <= 1:
            return []

        for rule in self.internal_edge_rules:
            for source, target in rule.build(
                self,
                points,
                node_list,
                graph_type=graph_type,
                main_axis=main_axis,
                params=params,
            ):
                edges.add(self.normalize_edge(source, target))

        for source, target in self.component_bridge_rule.build(
            self,
            points,
            node_list,
            edges=edges,
            main_axis=main_axis,
        ):
            edges.add(self.normalize_edge(source, target))

        return sorted(edges)

    def expand_edges_across_duplicate_points(self, points, edges):
            """Add equivalent edges for nodes that share identical coordinates."""
            duplicate_groups = self.duplicate_point_groups(points)

            if not duplicate_groups:
                return edges

            replacements = {}
            for group in duplicate_groups:
                for node in group:
                    replacements[node] = group

            expanded = set(edges)
            for source, target in edges:
                source_group = replacements.get(source, (source,))
                target_group = replacements.get(target, (target,))

                for source_equivalent in source_group:
                    for target_equivalent in target_group:
                        if source_equivalent != target_equivalent:
                            expanded.add(
                                self.normalize_edge(source_equivalent, target_equivalent)
                            )

            return expanded

    @staticmethod
    def duplicate_point_groups(points):
            """Return groups of node indices with exactly identical coordinates."""
            groups = {}

            for index, point in enumerate(points):
                key = (float(point[0]), float(point[1]))
                groups.setdefault(key, []).append(index)

            return [tuple(group) for group in groups.values() if len(group) > 1]
