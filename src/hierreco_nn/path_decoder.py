from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DECODER_CHOICES = ("greedy", "ilp")


@dataclass(frozen=True)
class PathDecodeResult:
    """Decoded edges together with the decoder path taken at runtime."""

    edges: set[tuple[int, int]]
    requested_decoder: str
    decoder_used: str
    fallback_reason: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None


def validate_decoder(decoder: str) -> str:
    """Validate a path decoder name and its optional runtime dependencies."""
    if decoder not in DECODER_CHOICES:
        raise ValueError(f"decoder must be one of {DECODER_CHOICES}, got {decoder!r}")

    if decoder == "ilp":
        try:
            from scipy.optimize import milp  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "ILP decoder requires scipy.optimize.milp. "
                "Install a SciPy version that provides MILP support."
            ) from error

    return decoder


def decode_path_edges(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
    decoder: str = "ilp",
) -> set[tuple[int, int]]:
    """Decode model edge logits into one undirected simple path."""
    return decode_path_edges_with_diagnostics(
        logits,
        edge_index,
        node_count=node_count,
        decoder=decoder,
    ).edges


def decode_path_edges_with_diagnostics(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
    decoder: str = "ilp",
) -> PathDecodeResult:
    """Decode edges and report whether an ILP request used greedy fallback."""
    validate_decoder(decoder)
    if decoder == "ilp":
        return _ilp_decode_path_edges_result(
            logits,
            edge_index,
            node_count=node_count,
        )
    return PathDecodeResult(
        edges=greedy_decode_path_edges(logits, edge_index, node_count=node_count),
        requested_decoder="greedy",
        decoder_used="greedy",
    )


def undirected_edge_scores(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
) -> list[tuple[float, float, int, int]]:
    """Return ``(mean_logit, probability, source, target)`` per undirected edge."""
    if edge_index.numel() == 0 or logits.numel() == 0:
        return []

    source, target = edge_index.detach().cpu()
    logit_values = logits.detach().cpu()
    groups: dict[tuple[int, int], list[float]] = {}

    for edge_id in range(int(logit_values.numel())):
        u = int(source[edge_id].item())
        v = int(target[edge_id].item())
        if u == v or not (0 <= u < node_count and 0 <= v < node_count):
            continue
        edge = (u, v) if u < v else (v, u)
        groups.setdefault(edge, []).append(float(logit_values[edge_id].item()))

    edges = []
    for (u, v), values in groups.items():
        mean_logit = sum(values) / max(len(values), 1)
        probability = 1.0 / (1.0 + math.exp(-mean_logit))
        edges.append((mean_logit, probability, u, v))

    edges.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
    return edges


def greedy_decode_path_edges(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
) -> set[tuple[int, int]]:
    """Greedily select a high-scoring degree-2 acyclic path candidate."""
    if node_count <= 1:
        return set()

    edges = undirected_edge_scores(logits, edge_index, node_count=node_count)
    parent = list(range(node_count))
    rank = [0 for _ in range(node_count)]
    degree = [0 for _ in range(node_count)]
    selected: set[tuple[int, int]] = set()

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1
        return True

    for _, _, source, target in edges:
        if len(selected) >= node_count - 1:
            break
        if degree[source] >= 2 or degree[target] >= 2:
            continue
        if not union(source, target):
            continue
        degree[source] += 1
        degree[target] += 1
        selected.add((source, target))

    return selected


def ilp_decode_path_edges(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
    max_cut_iterations: int = 64,
) -> set[tuple[int, int]]:
    """Return only the edge set from the diagnostic ILP decoder."""
    return _ilp_decode_path_edges_result(
        logits,
        edge_index,
        node_count=node_count,
        max_cut_iterations=max_cut_iterations,
    ).edges


def _ilp_decode_path_edges_result(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
    max_cut_iterations: int = 64,
) -> PathDecodeResult:
    """Solve a maximum-score Hamiltonian path ILP over candidate edges.

    The ILP chooses ``N - 1`` edges, constrains every node degree to ``1..2``,
    and adds lazy connectivity cuts until the selected graph is one component.
    With ``N - 1`` edges and max degree 2, a connected feasible solution is a path.
    """
    if node_count <= 1:
        return PathDecodeResult(set(), "ilp", "ilp")

    scored_edges = undirected_edge_scores(logits, edge_index, node_count=node_count)
    edge_count = len(scored_edges)
    if edge_count < node_count - 1:
        return _greedy_fallback_result(
            logits,
            edge_index,
            node_count=node_count,
            reason="insufficient_candidate_edges",
        )

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack

    costs = -np.asarray([score for score, _, _, _ in scored_edges], dtype=float)
    integrality = np.ones(edge_count, dtype=np.int8)
    bounds = Bounds(np.zeros(edge_count), np.ones(edge_count))

    base_rows = 1 + node_count
    base_matrix = lil_matrix((base_rows, edge_count), dtype=float)
    lower = np.zeros(base_rows, dtype=float)
    upper = np.zeros(base_rows, dtype=float)

    base_matrix[0, :] = 1.0
    lower[0] = float(node_count - 1)
    upper[0] = float(node_count - 1)

    for edge_id, (_, _, source, target) in enumerate(scored_edges):
        base_matrix[1 + source, edge_id] = 1.0
        base_matrix[1 + target, edge_id] = 1.0

    lower[1:] = 1.0
    upper[1:] = 2.0

    cut_matrix = lil_matrix((0, edge_count), dtype=float)
    cut_lower = np.empty(0, dtype=float)
    cut_upper = np.empty(0, dtype=float)

    for _ in range(max_cut_iterations):
        matrix = vstack([base_matrix.tocsr(), cut_matrix.tocsr()], format="csr")
        constraint = LinearConstraint(
            matrix,
            np.concatenate([lower, cut_lower]),
            np.concatenate([upper, cut_upper]),
        )
        result = milp(
            c=costs,
            integrality=integrality,
            bounds=bounds,
            constraints=constraint,
            options={"disp": False},
        )

        if not result.success or result.x is None:
            return _greedy_fallback_result(
                logits,
                edge_index,
                node_count=node_count,
                reason="solver_failure",
            )

        chosen_ids = np.flatnonzero(result.x >= 0.5).tolist()
        selected = {
            (scored_edges[edge_id][2], scored_edges[edge_id][3])
            for edge_id in chosen_ids
        }
        components = connected_components(node_count, selected)
        if len(components) == 1 and len(selected) == node_count - 1:
            return PathDecodeResult(selected, "ilp", "ilp")

        new_cut_rows = []
        new_cut_lower = []
        new_cut_upper = []
        for component in components:
            if len(component) == node_count:
                continue

            component_set = set(component)
            crossing = [
                edge_id
                for edge_id, (_, _, source, target) in enumerate(scored_edges)
                if (source in component_set) != (target in component_set)
            ]
            if not crossing:
                continue

            row = lil_matrix((1, edge_count), dtype=float)
            for edge_id in crossing:
                row[0, edge_id] = 1.0
            new_cut_rows.append(row)
            new_cut_lower.append(1.0)
            new_cut_upper.append(np.inf)

        if not new_cut_rows:
            return _greedy_fallback_result(
                logits,
                edge_index,
                node_count=node_count,
                reason="no_connectivity_cut",
            )

        cut_matrix = vstack([cut_matrix, *new_cut_rows], format="lil")
        cut_lower = np.concatenate([cut_lower, np.asarray(new_cut_lower, dtype=float)])
        cut_upper = np.concatenate([cut_upper, np.asarray(new_cut_upper, dtype=float)])

    return _greedy_fallback_result(
        logits,
        edge_index,
        node_count=node_count,
        reason="cut_iteration_limit",
    )


def _greedy_fallback_result(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    node_count: int,
    reason: str,
) -> PathDecodeResult:
    """Return a greedy result annotated as an ILP fallback."""
    return PathDecodeResult(
        edges=greedy_decode_path_edges(logits, edge_index, node_count=node_count),
        requested_decoder="ilp",
        decoder_used="greedy",
        fallback_reason=reason,
    )


def connected_components(
    node_count: int,
    edges: set[tuple[int, int]],
) -> list[tuple[int, ...]]:
    """Return connected components of an undirected edge set."""
    adjacency = [set() for _ in range(node_count)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen = [False for _ in range(node_count)]
    components = []
    for start in range(node_count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))

    return components
