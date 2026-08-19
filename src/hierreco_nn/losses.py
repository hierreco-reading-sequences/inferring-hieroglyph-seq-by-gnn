from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def edge_label_counts(dataset: Iterable) -> tuple[float, float]:
    """Return positive and negative directed edge-label counts for a dataset."""
    positives = 0.0
    total = 0.0

    for data in dataset:
        labels = data.edge_y.float()
        positives += float(labels.sum().item())
        total += float(labels.numel())

    negatives = max(total - positives, 0.0)
    return positives, negatives


def compute_pos_weight(dataset: Iterable, *, max_value: float | None = None) -> float:
    """Compute BCE positive-class weight as negatives / positives."""
    positives, negatives = edge_label_counts(dataset)

    if positives <= 0.0:
        return 1.0

    value = negatives / positives
    if max_value is not None:
        value = min(value, float(max_value))
    return float(max(value, 1e-6))


class EdgeClassificationLoss(nn.Module):
    """BCE edge-classification loss with light differentiable path regularizers.

    The main objective is still supervised candidate-edge classification. The
    optional topology terms only regularize aggregate properties of the
    undirected prediction: node degrees should match the GT path degrees and
    each graph should contain ``num_nodes - 1`` selected undirected edges.
    """

    def __init__(
        self,
        *,
        pos_weight: float | torch.Tensor | None = None,
        bce_loss_weight: float = 1.0,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        degree_loss_weight: float = 0.02,
        edge_count_loss_weight: float = 0.01,
        endpoint_loss_weight: float = 0.0,
        incident_ranking_loss_weight: float = 0.0,
        symmetry_loss_weight: float = 0.0,
        endpoint_sigma: float = 0.35,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if bce_loss_weight < 0.0:
            raise ValueError("bce_loss_weight must be non-negative")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if degree_loss_weight < 0.0:
            raise ValueError("degree_loss_weight must be non-negative")
        if edge_count_loss_weight < 0.0:
            raise ValueError("edge_count_loss_weight must be non-negative")

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            weight = torch.as_tensor([float(pos_weight)], dtype=torch.float32)
            self.register_buffer("pos_weight", weight)

        self.bce_loss_weight = float(bce_loss_weight)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.degree_loss_weight = float(degree_loss_weight)
        self.edge_count_loss_weight = float(edge_count_loss_weight)
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.incident_ranking_loss_weight = float(incident_ranking_loss_weight)
        self.symmetry_loss_weight = float(symmetry_loss_weight)
        self.endpoint_sigma = float(endpoint_sigma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target_or_data) -> torch.Tensor:
        if torch.is_tensor(target_or_data):
            return self.edge_bce_loss(logits, target_or_data)

        data = target_or_data
        loss = self.bce_loss_weight * self.edge_bce_loss(logits, data.edge_y)

        if self.reduction == "none":
            return loss

        return loss + self.path_topology_loss(logits, data)

    def edge_bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        hard_target = target

        if self.label_smoothing > 0.0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.focal_gamma > 0.0:
            probability = torch.sigmoid(logits)
            p_t = torch.where(hard_target >= 0.5, probability, 1.0 - probability)
            loss = loss * (1.0 - p_t).clamp_min(1e-6).pow(self.focal_gamma)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    def path_topology_loss(self, logits: torch.Tensor, data) -> torch.Tensor:
        """Return soft path penalties on undirected probabilities."""
        if self.degree_loss_weight == 0.0 and self.edge_count_loss_weight == 0.0:
            return logits.new_tensor(0.0)

        edge_index = data.edge_index
        if edge_index.numel() == 0:
            return logits.new_tensor(0.0)

        undirected = undirected_edge_logits(
            logits,
            data.edge_y,
            edge_index,
            num_nodes=int(data.num_nodes),
        )
        if undirected is None:
            return logits.new_tensor(0.0)

        source, target, _, probability, target_value = undirected
        node_count = int(data.num_nodes)
        soft_degree = logits.new_zeros(node_count)
        target_degree = logits.new_zeros(node_count)
        soft_degree.scatter_add_(0, source, probability)
        soft_degree.scatter_add_(0, target, probability)
        target_degree.scatter_add_(0, source, target_value)
        target_degree.scatter_add_(0, target, target_value)

        loss = logits.new_tensor(0.0)
        if self.degree_loss_weight:
            degree_error = (soft_degree - target_degree) / target_degree.clamp_min(1.0)
            loss = loss + self.degree_loss_weight * F.smooth_l1_loss(
                degree_error,
                torch.zeros_like(degree_error),
            )

        if self.edge_count_loss_weight:
            node_batch = getattr(data, "batch", None)
            if node_batch is None:
                node_batch = torch.zeros(node_count, dtype=torch.long, device=logits.device)
            graph_count = int(node_batch.max().item()) + 1 if node_count else 0
            graph_index = node_batch[source]

            soft_counts = logits.new_zeros(graph_count)
            soft_counts.scatter_add_(0, graph_index, probability)

            node_counts = logits.new_zeros(graph_count)
            node_counts.scatter_add_(0, node_batch, torch.ones_like(node_batch, dtype=logits.dtype))
            target_counts = (node_counts - 1.0).clamp_min(0.0)
            count_error = (soft_counts - target_counts) / target_counts.clamp_min(1.0)
            loss = loss + self.edge_count_loss_weight * F.smooth_l1_loss(
                count_error,
                torch.zeros_like(count_error),
            )

        return loss


def undirected_edge_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Collapse directed reverse edges into unique undirected edge logits."""
    source, target = edge_index
    mask = source != target
    if not bool(mask.any()):
        return None

    source = source[mask]
    target = target[mask]
    directed_logits = logits[mask]
    label = labels[mask].float()

    undirected_source = torch.minimum(source, target)
    undirected_target = torch.maximum(source, target)
    keys = undirected_source * int(num_nodes) + undirected_target
    unique_keys_cpu, inverse_cpu = torch.unique(
        keys.detach().cpu(),
        sorted=False,
        return_inverse=True,
    )
    unique_keys = unique_keys_cpu.to(device=logits.device)
    inverse = inverse_cpu.to(device=logits.device)
    unique_count = int(unique_keys.numel())

    counts = directed_logits.new_zeros(unique_count)
    counts.scatter_add_(0, inverse, torch.ones_like(directed_logits))

    logit_sum = directed_logits.new_zeros(unique_count)
    logit_sum.scatter_add_(0, inverse, directed_logits)
    logit_mean = logit_sum / counts.clamp_min(1.0)
    probability_mean = torch.sigmoid(logit_mean)

    label_sum = label.new_zeros(unique_count)
    label_sum.scatter_add_(0, inverse, label)
    label_mean = label_sum / counts.clamp_min(1.0)

    unique_source = unique_keys // int(num_nodes)
    unique_target = unique_keys % int(num_nodes)
    return (
        unique_source.long(),
        unique_target.long(),
        logit_mean,
        probability_mean,
        label_mean,
    )


def undirected_edge_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Collapse directed reverse edges into unique undirected probabilities."""
    values = undirected_edge_logits(
        logits,
        labels,
        edge_index,
        num_nodes=num_nodes,
    )
    if values is None:
        return None

    source, target, _, probability, label = values
    return source, target, probability, label
