from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv


class MLP(nn.Module):
    """Small feed-forward block used by encoders and the edge classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        final_activation: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        ]

        if final_activation:
            layers.extend(
                [
                    nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity(),
                    nn.ReLU(),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINEBranch(nn.Module):
    """Stable edge-conditioned message passing over the candidate graph."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            message_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GINEConv(
                    message_mlp,
                    edge_dim=hidden_dim,
                    train_eps=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(node_h, edge_index, edge_h)
            updated = F.relu(norm(updated))
            updated = F.dropout(updated, p=self.dropout, training=self.training)
            node_h = node_h + updated
        return node_h


class GATv2Branch(nn.Module):
    """Optional edge-aware attention branch for ablations."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.attention_heads,
                    heads=self.attention_heads,
                    concat=True,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(self.hidden_dim))

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            updated = conv(node_h, edge_index, edge_h)
            updated = F.relu(norm(updated))
            updated = F.dropout(updated, p=self.dropout, training=self.training)
            node_h = node_h + updated
        return node_h


class EdgeNodeGNN(nn.Module):
    """Path-aware classifier for candidate directed edges.

    Nodes and edge attributes are embedded, propagated over the candidate
    graph, and scored from endpoint context plus edge features. Path constraints
    are applied by the loss and decoder.
    """

    branch_choices = ("gine", "gatv2")

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        *,
        hidden_dim: int = 48,
        num_layers: int = 2,
        attention_heads: int = 4,
        gnn_branch: str = "gine",
        edge_competition_layers: int = 0,
        incident_selector: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if gnn_branch not in self.branch_choices:
            raise ValueError(
                f"gnn_branch must be one of {self.branch_choices}, got {gnn_branch!r}"
            )
        if attention_heads < 1:
            raise ValueError(f"attention_heads must be >= 1, got {attention_heads}")
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"attention_heads ({attention_heads})"
            )

        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.attention_heads = int(attention_heads)
        self.gnn_branch = str(gnn_branch)
        self.edge_competition_layers = int(edge_competition_layers)
        self.incident_selector = bool(incident_selector)
        self.dropout = float(dropout)

        self.node_encoder = MLP(
            self.node_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout=self.dropout,
            final_activation=True,
        )
        self.edge_encoder = MLP(
            self.edge_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout=self.dropout,
            final_activation=True,
        )

        self.gine_branch = (
            GINEBranch(self.hidden_dim, self.num_layers, self.dropout)
            if self.gnn_branch == "gine"
            else None
        )
        self.gatv2_branch = (
            GATv2Branch(
                self.hidden_dim,
                self.num_layers,
                self.attention_heads,
                self.dropout,
            )
            if self.gnn_branch == "gatv2"
            else None
        )

        self.fusion = nn.Identity()
        self.edge_competition = None
        self.path_neighbor_head = None

        edge_repr_dim = self.hidden_dim * 5
        self.edge_context_encoder = MLP(
            edge_repr_dim,
            self.hidden_dim,
            self.hidden_dim,
            dropout=self.dropout,
            final_activation=True,
        )
        self.edge_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if edge_index.numel() == 0:
            return edge_attr.new_empty((0,))

        node_h = self.node_encoder(data.x)
        edge_h = self.edge_encoder(edge_attr)

        if self.gnn_branch == "gine":
            node_h = self.gine_branch(node_h, edge_index, edge_h)
        else:
            node_h = self.gatv2_branch(node_h, edge_index, edge_h)

        source, target = edge_index
        source_h = node_h[source]
        target_h = node_h[target]
        edge_repr = torch.cat(
            [
                source_h,
                target_h,
                torch.abs(source_h - target_h),
                source_h * target_h,
                edge_h,
            ],
            dim=-1,
        )
        edge_context = self.edge_context_encoder(edge_repr)
        return self.edge_head(edge_context).view(-1)
