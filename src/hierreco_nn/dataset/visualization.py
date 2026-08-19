from __future__ import annotations

from pathlib import Path

from .hierreco_dataset import HierrecoDataset


class HierrecoVisualizer:
    """Render candidate and target graphs from a ``HierrecoDataset``."""

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def undirected_edges(edge_index):
        """Collapse a PyG edge_index tensor into canonical undirected edges."""
        edges = set()

        for source, target in edge_index.t().tolist():
            if source != target:
                edges.add(HierrecoDataset.normalize_edge(source, target))

        return sorted(edges)

    @classmethod
    def missing_target_edges(cls, data):
        """Return target edges that are absent from the candidate graph."""
        candidate_edges = set(cls.undirected_edges(data.edge_index))
        target_edges = set(cls.undirected_edges(data.target_edge_index))
        return sorted(target_edges - candidate_edges)

    def build_records(self):
        """Return all samples ordered with missing-coverage samples first."""
        records = []

        for sample_index in range(len(self.dataset)):
            data = self.dataset[sample_index]
            missing_edges = self.missing_target_edges(data)
            records.append((sample_index, data, missing_edges))

        return sorted(
            records,
            key=lambda record: (
                0 if record[2] else 1,
                -len(record[2]),
                record[0],
            ),
        )

    def choose_sample(self, requested_index):
        """Prefer a sample with missing target-edge coverage unless requested."""
        if requested_index is not None:
            data = self.dataset[requested_index]
            return requested_index, data, self.missing_target_edges(data)

        return self.build_records()[0]

    def render(
        self,
        data,
        selected_index,
        missing_edges,
        *,
        output_path=None,
        show_boundaries=True,
        show_bboxes=True,
    ):
        """Render one sample visualization to PNG or an interactive window."""
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        points = data.pos.numpy()
        candidate_edges = set(self.undirected_edges(data.edge_index))
        target_edges = set(self.undirected_edges(data.target_edge_index))

        fig, ax = plt.subplots(figsize=(10, 7))

        for source, target in sorted(candidate_edges | target_edges):
            edge = HierrecoDataset.normalize_edge(source, target)
            is_candidate = edge in candidate_edges
            is_target = edge in target_edges

            if is_candidate and is_target:
                color = "red"
                linewidth = 2.0
                linestyle = "-"
                zorder = 2
            elif is_target:
                color = "red"
                linewidth = 2.0
                linestyle = "--"
                zorder = 2
            else:
                color = "0.75"
                linewidth = 1.0
                linestyle = "-"
                zorder = 1

            ax.plot(
                [points[source, 0], points[target, 0]],
                [points[source, 1], points[target, 1]],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )

        if show_bboxes and hasattr(data, "sub_bboxes"):
            for sub_id, bbox in enumerate(data.sub_bboxes.tolist()):
                min_x, min_y, max_x, max_y = bbox
                rect = patches.Rectangle(
                    (min_x, min_y),
                    max(max_x - min_x, 1e-9),
                    max(max_y - min_y, 1e-9),
                    fill=False,
                    edgecolor="0.35",
                    linewidth=1.0,
                    linestyle=":",
                    zorder=2.5,
                )
                ax.add_patch(rect)
                ax.text(
                    min_x,
                    min_y,
                    f"sub {sub_id}",
                    fontsize=8,
                    color="0.35",
                    verticalalignment="bottom",
                    zorder=3,
                )

        ax.scatter(points[:, 0], points[:, 1], color="black", s=28, zorder=3)

        if show_boundaries and hasattr(data, "subline_boundary_mask"):
            boundary_mask = data.subline_boundary_mask.numpy()
            if boundary_mask.any():
                ax.scatter(
                    points[boundary_mask, 0],
                    points[boundary_mask, 1],
                    facecolors="none",
                    edgecolors="blue",
                    s=95,
                    linewidth=1.7,
                    zorder=4,
                )

        label_ids = getattr(data, "blob_ids", data.node_ids)
        for index, label_id in enumerate(label_ids):
            ax.annotate(
                label_id,
                (points[index, 0], points[index, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        sub_count = int(data.sub_bboxes.shape[0]) if hasattr(data, "sub_bboxes") else 0
        sample_name = Path(data.sample_path).name if hasattr(data, "sample_path") else ""
        component_id = getattr(data, "component_id", "")
        ax.set_title(
            f"{sample_name} | component_id={component_id} | index={selected_index} "
            f"graph_type={data.graph_type} subs={sub_count} "
            f"candidates={len(candidate_edges)} target={len(target_edges)} "
            f"missing={len(missing_edges)}"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="0.9", linewidth=0.8)
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def visualization_filename(order_index, sample_index, data, missing_edges):
        """Build a sortable visualization filename for one sample."""
        status = "bad" if missing_edges else "ok"
        sample_name = Path(data.sample_path).stem
        return f"{order_index:05d}_{status}_idx_{sample_index:05d}_{sample_name}.png"

    def render_all(self, *, output_dir=None, limit=None, show_boundaries=True, show_bboxes=True):
        """Render all samples, ordered with missing-coverage samples first."""
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        records = self.build_records()
        if limit is not None:
            records = records[:limit]

        for order_index, (sample_index, data, missing_edges) in enumerate(records):
            output_path = None
            if output_dir is not None:
                output_path = output_dir / self.visualization_filename(
                    order_index,
                    sample_index,
                    data,
                    missing_edges,
                )

            print(
                f"Visualizing {order_index + 1}/{len(records)}: "
                f"index={sample_index}, subs={data.sub_bboxes.shape[0]}, "
                f"missing={len(missing_edges)}"
            )
            self.render(
                data,
                sample_index,
                missing_edges,
                output_path=output_path,
                show_boundaries=show_boundaries,
                show_bboxes=show_bboxes,
            )

        return records
