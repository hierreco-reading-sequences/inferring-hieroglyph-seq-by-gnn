#!/usr/bin/env python3

import argparse
import json
import sys
from collections import deque
from math import factorial, hypot
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
DEFAULT_ZERNIKE_DEGREE = 8
DEFAULT_ZERNIKE_SIZE = 128


def parse_value(value):
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def sorted_node_ids(node_ids):
    def key(node_id):
        if node_id.startswith("n") and node_id[1:].isdigit():
            return int(node_id[1:])
        return node_id

    return sorted(node_ids, key=key)


def transform_point(x, y, position_type, reading_direction):
    """Normalize orientation using the requested flip/rotation rules."""
    transform = (position_type, reading_direction)

    if transform == ("down", "right"):
        return x, y
    if transform == ("down", "left"):
        return -x, y
    if transform == ("up", "right"):
        return x, -y
    if transform == ("up", "left"):
        return -x, -y
    if transform == ("right", "up"):
        return y, -x
    if transform == ("left", "up"):
        rotated_x, rotated_y = y, -x
        return rotated_x, -rotated_y
    if transform == ("right", "down"):
        rotated_x, rotated_y = -y, x
        return rotated_x, -rotated_y
    if transform == ("left", "down"):
        return -y, x

    raise ValueError(
        "Unsupported position_type/reading_direction combination: "
        f"{position_type!r}/{reading_direction!r}"
    )


def transform_bbox(bbox, position_type, reading_direction, origin_x, origin_y):
    min_x = float(bbox["bbox_x"])
    min_y = float(bbox["bbox_y"])
    max_x = min_x + float(bbox["bbox_width"])
    max_y = min_y + float(bbox["bbox_height"])
    corners = [
        transform_point(x, y, position_type, reading_direction)
        for x, y in (
            (min_x, min_y),
            (max_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
        )
    ]
    xs = [x - origin_x for x, _ in corners]
    ys = [y - origin_y for _, y in corners]
    bbox_min_x = min(xs)
    bbox_min_y = min(ys)
    bbox_max_x = max(xs)
    bbox_max_y = max(ys)

    return {
        "bbox_x": bbox_min_x,
        "bbox_y": bbox_min_y,
        "bbox_width": bbox_max_x - bbox_min_x,
        "bbox_height": bbox_max_y - bbox_min_y,
    }


def reverse_reading_direction(reading_direction):
    """Return the opposite reading direction before coordinate normalization."""
    opposites = {
        "right": "left",
        "left": "right",
        "up": "down",
        "down": "up",
    }

    try:
        return opposites[reading_direction]
    except KeyError as error:
        raise ValueError(
            f"Unsupported reading_direction: {reading_direction!r}"
        ) from error


def resolve_graphml_path(graphml_path):
    graphml_path = Path(graphml_path)
    if graphml_path.is_file():
        return graphml_path

    if not graphml_path.is_dir():
        raise FileNotFoundError(f"GraphML path does not exist: {graphml_path}")

    preferred = graphml_path / "graphs_v2.graphml"
    if preferred.is_file():
        return preferred

    candidates = sorted(
        path
        for path in graphml_path.glob("*.graphml")
        if not path.name.startswith("._")
    )
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No GraphML file was found in directory: {graphml_path}"
        )

    raise ValueError(
        "Could not choose a GraphML file automatically. Use the exact file path. "
        f"Candidates: {[str(path) for path in candidates]}"
    )


def find_default_pgm_path(graphml_path):
    graphml_path = Path(graphml_path)
    candidates = sorted(
        path
        for path in graphml_path.parent.glob("*.pgm")
        if not path.name.startswith("._")
    )

    if not candidates:
        return None

    preferred = [path for path in candidates if path.name.startswith("ThrFill")]
    if len(preferred) == 1:
        return preferred[0]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not choose a PGM image automatically. Use --pgm-path. "
        f"Candidates: {[str(path) for path in candidates]}"
    )


def load_binary_pgm(pgm_path):
    image = Image.open(pgm_path).convert("L")
    return np.asarray(image)


def bbox_union(group, node_attrs):
    boxes = []

    for node_id in group:
        attrs = node_attrs[node_id]
        bbox_x = attrs.get("bbox_x")
        bbox_y = attrs.get("bbox_y")
        bbox_width = attrs.get("bbox_width")
        bbox_height = attrs.get("bbox_height")

        if None in (bbox_x, bbox_y, bbox_width, bbox_height):
            raise ValueError(f"Missing bbox attributes for node {node_id}.")

        min_x = int(bbox_x)
        min_y = int(bbox_y)
        max_x = min_x + int(bbox_width)
        max_y = min_y + int(bbox_height)
        boxes.append((min_x, min_y, max_x, max_y))

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)

    return {
        "bbox_x": min_x,
        "bbox_y": min_y,
        "bbox_width": max_x - min_x,
        "bbox_height": max_y - min_y,
    }


def crop_foreground_mask(image, bbox):
    image_height, image_width = image.shape[:2]
    min_x = max(int(bbox["bbox_x"]), 0)
    min_y = max(int(bbox["bbox_y"]), 0)
    max_x = min(min_x + int(bbox["bbox_width"]), image_width)
    max_y = min(min_y + int(bbox["bbox_height"]), image_height)

    if min_x >= max_x or min_y >= max_y:
        raise ValueError(f"Invalid bbox for image crop: {bbox}")

    crop = image[min_y:max_y, min_x:max_x]

    # PGM is binary with black glyph blobs on a white background.
    return crop < 128


def normalize_mask_for_zernike(mask, size=DEFAULT_ZERNIKE_SIZE, padding=2):
    height, width = mask.shape

    if height <= 0 or width <= 0:
        raise ValueError("Cannot normalize an empty mask for Zernike moments.")

    available_diameter = max(size - 2 * padding, 1)
    scale = available_diameter / max(hypot(width, height), 1e-9)
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))

    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (resized_width, resized_height),
        Image.Resampling.NEAREST,
    )
    resized_mask = np.asarray(resized) > 0

    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized_height) // 2
    x0 = (size - resized_width) // 2
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized_mask.astype(
        np.uint8
    )
    return canvas


def zernike_orders(degree):
    """Return the standard non-negative Zernike order family."""
    return [
        (n, m)
        for n in range(degree + 1)
        for m in range(n + 1)
        if (n - m) % 2 == 0
    ]


def zernike_radial(n, m, rho):
    """Evaluate the radial Zernike polynomial R_n^m."""
    radial = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + m) // 2 - k)
                * factorial((n - m) // 2 - k)
            )
        )
        radial += coefficient * np.power(rho, n - 2 * k)
    return radial


def zernike_moments_numpy(canvas, radius, degree):
    """Compute rotation-invariant Zernike moment magnitudes using NumPy."""
    values = canvas.astype(float)
    height, width = values.shape
    y_indices, x_indices = np.indices((height, width), dtype=float)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    x = (x_indices - center_x) / max(radius, 1e-12)
    y = (y_indices - center_y) / max(radius, 1e-12)
    rho = np.sqrt((x * x) + (y * y))
    theta = np.arctan2(y, x)
    support = (rho <= 1.0) & (values > 0.0)

    orders = zernike_orders(degree)
    if not np.any(support):
        return [0.0 for _ in orders]

    support_values = values[support]
    rho_values = rho[support]
    theta_values = theta[support]
    normalizer = max(float(support_values.sum()), 1e-12)

    moments = []
    for n, m in orders:
        radial = zernike_radial(n, m, rho_values)
        basis = radial * np.exp(-1j * m * theta_values)
        moment = (n + 1) * np.sum(support_values * basis) / normalizer
        moments.append(float(abs(moment)))
    return moments


def zernike_moments_for_bbox(
    image,
    bbox,
    degree,
    size,
):
    mask = crop_foreground_mask(image, bbox)

    if not mask.any():
        raise ValueError(f"Empty foreground crop for bbox: {bbox}")

    canvas = normalize_mask_for_zernike(mask, size=size)
    radius = size // 2
    return zernike_moments_numpy(canvas, radius, degree)


def load_graphml(graphml_path):
    root = ET.parse(graphml_path).getroot()
    graph = root.find("g:graph", GRAPHML_NS)

    if graph is None:
        raise ValueError(f"No <graph> element found in {graphml_path}")

    node_attrs = {}
    adjacency = {}

    for node in graph.findall("g:node", GRAPHML_NS):
        node_id = node.attrib["id"]
        node_attrs[node_id] = {
            data.attrib["key"]: parse_value(data.text)
            for data in node.findall("g:data", GRAPHML_NS)
        }
        adjacency[node_id] = set()

    edges = []
    for edge in graph.findall("g:edge", GRAPHML_NS):
        source = edge.attrib["source"]
        target = edge.attrib["target"]

        edges.append([source, target])
        adjacency[source].add(target)
        adjacency[target].add(source)

    return node_attrs, adjacency, edges


def connected_components(node_attrs, adjacency):
    seen = set()

    for start in sorted_node_ids(node_attrs):
        if start in seen:
            continue

        component = []
        queue = deque([start])
        seen.add(start)

        while queue:
            node_id = queue.popleft()
            component.append(node_id)

            for neighbor in sorted_node_ids(adjacency[node_id]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        yield sorted_node_ids(component)


def component_orientation(component_nodes, node_attrs):
    orientation_nodes = [
        node_id
        for node_id in component_nodes
        if all(
            key in node_attrs[node_id]
            for key in ("seq_type", "position_type", "reading_direction")
        )
    ]

    if len(orientation_nodes) != 1:
        raise ValueError(
            "Each connected component must contain exactly one node with "
            "seq_type, position_type and reading_direction. "
            f"Found {len(orientation_nodes)} in component {component_nodes}."
        )

    attrs = node_attrs[orientation_nodes[0]]
    graph_type = attrs["seq_type"]

    if graph_type not in ("row", "col"):
        raise ValueError(f"Unsupported seq_type: {graph_type!r}")

    return graph_type, attrs["position_type"], attrs["reading_direction"]


def validate_component_attributes(component_nodes, node_attrs):
    missing = []

    for node_id in component_nodes:
        attrs = node_attrs[node_id]
        missing_keys = [
            key
            for key in ("x", "y")
            if key not in attrs or attrs[key] is None
        ]

        if missing_keys:
            missing.append(f"{node_id}: {', '.join(missing_keys)}")

    if missing:
        raise ValueError(
            "Missing required node attributes: " + "; ".join(missing)
        )


def component_start_node(component_nodes, node_attrs):
    origin_nodes = [
        node_id
        for node_id in component_nodes
        if node_attrs[node_id].get("sequence_pos") == 0
    ]

    if origin_nodes:
        return origin_nodes[0]

    return component_nodes[0] if component_nodes else None


def describe_component_error(component_index, component_nodes, node_attrs, error):
    start_node = component_start_node(component_nodes, node_attrs)
    start_attrs = node_attrs.get(start_node, {}) if start_node is not None else {}
    start_x = start_attrs.get("x", "<missing>")
    start_y = start_attrs.get("y", "<missing>")

    return (
        f"Skipping component {component_index}: {error}. "
        f"start_node={start_node}, start_x={start_x}, start_y={start_y}, "
        f"nodes={component_nodes}"
    )


def is_nonzero_superblob_id(value):
    return value not in (None, 0, "0", "")


def merge_groups_by_superblob(component_nodes, node_attrs):
    superblob_groups = {}
    passthrough_groups = []

    for node_id in component_nodes:
        superblob_id = node_attrs[node_id].get("superblob_id")

        if is_nonzero_superblob_id(superblob_id):
            superblob_groups.setdefault(superblob_id, []).append(node_id)
        else:
            passthrough_groups.append([node_id])

    groups = passthrough_groups + list(superblob_groups.values())
    groups = [sorted_node_ids(group) for group in groups]

    def group_key(group):
        positions = [
            node_attrs[node_id].get("sequence_pos")
            for node_id in group
            if node_attrs[node_id].get("sequence_pos") is not None
        ]

        if positions:
            return min(positions)

        return sorted_node_ids(group)[0]

    return sorted(groups, key=group_key)


def group_representative_id(group):
    return sorted_node_ids(group)[0]


def remap_edges(component_edges, node_to_merged_id, merged_nodes):
    node_order = {node_id: index for index, node_id in enumerate(merged_nodes)}
    remapped_edges = set()

    for source, target in component_edges:
        merged_source = node_to_merged_id[source]
        merged_target = node_to_merged_id[target]

        if merged_source == merged_target:
            continue

        edge = tuple(
            sorted(
                (merged_source, merged_target),
                key=lambda node_id: node_order[node_id],
            )
        )
        remapped_edges.add(edge)

    return [list(edge) for edge in sorted(remapped_edges, key=lambda e: (
        node_order[e[0]],
        node_order[e[1]],
    ))]


def validate_connected_after_merge(nodes, edges):
    if not nodes:
        raise ValueError("Merged component has no nodes.")

    adjacency = {node_id: set() for node_id in nodes}

    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen = set()
    queue = deque([nodes[0]])
    seen.add(nodes[0])

    while queue:
        node_id = queue.popleft()

        for neighbor in adjacency[node_id]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    if len(seen) != len(nodes):
        missing_nodes = [node_id for node_id in nodes if node_id not in seen]
        raise ValueError(
            "Merged component is disconnected after superblob_id merge. "
            f"Unreachable nodes: {missing_nodes}"
        )


def merged_sequence_pos(group, node_attrs):
    values = [
        node_attrs[node_id].get("sequence_pos")
        for node_id in group
        if node_attrs[node_id].get("sequence_pos") is not None
    ]

    if not values:
        return None

    return min(values)


def group_node_with_lowest_sequence_pos(group, node_attrs):
    return min(
        group,
        key=lambda node_id: (
            node_attrs[node_id].get("sequence_pos")
            if node_attrs[node_id].get("sequence_pos") is not None
            else float("inf"),
            sorted_node_ids([node_id])[0],
        ),
    )


def merged_quadrat(group, node_attrs):
    node_id = group_node_with_lowest_sequence_pos(group, node_attrs)
    return node_attrs[node_id].get("quadrat")


def merged_bbox(group, node_attrs):
    return bbox_union(group, node_attrs)


def merged_blob_id(group, node_attrs):
    node_id = group_node_with_lowest_sequence_pos(group, node_attrs)
    return node_attrs[node_id].get("blob_id")


def average_original_point(group, node_attrs):
    return (
        sum(float(node_attrs[node_id]["x"]) for node_id in group) / len(group),
        sum(float(node_attrs[node_id]["y"]) for node_id in group) / len(group),
    )


def component_id_for_component(component_nodes, node_attrs):
    component_ids = {
        node_attrs[node_id].get("component_id")
        for node_id in component_nodes
    }

    if len(component_ids) != 1:
        raise ValueError(
            "Each connected component must contain exactly one component_id. "
            f"Found {sorted(component_ids)} in component {component_nodes}."
        )

    return next(iter(component_ids))


def component_to_sample(
    component_nodes,
    node_attrs,
    edges,
    *,
    image=None,
    zernike_degree=DEFAULT_ZERNIKE_DEGREE,
    zernike_size=DEFAULT_ZERNIKE_SIZE,
):
    component_set = set(component_nodes)
    component_id = component_id_for_component(component_nodes, node_attrs)
    graph_type, position_type, reading_direction = component_orientation(
        component_nodes,
        node_attrs,
    )
    reading_direction = reverse_reading_direction(reading_direction)
    validate_component_attributes(component_nodes, node_attrs)

    component_edges = [
        edge
        for edge in edges
        if edge[0] in component_set and edge[1] in component_set
    ]

    merge_groups = merge_groups_by_superblob(component_nodes, node_attrs)
    merged_nodes = [group_representative_id(group) for group in merge_groups]
    node_to_merged_id = {
        node_id: group_representative_id(group)
        for group in merge_groups
        for node_id in group
    }

    merged_points = {}
    merged_sequence_positions = {}
    merged_quadrats = {}
    merged_bboxes = {}
    merged_blob_ids = {}
    merged_original_points = {}
    merged_zernike_moments = {}

    for group in merge_groups:
        merged_id = group_representative_id(group)
        original_x, original_y = average_original_point(group, node_attrs)
        bbox = merged_bbox(group, node_attrs)
        merged_points[merged_id] = transform_point(
            original_x,
            original_y,
            position_type,
            reading_direction,
        )
        merged_original_points[merged_id] = (original_x, original_y)
        merged_sequence_positions[merged_id] = merged_sequence_pos(group, node_attrs)
        merged_quadrats[merged_id] = merged_quadrat(group, node_attrs)
        merged_bboxes[merged_id] = bbox
        merged_blob_ids[merged_id] = merged_blob_id(group, node_attrs)

        if image is not None:
            merged_zernike_moments[merged_id] = zernike_moments_for_bbox(
                image,
                bbox,
                zernike_degree,
                zernike_size,
            )

    origin_nodes = [
        node_id
        for node_id in merged_nodes
        if merged_sequence_positions[node_id] == 0
    ]

    if len(origin_nodes) != 1:
        raise ValueError(
            "Each connected component must contain exactly one node with "
            f"sequence_pos == 0. Found {len(origin_nodes)} in component "
            f"{component_nodes}."
        )

    origin_x, origin_y = merged_points[origin_nodes[0]]
    remapped_edges = remap_edges(component_edges, node_to_merged_id, merged_nodes)
    validate_connected_after_merge(merged_nodes, remapped_edges)

    features = {}
    for node_id in merged_nodes:
        x, y = merged_points[node_id]
        original_x, original_y = merged_original_points[node_id]
        transformed_bbox = transform_bbox(
            merged_bboxes[node_id],
            position_type,
            reading_direction,
            origin_x,
            origin_y,
        )

        features[node_id] = {
            "x": x - origin_x,
            "y": y - origin_y,
            "original_x": original_x,
            "original_y": original_y,
            "sequence_pos": merged_sequence_positions[node_id],
            "quadrat": merged_quadrats[node_id],
            "blob_id": merged_blob_ids[node_id],
            **transformed_bbox,
        }

        if image is not None:
            features[node_id]["zernike_moments"] = merged_zernike_moments[node_id]

    return {
        "edges_list": remapped_edges,
        "nodes_array": merged_nodes,
        "component_id": component_id,
        "graph_type": graph_type,
        "feature": features,
    }


def export_components_to_json(
    graphml_path,
    output_dir,
    min_nodes=5,
    *,
    pgm_path=None,
    zernike=True,
    zernike_degree=DEFAULT_ZERNIKE_DEGREE,
    zernike_size=DEFAULT_ZERNIKE_SIZE,
):
    graphml_path = resolve_graphml_path(graphml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = graphml_path.parent.name

    image = None
    if zernike:
        if pgm_path is None:
            pgm_path = find_default_pgm_path(graphml_path)

        if pgm_path is None:
            raise ValueError(
                "Zernike export is enabled, but no PGM image was found. "
                "Use --pgm-path or --no-zernike."
            )

        image = load_binary_pgm(pgm_path)

    node_attrs, adjacency, edges = load_graphml(graphml_path)
    components = list(connected_components(node_attrs, adjacency))
    exported_count = 0
    skipped_count = 0

    for sample_index, component_nodes in enumerate(components):
        if len(component_nodes) < min_nodes:
            skipped_count += 1
            continue

        try:
            sample = component_to_sample(
                component_nodes,
                node_attrs,
                edges,
                image=image,
                zernike_degree=zernike_degree,
                zernike_size=zernike_size,
            )
        except (KeyError, TypeError, ValueError) as error:
            skipped_count += 1
            print(
                describe_component_error(
                    sample_index,
                    component_nodes,
                    node_attrs,
                    error,
                ),
                file=sys.stderr,
            )
            continue

        if len(sample["nodes_array"]) < min_nodes:
            skipped_count += 1
            continue

        output_path = output_dir / f"{sample_name}_sample_{sample_index:05d}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(sample, file, ensure_ascii=False, indent=2)
            file.write("\n")

        exported_count += 1

    print(
        f"Exported {exported_count} samples to: {output_dir} "
        f"(skipped {skipped_count} of {len(components)} components)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export connected GraphML components as normalized JSON samples."
    )
    parser.add_argument(
        "graphml_path",
        help=(
            "Path to the input GraphML file, or to a directory containing "
            "graphs_v2.graphml."
        ),
    )
    parser.add_argument("output_dir", help="Directory where JSON samples are saved.")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=5,
        help="Minimum number of nodes required to export a connected component.",
    )
    parser.add_argument(
        "--pgm-path",
        type=Path,
        help=(
            "Path to the binary PGM image used for Zernike blob features. "
            "Defaults to the ThrFill*.pgm file next to the GraphML file."
        ),
    )
    parser.add_argument(
        "--zernike-degree",
        type=int,
        default=DEFAULT_ZERNIKE_DEGREE,
        help="Maximum Zernike moment degree.",
    )
    parser.add_argument(
        "--zernike-size",
        type=int,
        default=DEFAULT_ZERNIKE_SIZE,
        help=(
            "Fixed square mask size used before computing Zernike moments. "
            "The crop is scaled to fit fully inside the Zernike circle."
        ),
    )
    parser.add_argument(
        "--no-zernike",
        action="store_true",
        help="Skip Zernike blob features.",
    )

    args = parser.parse_args()
    export_components_to_json(
        args.graphml_path,
        args.output_dir,
        args.min_nodes,
        pgm_path=args.pgm_path,
        zernike=not args.no_zernike,
        zernike_degree=args.zernike_degree,
        zernike_size=args.zernike_size,
    )


if __name__ == "__main__":
    main()
