import math


class GeometryMixin:
    """Shared geometry and numeric helper methods."""

    @staticmethod
    def safe_float(value):
        """Convert a JSON scalar to float, using zero for missing values."""
        if value is None:
            return 0.0
        return float(value)

    @staticmethod
    def normalized_coordinates(points):
        """Normalize x and y coordinates independently into the ``0..1`` range."""
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = min(xs)
        min_y = min(ys)
        width = max(max(xs) - min_x, 1e-9)
        height = max(max(ys) - min_y, 1e-9)

        return (
            [(x - min_x) / width for x in xs],
            [(y - min_y) / height for y in ys],
        )

    @staticmethod
    def main_axis(graph_type):
        """Return reading axis index: x for rows, y for columns."""
        return 0 if graph_type == "row" else 1

    @staticmethod
    def axis_ranks(points, axis):
        """Return normalized ranks of points sorted along one coordinate axis."""
        ranks = [0.0 for _ in points]
        integer_ranks = GeometryMixin.integer_axis_ranks(points, axis)
        denominator = max(len(points) - 1, 1)

        for index, rank in enumerate(integer_ranks):
            ranks[index] = rank / denominator

        return ranks

    @staticmethod
    def integer_axis_ranks(points, axis):
        """Return integer ranks of points sorted along one coordinate axis."""
        ranks = [0 for _ in points]
        other_axis = 1 - axis
        ordered = sorted(
            range(len(points)),
            key=lambda index: (points[index][axis], points[index][other_axis], index),
        )

        for rank, index in enumerate(ordered):
            ranks[index] = rank

        return ranks

    @classmethod
    def local_density(cls, points, k):
        """Estimate local sparsity as mean distance to nearest ``k`` nodes."""
        densities = []

        for source in range(len(points)):
            distances = cls.neighbor_distances(points, source)
            nearest = distances[: min(k, len(distances))]

            if not nearest:
                densities.append(0.0)
                continue

            densities.append(sum(distance for distance, _ in nearest) / len(nearest))

        max_density = max(max(densities), 1e-9)
        return [density / max_density for density in densities]

    @classmethod
    def axis_neighbor_distances(cls, points, axis):
        """Return normalized distances to previous and next nodes in axis order."""
        prev_distances = [0.0 for _ in points]
        next_distances = [0.0 for _ in points]
        other_axis = 1 - axis
        ordered = sorted(
            range(len(points)),
            key=lambda index: (points[index][axis], points[index][other_axis], index),
        )
        all_distances = []

        for order_index, source in enumerate(ordered):
            if order_index > 0:
                previous_node = ordered[order_index - 1]
                distance = cls.l2_distance(points[source], points[previous_node])
                prev_distances[source] = distance
                all_distances.append(distance)

            if order_index < len(ordered) - 1:
                next_node = ordered[order_index + 1]
                distance = cls.l2_distance(points[source], points[next_node])
                next_distances[source] = distance
                all_distances.append(distance)

        normalizer = max(max(all_distances, default=0.0), 1e-9)
        return (
            [distance / normalizer for distance in prev_distances],
            [distance / normalizer for distance in next_distances],
        )

    @staticmethod
    def nearest_neighbors(points, source, k):
        """Return indices of the ``k`` nearest points to ``source``."""
        distances = GeometryMixin.neighbor_distances(points, source)
        return [target for _, target in distances[:k]]

    @staticmethod
    def neighbor_distances(points, source):
        """Return sorted ``(distance, index)`` pairs from source to all others."""
        distances = []

        for target, point in enumerate(points):
            if source == target:
                continue

            distances.append((GeometryMixin.l2_distance(points[source], point), target))

        distances.sort(key=lambda item: (item[0], item[1]))
        return distances

    @staticmethod
    def l2_distance(source, target):
        """Return Euclidean distance between two ``[x, y]`` points."""
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        return math.hypot(dx, dy)

    @classmethod
    def triangle_is_empty(cls, points, triangle, candidate_nodes):
        """Return true when no other candidate node lies inside a triangle."""
        first, second, third = triangle
        a = points[first]
        b = points[second]
        c = points[third]
        area = cls.signed_triangle_area(a, b, c)

        if abs(area) <= 1e-9:
            return False

        min_x = min(a[0], b[0], c[0]) - 1e-9
        max_x = max(a[0], b[0], c[0]) + 1e-9
        min_y = min(a[1], b[1], c[1]) - 1e-9
        max_y = max(a[1], b[1], c[1]) + 1e-9
        triangle_nodes = {first, second, third}

        for node in candidate_nodes:
            if node in triangle_nodes:
                continue

            point = points[node]
            if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
                continue

            if cls.point_in_triangle(point, a, b, c):
                return False

        return True

    def triangle_has_min_angle(self, points, triangle, *, params):
        """Return true when every internal triangle angle is large enough."""
        first, second, third = triangle
        min_angle = self.minimum_triangle_angle_degrees(
            points[first],
            points[second],
            points[third],
        )
        return min_angle >= params.empty_triangle_min_angle_degrees

    @staticmethod
    def is_diagonal_box_pair(first, second, *, params):
        """Return true when two points differ meaningfully in both axes."""
        dx = abs(second[0] - first[0])
        dy = abs(second[1] - first[1])
        longer_axis = max(dx, dy)

        if longer_axis <= 1e-9:
            return False

        shorter_axis = min(dx, dy)
        return shorter_axis / longer_axis >= params.empty_box_min_axis_fraction

    @classmethod
    def axis_aligned_box_inside_count(cls, points, source, target, candidate_nodes):
        """Count candidate nodes strictly inside the source-target rectangle."""
        first = points[source]
        second = points[target]
        min_x = min(first[0], second[0]) + 1e-9
        max_x = max(first[0], second[0]) - 1e-9
        min_y = min(first[1], second[1]) + 1e-9
        max_y = max(first[1], second[1]) - 1e-9

        if min_x >= max_x or min_y >= max_y:
            return 0

        inside_count = 0
        endpoints = {source, target}

        for node in candidate_nodes:
            if node in endpoints:
                continue

            point = points[node]
            if min_x < point[0] < max_x and min_y < point[1] < max_y:
                inside_count += 1

        return inside_count

    @classmethod
    def minimum_triangle_angle_degrees(cls, first, second, third):
        """Return the smallest internal angle of a triangle in degrees."""
        side_a = cls.l2_distance(second, third)
        side_b = cls.l2_distance(first, third)
        side_c = cls.l2_distance(first, second)

        if side_a <= 1e-9 or side_b <= 1e-9 or side_c <= 1e-9:
            return 0.0

        angles = (
            cls.angle_from_sides(side_b, side_c, side_a),
            cls.angle_from_sides(side_a, side_c, side_b),
            cls.angle_from_sides(side_a, side_b, side_c),
        )
        return min(angles)

    @staticmethod
    def angle_from_sides(first_side, second_side, opposite_side):
        """Return angle opposite ``opposite_side`` using the cosine rule."""
        denominator = 2.0 * first_side * second_side
        if denominator <= 1e-12:
            return 0.0

        cosine = (
            first_side * first_side
            + second_side * second_side
            - opposite_side * opposite_side
        ) / denominator
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    @classmethod
    def point_in_triangle(cls, point, first, second, third):
        """Return true when a point lies inside or on a triangle boundary."""
        epsilon = 1e-9
        area = cls.signed_triangle_area(first, second, third)

        if abs(area) <= epsilon:
            return False

        first_weight = cls.signed_triangle_area(point, second, third) / area
        second_weight = cls.signed_triangle_area(first, point, third) / area
        third_weight = cls.signed_triangle_area(first, second, point) / area

        return (
            first_weight >= -epsilon
            and second_weight >= -epsilon
            and third_weight >= -epsilon
            and first_weight <= 1.0 + epsilon
            and second_weight <= 1.0 + epsilon
            and third_weight <= 1.0 + epsilon
        )

    @staticmethod
    def signed_triangle_area(first, second, third):
        """Return signed doubled area for a 2-D triangle."""
        return (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )

    @staticmethod
    def sort_by_axis(points, node_indices, axis):
        """Sort node indices by one axis and use the other axis as tiebreaker."""
        other_axis = 1 - axis
        return sorted(
            node_indices,
            key=lambda index: (points[index][axis], points[index][other_axis], index),
        )

    @staticmethod
    def normalize_edge(source, target):
        """Return canonical undirected edge tuple ``(min, max)``."""
        if source < target:
            return source, target
        return target, source

    @staticmethod
    def bounding_box(points, nodes):
        """Return bbox ``(min_x, min_y, max_x, max_y)`` for selected nodes."""
        xs = [points[index][0] for index in nodes]
        ys = [points[index][1] for index in nodes]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def bbox_center(bbox, axis):
        """Return bbox center coordinate on x/y axis."""
        return 0.5 * (bbox[axis] + bbox[axis + 2])

    @staticmethod
    def bbox_diagonal(bbox):
        """Return bbox diagonal length."""
        min_x, min_y, max_x, max_y = bbox
        return math.hypot(max_x - min_x, max_y - min_y)

    @staticmethod
    def bbox_for_points(points):
        """Return bbox for all points."""
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        return GeometryMixin.bounding_box(points, range(len(points)))

    @classmethod
    def boundary_nodes_for_bbox(cls, points, nodes, bbox):
        """Assign nearest real node to each bbox corner."""
        return cls.boundary_nodes_for_named_corners(
            points,
            nodes,
            bbox,
            ("top_left", "top_right", "bottom_left", "bottom_right"),
        )

    @classmethod
    def boundary_nodes_for_named_corners(cls, points, nodes, bbox, corner_names):
        """Return nearest real nodes for selected named bbox corners."""
        selected = [
            cls.boundary_node_for_named_corner(points, nodes, bbox, name)
            for name in corner_names
        ]
        return tuple(dict.fromkeys(selected))

    @classmethod
    def boundary_node_for_named_corner(cls, points, nodes, bbox, corner_name):
        """Return nearest real node for one named bbox corner."""
        corner = cls.named_bbox_corners(bbox)[corner_name]
        return min(
            nodes,
            key=lambda node: (cls.l2_distance(points[node], corner), node),
        )

    @staticmethod
    def named_bbox_corners(bbox):
        """Return named bbox corner coordinates."""
        min_x, min_y, max_x, max_y = bbox
        return {
            "top_left": (min_x, min_y),
            "top_right": (max_x, min_y),
            "bottom_left": (min_x, max_y),
            "bottom_right": (max_x, max_y),
        }

    @staticmethod
    def axis_interval(points, nodes, axis):
        """Return min/max coordinate interval for nodes on one axis."""
        values = [points[index][axis] for index in nodes]
        return min(values), max(values)

    @staticmethod
    def interval_overlap(first, second):
        """Return length of overlap between two 1-D intervals."""
        return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))

    @staticmethod
    def axis_interval_gap(first, second):
        """Return positive distance between two 1-D intervals, or zero."""
        if first[1] < second[0]:
            return second[0] - first[1]
        if second[1] < first[0]:
            return first[0] - second[1]
        return 0.0

    @classmethod
    def cluster_axis_width(cls, points, nodes, axis):
        """Return coordinate width of selected nodes on one axis."""
        interval = cls.axis_interval(points, nodes, axis)
        return max(interval[1] - interval[0], 0.0)

    @staticmethod
    def mean_axis_value(points, nodes, axis):
        """Return mean coordinate on one axis for selected nodes."""
        if not nodes:
            return 0.0
        return sum(points[index][axis] for index in nodes) / len(nodes)

    @staticmethod
    def median(values):
        """Return median of a non-empty numeric sequence."""
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    @staticmethod
    def quantile(values, q):
        """Return a linear-interpolated quantile for a non-empty sequence."""
        ordered = sorted(values)
        if not ordered:
            raise ValueError("quantile() requires at least one value")

        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))

        if lower == upper:
            return ordered[lower]

        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def axis_positive_gaps(self, points, nodes, axis):
        """Return positive gaps between neighbors sorted on one axis."""
        ordered = self.sort_by_axis(points, nodes, axis)
        values = [points[index][axis] for index in ordered]
        return [
            values[index + 1] - values[index]
            for index in range(len(values) - 1)
            if values[index + 1] - values[index] > 1e-9
        ]

    @staticmethod
    def clamp_float(value, minimum, maximum):
        """Clamp a float into a closed interval."""
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def clamp_int(value, minimum, maximum):
        """Clamp an int into a closed interval."""
        return max(minimum, min(maximum, int(value)))

    @classmethod
    def estimate_local_scale(cls, points):
        """Estimate typical node spacing using nearest-neighbor distances."""
        if len(points) <= 1:
            return 1.0

        nearest_distances = []
        for source in range(len(points)):
            distances = cls.neighbor_distances(points, source)
            if distances:
                nearest_distances.append(distances[0][0])

        positive = [distance for distance in nearest_distances if distance > 1e-9]
        if positive:
            return max(cls.median(positive), 1e-9)

        bbox = cls.bbox_for_points(points)
        return max(cls.bbox_diagonal(bbox), 1.0)

    @classmethod
    def max_allowed_edge_length(cls, points, sample_scale, *, factor, ratio):
        """Return a conservative hard maximum edge length."""
        bbox = cls.bbox_for_points(points)
        diagonal = max(cls.bbox_diagonal(bbox), 1e-9)
        scale_limit = max(sample_scale * factor, 1e-9)
        ratio_limit = max(diagonal * ratio, 1e-9)
        return min(scale_limit, ratio_limit)
