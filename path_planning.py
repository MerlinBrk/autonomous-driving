import numpy as np
from scipy.interpolate import splev, splprep


class PathPlanning:
    """Plans the path (center line of the lane) from the two lane boundaries.

    Coordinates: image coordinates (x to the right, y down), unit pixels.
    """

    def __init__(self, num_waypoints: int = 30, lane_width: float = 20.0):
        self._car_position = np.array([48, 64])  # car position in image coordinates (x, y)
        self._num_waypoints = num_waypoints      # number of waypoints in the planned path
        self._lane_width = lane_width            # initial value in pixels, continuously estimated from the boundaries
        self._last_plan = None                   # fallback if no usable boundaries are found

    def plan(self, left_boundary: np.ndarray, right_boundary: np.ndarray) -> tuple[np.ndarray, float]:
        """Plan the path for the current frame.

        Args:
            left_boundary, right_boundary: (N, 2) arrays with pixel coordinates [x, y]
                of the lane boundaries, in any order (from the lane detection or the
                ground truth of the simulation). May be empty.

        Returns:
            (way_points, curvature): way_points is a (num_waypoints, 2) array [x, y],
            ordered from the car forward. curvature is the max. curvature κ of the
            path in 1/px (used by the longitudinal control for the target speed).
        """
        # Order boundary points along the road and resample them at equal spacing
        left  = self._resample(self._order_points(np.asarray(left_boundary, dtype=float), self._car_position[0]))
        right = self._resample(self._order_points(np.asarray(right_boundary, dtype=float), self._car_position[0]))

        center = self._center_line(left, right)
        if len(center) < 4:
            # No usable boundaries -> keep following the last plan, otherwise drive straight ahead
            return self._last_plan if self._last_plan is not None else self._straight_ahead()

        # Smooth the center line with a spline and compute its curvature
        way_points, curvature = self._smooth(center)
        self._last_plan = (way_points, curvature)
        return way_points, curvature

    def _center_line(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Center points of the lane: midpoints of matching boundary points, or the
        reference boundary shifted by half a lane width if there is no valid partner."""
        if len(left) < 2 and len(right) < 2:
            return np.empty((0, 2))

        # Reference = boundary with more points (equal spacing -> longer boundary)
        if len(left) >= len(right):
            ref, other, inward = left, right, 1     # road lies to the right of the left boundary
        else:
            ref, other, inward = right, left, -1    # road lies to the left of the right boundary

        # Tangent over ±2 points (≈ 8 px) -> normal is robust against pixel noise
        idx = np.arange(len(ref))
        tangent = ref[np.minimum(idx + 2, len(ref) - 1)] - ref[np.maximum(idx - 2, 0)]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        normal = inward * np.column_stack([-tangent[:, 1], tangent[:, 0]])   # right side in image coordinates (y points down)
        # Fallback center: shift the reference boundary by half a lane width towards the road
        offset_center = ref + normal * self._lane_width / 2

        if len(other) < 2:
            return offset_center

        # For each reference point, find the nearest point on the other boundary
        dist = np.linalg.norm(ref[:, None, :] - other[None, :, :], axis=2)
        nearest = np.argmin(dist, axis=1)
        width = dist[np.arange(len(ref)), nearest]

        # Valid pair: partner is not at the end of the other boundary and width is plausible
        valid = (nearest < len(other) - 1) & (np.abs(width - self._lane_width) < 0.5 * self._lane_width)
        if np.count_nonzero(valid) >= 3:
            self._lane_width = 0.9 * self._lane_width + 0.1 * np.median(width[valid])   # update slowly (low-pass filter)

        # Use the midpoint of valid pairs, otherwise the offset fallback
        return np.where(valid[:, None], (ref + other[nearest]) / 2, offset_center)

    def _smooth(self, points: np.ndarray) -> tuple[np.ndarray, float]:
        """Fit a smoothing cubic spline through the center points, sample num_waypoints
        points and compute the max. curvature from the spline derivatives."""
        # noinspection PyTupleAssignmentBalance
        tck, _ = splprep([points[:, 0], points[:, 1]], s=4 * len(points), k=3)
        t = np.linspace(0, 1, self._num_waypoints)
        x, y = splev(t, tck)                 # positions
        dx, dy = splev(t, tck, der=1)        # first derivative
        ddx, ddy = splev(t, tck, der=2)      # second derivative

        # κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        kappa = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2) ** 1.5 + 1e-9)
        # Ignore the spline ends, otherwise they cause curvature outliers
        curvature = float(np.max(kappa[2:-2]))

        return np.column_stack([x, y]), curvature

    def _straight_ahead(self) -> tuple[np.ndarray, float]:
        """Straight path from the car position up to the top of the image (curvature 0)."""
        y = np.linspace(self._car_position[1], 0, self._num_waypoints)
        return np.column_stack([np.full_like(y, self._car_position[0]), y]), 0.0

    @staticmethod
    def _order_points(points: np.ndarray, car_x: float, radius: float = 2.5, max_gap: float = 6.0) -> np.ndarray:
        """Order unsorted boundary pixels into a polyline, starting near the car.

        This is the only place where boundary points are ordered (the lane detection
        returns them unordered). Besides ordering it also
          - averages all points within `radius` into one point -> the 2 px wide edge band
            becomes a thin line,
          - stops at a gap > max_gap or if the line turns back on itself -> other visible
            parts of the track are cut off.
        """
        if len(points) == 0:
            return np.empty((0, 2))

        start = np.lexsort((np.abs(points[:, 0] - car_x), -points[:, 1]))[0]   # largest y, then closest to car_x
        current = points[start]
        remaining = np.ones(len(points), dtype=bool)
        ordered = []

        while remaining.any():
            # Merge all remaining points within the radius into one averaged point
            dist = np.linalg.norm(points - current, axis=1)
            near = remaining & (dist <= radius)
            ordered.append(points[near].mean(axis=0))
            remaining &= ~near
            # Stop if the direction reverses (path turns back on itself)
            if len(ordered) >= 3 and np.dot(ordered[-1] - ordered[-2], ordered[-2] - ordered[-3]) < 0:
                ordered.pop()
                break
            # Jump to the nearest remaining point, stop if the gap is too large
            dist[~remaining] = np.inf
            nxt = np.argmin(dist)
            if dist[nxt] > max_gap:
                break
            current = points[nxt]

        return np.asarray(ordered)

    @staticmethod
    def _resample(polyline: np.ndarray, spacing: float = 2.0) -> np.ndarray:
        """Resample the polyline at equal arc-length spacing -> boundaries become comparable."""
        if len(polyline) < 2:
            return polyline
        # Cumulative arc length along the polyline
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
        s = np.arange(0, arc[-1], spacing)
        return np.column_stack([np.interp(s, arc, polyline[:, 0]), np.interp(s, arc, polyline[:, 1])])
