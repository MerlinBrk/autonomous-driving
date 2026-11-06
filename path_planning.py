import numpy as np
from scipy.interpolate import splev, splprep


class PathPlanning:

    def __init__(self, num_waypoints=30, lane_width=20.0):
        self._car_position = np.array([48, 64])
        self._num_waypoints = num_waypoints
        self._lane_width = lane_width      # Startwert in Pixeln, wird laufend aus den Rändern geschätzt
        self._last_plan = None

    def plan(self, left_boundary, right_boundary):
        left  = self._resample(self._order_points(np.asarray(left_boundary, dtype=float), self._car_position[0]))
        right = self._resample(self._order_points(np.asarray(right_boundary, dtype=float), self._car_position[0]))

        center = self._center_line(left, right)
        if len(center) < 4:
            # Keine brauchbaren Ränder → letzten Plan weiterfahren, sonst geradeaus
            return self._last_plan if self._last_plan is not None else self._straight_ahead()

        way_points, curvature = self._smooth(center)
        self._last_plan = (way_points, curvature)
        return way_points, curvature

    def _center_line(self, left, right):
        if len(left) < 2 and len(right) < 2:
            return np.empty((0, 2))

        # Referenz = Rand mit mehr Punkten (gleicher Abstand → längerer Rand)
        if len(left) >= len(right):
            ref, other, inward = left, right, 1     # Straße liegt rechts vom linken Rand
        else:
            ref, other, inward = right, left, -1    # Straße liegt links vom rechten Rand

        # Tangente über ±2 Punkte (≈ 8 px) -> Normale robust gegen Pixelrauschen
        idx = np.arange(len(ref))
        tangent = ref[np.minimum(idx + 2, len(ref) - 1)] - ref[np.maximum(idx - 2, 0)]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        normal = inward * np.column_stack([-tangent[:, 1], tangent[:, 0]])   # rechts in Bildkoordinaten (y nach unten)
        offset_center = ref + normal * self._lane_width / 2

        if len(other) < 2:
            return offset_center

        dist = np.linalg.norm(ref[:, None, :] - other[None, :, :], axis=2)
        nearest = np.argmin(dist, axis=1)
        width = dist[np.arange(len(ref)), nearest]

        # Gültiges Paar: Partner nicht am Ende des anderen Rands und Breite plausibel
        valid = (nearest < len(other) - 1) & (np.abs(width - self._lane_width) < 0.5 * self._lane_width)
        if np.count_nonzero(valid) >= 3:
            self._lane_width = 0.9 * self._lane_width + 0.1 * np.median(width[valid])   # langsam nachführen

        return np.where(valid[:, None], (ref + other[nearest]) / 2, offset_center)

    def _smooth(self, points):
        # noinspection PyTupleAssignmentBalance
        tck, _ = splprep([points[:, 0], points[:, 1]], s=4 * len(points), k=3)
        t = np.linspace(0, 1, self._num_waypoints)
        x, y = splev(t, tck)
        dx, dy = splev(t, tck, der=1)
        ddx, ddy = splev(t, tck, der=2)

        # κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        kappa = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2) ** 1.5 + 1e-9)
        # Spline-Enden ausschwingen lassen, sonst Ausreißer in der Krümmung
        curvature = float(np.max(kappa[2:-2]))

        return np.column_stack([x, y]), curvature

    def _straight_ahead(self):
        y = np.linspace(self._car_position[1], 0, self._num_waypoints)
        return np.column_stack([np.full_like(y, self._car_position[0]), y]), 0.0

    @staticmethod
    def _order_points(points, car_x, radius=2.5, max_gap=6.0):
        if len(points) == 0:
            return np.empty((0, 2))

        start = np.lexsort((np.abs(points[:, 0] - car_x), -points[:, 1]))[0]   # größtes y, dann nah an car_x
        current = points[start]
        remaining = np.ones(len(points), dtype=bool)
        ordered = []

        while remaining.any():
            dist = np.linalg.norm(points - current, axis=1)
            near = remaining & (dist <= radius)
            ordered.append(points[near].mean(axis=0))
            remaining &= ~near
            if len(ordered) >= 3 and np.dot(ordered[-1] - ordered[-2], ordered[-2] - ordered[-3]) < 0:
                ordered.pop()
                break
            dist[~remaining] = np.inf
            nxt = np.argmin(dist)
            if dist[nxt] > max_gap:
                break
            current = points[nxt]

        return np.asarray(ordered)

    @staticmethod
    def _resample(polyline, spacing=2.0):
        """Polylinie in gleichen Abständen (Bogenlänge) neu abtasten → Ränder vergleichbar."""
        if len(polyline) < 2:
            return polyline
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
        s = np.arange(0, arc[-1], spacing)
        return np.column_stack([np.interp(s, arc, polyline[:, 0]), np.interp(s, arc, polyline[:, 1])])
