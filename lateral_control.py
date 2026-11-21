import numpy as np


class LateralControl:
    """Lateral control with switchable controller: Stanley or Pure Pursuit.

    Coordinates: image coordinates (x to the right, y down), unit pixels.
    Driving direction of the car = up in the image (-y).
    Output: steering value for CarRacing (+ = right). The value roughly equals the
    wheel steering angle in rad; the wheels are mechanically limited to +-0.4 rad.
    """

    STANLEY = "stanley"
    PURE_PURSUIT = "pure_pursuit"

    def __init__(
        self,
        mode="stanley",
        # --- Stanley ---
        k=3.0,                 # gain for the cross-track error
        k_soft=5.0,            # prevents overreaction at low speed
        heading_window=6.0,    # [px] distance over which the path tangent is averaged
        # --- Pure Pursuit ---
        wheelbase=5.0,         # [px] wheelbase (~3.2 world units * ~1.55 px)
        ld_min=8.0,            # [px] minimum lookahead distance
        ld_gain=0.15,          # [px per speed unit]
        ld_max=30.0,           # [px] maximum lookahead distance
        # --- general ---
        smoothing=0.3,         # low-pass on steering: 0 = off, ->1 = very sluggish
        max_steer=1.0,
        max_gap=15.0,          # [px] larger point spacing = different part of the track
        warmup_steps=50,       # steps at the start without steering (camera still zooming, image unusable)
    ):
        if mode not in (self.STANLEY, self.PURE_PURSUIT):
            raise ValueError(f"Unknown mode '{mode}', allowed: 'stanley', 'pure_pursuit'")

        self._car_position = np.array([48.0, 64.0])
        self.mode = mode

        self.k = k
        self.k_soft = k_soft
        self.heading_window = heading_window

        self.wheelbase = wheelbase
        self.ld_min = ld_min
        self.ld_gain = ld_gain
        self.ld_max = ld_max

        self.smoothing = smoothing
        self.max_steer = max_steer
        self.max_gap = max_gap

        self.warmup_steps = warmup_steps

        self._prev_steer = 0.0
        self._step = 0

    # ------------------------------------------------------------------ API
    def control(self, trajectory, speed):
        # Start phase: the camera is still zooming in during the first ~50 steps, pixel
        # distances are wrong and the lane detection returns garbage. The car starts straight
        # on the track -> just drive straight ahead, otherwise it starts to oscillate.
        self._step += 1
        if self._step <= self.warmup_steps:
            return 0.0

        path = self._prepare_path(trajectory)
        if path is None:
            # no usable path -> slowly decay the last steering value
            self._prev_steer *= 0.9
            return float(self._prev_steer)

        if self.mode == self.STANLEY:
            delta = self._stanley(path, speed)
        else:
            delta = self._pure_pursuit(path, speed)

        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        # low-pass filter against jitter
        delta = self.smoothing * self._prev_steer + (1.0 - self.smoothing) * delta
        self._prev_steer = delta
        return delta

    def reset(self):
        self._prev_steer = 0.0
        self._step = 0

    # ------------------------------------------------------------ Stanley
    def _stanley(self, path, speed):
        s, i = self._nearest_on_path(path)

        # average the tangent over a short distance starting at the nearest point
        p0 = self._point_at(path, s)
        p1 = self._point_at(path, s + self.heading_window)
        tangent = p1 - p0
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            norm = np.linalg.norm(tangent) + 1e-6
        t = tangent / norm

        # heading error: angle between driving direction (0,-1) and tangent, >0 = right
        heading_error = np.arctan2(t[0], -t[1])

        # signed perpendicular cross-track error, >0 = path is right of the car
        r = self._car_position - p0
        # (r x t: path pointing up, car left of it -> e > 0 -> steer right)
        e = r[0] * t[1] - r[1] * t[0]

        return heading_error + np.arctan2(self.k * e, self.k_soft + speed)

    # ------------------------------------------------------- Pure Pursuit
    def _pure_pursuit(self, path, speed):
        ld = np.clip(self.ld_min + self.ld_gain * speed, self.ld_min, self.ld_max)

        # first path point ahead of the car with distance >= ld
        s_near, _ = self._nearest_on_path(path)
        dist = np.linalg.norm(path - self._car_position, axis=1)
        arc = self._arc_length(path)
        candidates = np.where((arc >= s_near) & (dist >= ld))[0]
        target = path[candidates[0]] if len(candidates) else path[-1]

        d = target - self._car_position
        ld_actual = max(np.linalg.norm(d), 1e-6)
        alpha = np.arctan2(d[0], -d[1])      # >0 = target is to the right

        return np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld_actual)

    # ------------------------------------------------------------ Helpers
    def _prepare_path(self, trajectory):
        """Order the trajectory along the driving direction and resample it with
        1 px spacing. Cuts off other visible parts of the track."""
        if trajectory is None:
            return None
        pts = np.asarray(trajectory, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            return None

        i = int(np.argmin(np.linalg.norm(pts - self._car_position, axis=1)))

        # determine driving direction: which index direction goes up in the image?
        n = 5
        fwd_y = pts[min(i + n, len(pts) - 1)][1] - pts[i][1]
        bwd_y = pts[max(i - n, 0)][1] - pts[i][1]
        if bwd_y < fwd_y:
            pts = pts[::-1]
            i = len(pts) - 1 - i

        # contiguous piece around i (jumps = different part of the track)
        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1) > self.max_gap
        start = i
        while start > 0 and not gaps[start - 1]:
            start -= 1
        end = i
        while end < len(pts) - 1 and not gaps[end]:
            end += 1
        pts = pts[start:end + 1]
        if len(pts) < 2:
            return None

        # resample uniformly (1 px)
        arc = self._arc_length(pts)
        if arc[-1] < 1e-6:
            return None
        s = np.arange(0.0, arc[-1], 1.0)
        x = np.interp(s, arc, pts[:, 0])
        y = np.interp(s, arc, pts[:, 1])
        return np.stack([x, y], axis=1)

    @staticmethod
    def _arc_length(path):
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def _nearest_on_path(self, path):
        i = int(np.argmin(np.linalg.norm(path - self._car_position, axis=1)))
        return self._arc_length(path)[i], i

    def _point_at(self, path, s):
        arc = self._arc_length(path)
        s = np.clip(s, 0.0, arc[-1])
        return np.array([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])
