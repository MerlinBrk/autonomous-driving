import numpy as np


class LongitudinalControl:
    """Speed control: target speed from the path curvature + PID controller for gas/brake."""

    def __init__(self, kp: float = 0.1, ki: float = 0.002, kd: float = 0.02,
                 v_min: float = 25.0, v_max: float = 90.0, a_lat: float = 110.0):
        # PID gains (error = target speed - current speed)
        self._kp = kp
        self._ki = ki
        self._kd = kd

        # Target speed parameters
        self._v_min = v_min      # lowest target speed (tight curves, start phase)
        self._v_max = v_max      # highest target speed (straights)
        self._a_lat = a_lat      # allowed "lateral acceleration" in px-based units, tuned empirically (>=130 -> car leaves the track in curves)

        # Controller state
        self._integral = 0.0          # accumulated error for the I-term
        self._last_speed = None       # previous speed for the D-term
        self._d_filtered = 0.0        # low-pass filtered derivative
        self._target_speed = v_min    # smoothed target speed (start slow, it is a race car)

    def control(self, speed: float, target_speed: float, steering_angle: float) -> tuple[float, float]:
        """PID speed controller.

        Args:
            speed: current speed of the car (info["speed"]).
            target_speed: desired speed (from predict_target_speed).
            steering_angle: current steering value; gas/brake are reduced when steering hard.

        Returns:
            (acceleration, braking): gas in [0, 1] and brake in [0, 0.8], sent separately
            to the simulation.
        """
        error = target_speed - speed

        # D-term on the measured speed instead of the error -> no derivative kick when the target jumps
        if self._last_speed is None:
            self._last_speed = speed
        d_speed = speed - self._last_speed
        self._last_speed = speed
        self._d_filtered = 0.7 * self._d_filtered + 0.3 * d_speed   # low-pass, the speed signal is noisy

        # I-term with anti-windup: only integrate while the output is not saturated
        u_without_i = self._kp * error - self._kd * self._d_filtered
        if abs(u_without_i + self._ki * self._integral) < 1.0:
            self._integral += error
        self._integral = np.clip(self._integral, -100.0, 100.0)

        u = u_without_i + self._ki * self._integral

        # Split the controller output: positive -> gas, negative -> brake (sent separately to the simulation)
        acceleration = float(np.clip(u, 0.0, 1.0))
        braking = float(np.clip(-u, 0.0, 0.8))   # >0.8 locks the wheels in CarRacing

        # Avoid strong gas/brake while steering hard -> otherwise the rear wheels lose grip and the car spins
        grip = 1.0 - min(abs(steering_angle), 1.0)
        acceleration *= 0.3 + 0.7 * grip
        braking *= 0.5 + 0.5 * grip

        # Small dead band: don't brake for tiny overspeed, just roll
        if braking < 0.05:
            braking = 0.0

        return acceleration, braking

    def predict_target_speed(self, curvature: float) -> float:
        """Target speed from path curvature: v = sqrt(a_lat / κ), limited to [v_min, v_max].

        Args:
            curvature: max. curvature κ of the planned path in 1/px (from the path planning).

        Returns:
            Smoothed target speed (decreases immediately, increases gradually).
        """
        # Curvature of the planned path in 1/px (max over the look-ahead, so curves are anticipated)
        kappa = max(float(curvature), 1e-6)
        v = np.sqrt(self._a_lat / kappa)
        v = float(np.clip(v, self._v_min, self._v_max))

        # Asymmetric smoothing: slow down immediately, speed up gradually (curvature estimate jitters)
        if v < self._target_speed:
            self._target_speed = v
        else:
            self._target_speed += 0.05 * (v - self._target_speed)

        return self._target_speed
