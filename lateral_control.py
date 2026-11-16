import numpy as np


class LateralControl:
    """Querregelung mit umschaltbarem Regler: Stanley oder Pure Pursuit.

    Koordinaten: Bildkoordinaten (x nach rechts, y nach unten), Einheit Pixel.
    Fahrtrichtung des Autos = im Bild nach oben (-y).
    Ausgabe: Lenkwert für CarRacing (+ = rechts). Der Wert entspricht etwa dem
    Radlenkwinkel in rad, die Räder sind mechanisch auf +-0.4 rad begrenzt.
    """

    STANLEY = "stanley"
    PURE_PURSUIT = "pure_pursuit"

    def __init__(
        self,
        mode="stanley",
        # --- Stanley ---
        k=3.0,                 # Verstärkung Querabweichung
        k_soft=5.0,            # verhindert Überreaktion bei kleiner Geschwindigkeit
        heading_window=6.0,    # [px] Strecke, über die die Pfadtangente gemittelt wird
        # --- Pure Pursuit ---
        wheelbase=5.0,         # [px] Radstand (~3.2 Weltheinheiten * ~1.55 px)
        ld_min=8.0,            # [px] minimale Lookahead-Distanz
        ld_gain=0.15,          # [px pro Geschwindigkeitseinheit]
        ld_max=30.0,           # [px] maximale Lookahead-Distanz
        # --- allgemein ---
        smoothing=0.3,         # Tiefpass auf Lenkwert: 0 = aus, ->1 = sehr träge
        max_steer=1.0,
        max_gap=15.0,          # [px] größerer Punktabstand = anderer Streckenteil
        warmup_steps=20,       # Schritte am Start ohne Lenkung (Kamera zoomt noch, Bild unbrauchbar)
    ):
        if mode not in (self.STANLEY, self.PURE_PURSUIT):
            raise ValueError(f"Unbekannter Modus '{mode}', erlaubt: 'stanley', 'pure_pursuit'")

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
        # Startphase: Die Kamera zoomt in den ersten ~50 Schritten noch hinein, die Pixel-
        # abstände stimmen nicht und die Spurerkennung liefert Unsinn. Das Auto steht am
        # Start gerade auf der Strecke -> einfach geradeaus fahren, sonst schaukelt es sich auf.
        self._step += 1
        if self._step <= self.warmup_steps:
            return 0.0

        path = self._prepare_path(trajectory)
        if path is None:
            # kein brauchbarer Pfad -> letzten Lenkwert langsam abbauen
            self._prev_steer *= 0.9
            return float(self._prev_steer)

        if self.mode == self.STANLEY:
            delta = self._stanley(path, speed)
        else:
            delta = self._pure_pursuit(path, speed)

        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        # Tiefpass gegen Zappeln
        delta = self.smoothing * self._prev_steer + (1.0 - self.smoothing) * delta
        self._prev_steer = delta
        return delta

    def reset(self):
        self._prev_steer = 0.0
        self._step = 0

    # ------------------------------------------------------------ Stanley
    def _stanley(self, path, speed):
        s, i = self._nearest_on_path(path)

        # Tangente über ein kurzes Stück ab dem nächsten Punkt mitteln
        p0 = self._point_at(path, s)
        p1 = self._point_at(path, s + self.heading_window)
        tangent = p1 - p0
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            norm = np.linalg.norm(tangent) + 1e-6
        t = tangent / norm

        # Heading-Fehler: Winkel zwischen Fahrtrichtung (0,-1) und Tangente, >0 = rechts
        heading_error = np.arctan2(t[0], -t[1])

        # vorzeichenbehaftete senkrechte Querabweichung, >0 = Pfad rechts vom Auto
        r = self._car_position - p0
        # (r x t: Pfad nach oben, Auto links davon -> e > 0 -> nach rechts lenken)
        e = r[0] * t[1] - r[1] * t[0]

        return heading_error + np.arctan2(self.k * e, self.k_soft + speed)

    # ------------------------------------------------------- Pure Pursuit
    def _pure_pursuit(self, path, speed):
        ld = np.clip(self.ld_min + self.ld_gain * speed, self.ld_min, self.ld_max)

        # erster Pfadpunkt vor dem Auto mit Abstand >= ld
        s_near, _ = self._nearest_on_path(path)
        dist = np.linalg.norm(path - self._car_position, axis=1)
        arc = self._arc_length(path)
        candidates = np.where((arc >= s_near) & (dist >= ld))[0]
        target = path[candidates[0]] if len(candidates) else path[-1]

        d = target - self._car_position
        ld_actual = max(np.linalg.norm(d), 1e-6)
        alpha = np.arctan2(d[0], -d[1])      # >0 = Ziel rechts

        return np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld_actual)

    # ------------------------------------------------------------ Helfer
    def _prepare_path(self, trajectory):
        """Bringt die Trajektorie in Fahrtrichtung geordnete Form und tastet sie
        mit 1 px Abstand neu ab. Schneidet andere sichtbare Streckenteile ab."""
        if trajectory is None:
            return None
        pts = np.asarray(trajectory, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            return None

        i = int(np.argmin(np.linalg.norm(pts - self._car_position, axis=1)))

        # Fahrtrichtung bestimmen: in welche Index-Richtung geht es im Bild nach oben?
        n = 5
        fwd_y = pts[min(i + n, len(pts) - 1)][1] - pts[i][1]
        bwd_y = pts[max(i - n, 0)][1] - pts[i][1]
        if bwd_y < fwd_y:
            pts = pts[::-1]
            i = len(pts) - 1 - i

        # zusammenhängendes Stück um i herum (Sprünge = anderer Streckenteil)
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

        # gleichmäßig neu abtasten (1 px)
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
