import numpy as np
from scipy import ndimage


class LaneDetection:

    def __init__(self):
        pass

    def detect(self,image):
        image = np.copy(image)[:77,:]
        car_mask = np.zeros(image.shape[:2], dtype=bool)
        car_mask[66:78, 45:51] = True

        self.debug_image = np.copy(image)
        self.debug_image[car_mask] = [102, 102, 102]
        gray = self.debug_image[..., 1].astype(np.float32)   # float, sonst Überlauf bei negativen Gradienten (uint8)

        gx = ndimage.sobel(gray, axis=1)   # Änderung nach rechts  → senkrechte Kanten
        gy = ndimage.sobel(gray, axis=0)   # Änderung nach unten   → waagrechte Kanten

        magnitude = np.hypot(gx, gy)
        edges = magnitude > 200            # Gras-Muster ≈ 104, Gras↔Straße ≈ 400–512
        edges[ndimage.binary_dilation(car_mask, iterations=2)] = False   # Kanten der Auto-Maske ignorieren
        labels, n = ndimage.label(edges, structure=np.ones((3, 3)))

        # Straßenpixel: Grünkanal ≈ 102–107 (Gras 204/230, Randsteine 0/255)
        road = (gray > 90) & (gray < 120) & ~car_mask

        # Von unten nach oben die erste Zeile mit Straße suchen und dort das Straßenstück nehmen,
        # das dem Auto am nächsten ist. Dessen Enden sind linker und rechter Rand –
        # das funktioniert auch, wenn das Auto neben der Straße steht.
        car_x = 48
        left_label = right_label = 0
        cols = np.arange(road.shape[1])
        for row in range(63, -1, -1):       # erst oberhalb der Auto-Maske (inkl. Dilatation) starten
            segments, n_seg = ndimage.label(road[row])
            # kurze Stücke (< 5 px) sind keine Straße, sondern Kantenglättung z. B. zwischen Gras und Kartenrand
            valid = [s for s in range(1, n_seg + 1) if np.count_nonzero(segments == s) >= 5]
            if not valid:
                continue
            dist = {s: np.abs(cols[segments == s] - car_x).min() for s in valid}
            best = min(dist, key=dist.get)
            seg_cols = cols[segments == best]
            start, end = seg_cols[0], seg_cols[-1]
            # Kante liegt nicht immer genau auf dem letzten Straßenpixel (Kantenglättung, z. B. 102 → 127 → 202),
            # daher im Umkreis von 3 Pixeln die nächste Kante nehmen. Am Bildrand ist der Rand nicht sichtbar.
            left_label  = self._nearest_label(labels[row], start) if start > 0 else 0
            right_label = self._nearest_label(labels[row], end) if end < road.shape[1] - 1 else 0
            break

        left_mask  = (labels == left_label)  & (left_label  > 0)   # Label 0 = Hintergrund, nicht als Spur nehmen
        right_mask = (labels == right_label) & (right_label > 0)

        left_boundary  = self._order_points(left_mask, car_x)
        right_boundary = self._order_points(right_mask, car_x)

        # Debug-Bild (RGB): übrige Kanten grau, links rot, rechts blau
        self.debug_image = np.zeros((*edges.shape, 3), dtype=np.uint8)
        self.debug_image[edges]      = [80, 80, 80]
        self.debug_image[left_mask]  = [255, 0, 0]
        self.debug_image[right_mask] = [0, 0, 255]

        return left_boundary, right_boundary

    @staticmethod
    def _order_points(mask, car_x, max_step=3.0):
        """
        Kantenpixel als (N, 2)-Array [x, y] zurückgeben, sortiert entlang der Linie.
        Start ist der Pixel, der dem Auto am nächsten ist (unterste Zeile); danach wird
        immer zum nächsten noch nicht besuchten Pixel gesprungen. Ist der nächste weiter
        als max_step entfernt, endet die Linie (Ausreißer wie Randstein-Reste fallen weg).
        """
        y, x = np.nonzero(mask)
        if x.size == 0:
            return np.empty((0, 2))
        points = np.column_stack([x, y]).astype(float)

        start = np.lexsort((np.abs(x - car_x), -y))[0]   # größtes y, bei Gleichstand nah an car_x
        order = [start]
        visited = np.zeros(len(points), dtype=bool)
        visited[start] = True

        for _ in range(len(points) - 1):
            dist = np.linalg.norm(points - points[order[-1]], axis=1)
            dist[visited] = np.inf
            nxt = np.argmin(dist)
            if dist[nxt] > max_step:
                break
            order.append(nxt)
            visited[nxt] = True

        return points[order]

    @staticmethod
    def _nearest_label(label_row, col, radius=3):
        """Label der Kante, die in dieser Zeile am nächsten an col liegt (0, wenn keine in Reichweite)."""
        lo, hi = max(col - radius, 0), min(col + radius + 1, label_row.size)
        candidates = np.nonzero(label_row[lo:hi])[0] + lo
        if candidates.size == 0:
            return 0
        return label_row[candidates[np.argmin(np.abs(candidates - col))]]
