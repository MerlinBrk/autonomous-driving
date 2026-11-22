import numpy as np
from scipy import ndimage


class LaneDetection:

    def __init__(self):
        # Debug image of the last detection (other edges gray, left red, right blue), used by the test scripts
        self.debug_image = np.zeros((77, 96, 3), dtype=np.uint8)

    def detect(self,image):
        image = np.copy(image)[:77,:].astype(np.float32)
        car_mask = np.zeros(image.shape[:2], dtype=bool)
        car_mask[66:78, 45:51] = True

        # With domain_randomize the road, background and grass colors are random.
        # So instead of fixed color thresholds, estimate the road color in every frame
        # and classify road pixels by their color distance (all 3 channels).
        road_color, tol = self._estimate_road_color(image, car_mask)
        color_dist = np.linalg.norm(image - road_color, axis=2)
        road = (color_dist < tol) & ~car_mask

        # Fill small holes (anti-aliasing at tile borders), remove small blobs (false positives)
        road = self._remove_small(road, 15)
        road = ~self._remove_small(~road, 10)
        road &= ~car_mask

        # Boundaries = 2 px wide band around the road <-> non-road border (last road pixel
        # + first non-road pixel, matches the true edge best). The car counts as road and
        # the image border does not count as a boundary (border_value=1), otherwise the left
        # and right boundary would be connected via the top image border.
        road_with_car = road | car_mask
        inner = ndimage.binary_erosion(road_with_car, structure=np.ones((3, 3)), border_value=1)
        outer = ndimage.binary_dilation(road_with_car, structure=np.ones((3, 3)))
        edges = outer & ~inner
        edges[ndimage.binary_dilation(car_mask, iterations=2)] = False   # ignore edges of the car mask
        labels, _ = ndimage.label(edges, structure=np.ones((3, 3)))

        # Search bottom-up for the first row containing road and take the road segment
        # closest to the car. Its ends are the left and right boundary –
        # this also works when the car is next to the road.
        car_x = 48
        left_label = right_label = 0
        cols = np.arange(road.shape[1])
        for row in range(63, -1, -1):       # start above the car mask (incl. dilation)
            segments, n_seg = ndimage.label(road[row])
            # short segments (< 5 px) are not road but anti-aliasing, e.g. between grass and map border
            valid = [s for s in range(1, n_seg + 1) if np.count_nonzero(segments == s) >= 5]
            if not valid:
                continue
            dist = {s: np.abs(cols[segments == s] - car_x).min() for s in valid}
            best = min(dist, key=dist.get)
            seg_cols = cols[segments == best]
            start, end = seg_cols[0], seg_cols[-1]
            # The edge is not always exactly on the last road pixel (anti-aliasing, e.g. 102 → 127 → 202),
            # so take the nearest edge within 3 pixels. At the image border the boundary is not visible.
            left_label  = self._nearest_label(labels[row], start) if start > 0 else 0
            right_label = self._nearest_label(labels[row], end) if end < road.shape[1] - 1 else 0
            break

        left_mask  = (labels == left_label)  & (left_label  > 0)   # label 0 = background, not a lane boundary
        right_mask = (labels == right_label) & (right_label > 0)

        left_boundary  = self._order_points(left_mask, car_x)
        right_boundary = self._order_points(right_mask, car_x)

        # Debug image (RGB): other edges gray, left red, right blue
        self.debug_image = np.zeros((*edges.shape, 3), dtype=np.uint8)
        self.debug_image[edges]      = [80, 80, 80]
        self.debug_image[left_mask]  = [255, 0, 0]
        self.debug_image[right_mask] = [0, 0, 255]

        return left_boundary, right_boundary

    @staticmethod
    def _estimate_road_color(image, car_mask):
        """
        Estimate the road color and color tolerance from the image.

        Grass and background form a checkerboard of two colors that differ in only ONE
        channel by about 20–26 (this is how CarRacing generates them, also with domain_randomize).
        This color pair is searched for and marked as "not road". The road is then the area
        directly in front of the car, or – if there is grass (car next to the road) – the most
        frequent remaining color. Curbs (red/white) are excluded.
        """
        pixels = np.round(image[~car_mask]).astype(np.int32)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        order = np.argsort(-counts)
        colors, counts = colors[order], counts[order]
        frequent = colors[counts >= 20][:8].astype(np.float32)

        # Find the grass/background pair
        background = []
        for i in range(len(frequent)):
            for j in range(i + 1, len(frequent)):
                diff = np.abs(frequent[i] - frequent[j])
                if np.count_nonzero(diff > 2) == 1 and 15 <= diff.max() <= 30:
                    background = [frequent[i], frequent[j]]
                    break
            if background:
                break

        curbs = np.array([[255, 0, 0], [255, 255, 255]], dtype=np.float32)

        def is_excluded(c):
            excl = background + list(curbs)
            return min(np.linalg.norm(c - e) for e in excl) < 12

        # First choice: color directly in front of the car
        front = np.median(image[56:64, 44:52].reshape(-1, 3), axis=0)
        if not is_excluded(front):
            road_color = front
        else:
            candidates = [c for c in frequent if not is_excluded(c)]
            road_color = candidates[0] if candidates else front

        # Tolerance: tiles vary by up to +5 per channel (≈ 9 distance). If the background
        # color is close to the road color, shrink the tolerance accordingly.
        tol = 14.0
        if background:
            d_bg = min(np.linalg.norm(road_color - b) for b in background)
            tol = max(min(tol, 0.5 * d_bg), 3.0)
        return road_color, tol

    @staticmethod
    def _remove_small(mask, min_size):
        """Remove connected regions with fewer than min_size pixels from the mask."""
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        return keep[labels]

    @staticmethod
    def _order_points(mask, car_x, max_step=3.0):
        """
        Return edge pixels as an (N, 2) array [x, y], ordered along the line.
        Starts at the pixel closest to the car (bottom row); then always jumps to the
        nearest unvisited pixel. If that one is farther than max_step away, the line
        ends (outliers such as leftover curb pixels are dropped).
        """
        y, x = np.nonzero(mask)
        if x.size == 0:
            return np.empty((0, 2))
        points = np.column_stack([x, y]).astype(float)

        start = np.lexsort((np.abs(x - car_x), -y))[0]   # largest y, ties broken by closeness to car_x
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
        """Label of the edge closest to col in this row (0 if none within radius)."""
        lo, hi = max(col - radius, 0), min(col + radius + 1, label_row.size)
        candidates = np.nonzero(label_row[lo:hi])[0] + lo
        if candidates.size == 0:
            return 0
        return label_row[candidates[np.argmin(np.abs(candidates - col))]]
