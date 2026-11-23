import numpy as np
from scipy import ndimage


class LaneDetection:
    """Detects the left and right lane boundary in the 96x96 camera image.

    Coordinates: image coordinates (x to the right, y down), unit pixels.
    Only numpy and scipy are used for the image processing.
    """

    # --- Image geometry (fixed for the 96x96 CarRacing image) ---
    CROP_HEIGHT = 77                 # only rows 0..76 are used (below: area behind the car and the status bar)
    IMAGE_WIDTH = 96
    CAR_X = 48                       # column of the car center
    CAR_ROWS = slice(66, 78)         # rows covered by the car (masked out)
    CAR_COLS = slice(45, 51)         # columns covered by the car (masked out)
    FRONT_ROWS = slice(56, 64)       # patch directly in front of the car, used to estimate the road color
    FRONT_COLS = slice(44, 52)
    SEARCH_START_ROW = 63            # first row above the car mask incl. its 2 px dilation

    # --- Road color estimation ---
    MIN_COLOR_COUNT = 20             # a color must cover at least this many pixels to count as "frequent"
    N_FREQUENT_COLORS = 8            # number of most frequent colors that are compared
    BG_CHANNEL_DIFF = (15, 30)       # grass/background checkerboard: the two colors differ by 15..30 in ONE channel
    CURB_COLORS = np.array([[255, 0, 0], [255, 255, 255]], dtype=np.float32)   # red/white curbs
    EXCLUDE_DIST = 12                # colors closer than this to grass/background/curb are never road
    ROAD_COLOR_TOL = 14.0            # max color distance to the road color (tiles vary by up to +5 per channel)
    MIN_ROAD_COLOR_TOL = 3.0         # lower limit if the background color is close to the road color

    # --- Cleanup and assignment ---
    MIN_ROAD_BLOB = 15               # [px] smaller road blobs are false positives
    MIN_HOLE = 10                    # [px] smaller holes in the road are filled (anti-aliasing at tile borders)
    MIN_SEGMENT_WIDTH = 5            # [px] shorter road segments in a row are anti-aliasing, not road
    EDGE_SEARCH_RADIUS = 3           # [px] search radius for the edge next to the end of a road segment

    def __init__(self):
        # Debug image of the last detection (other edges gray, left red, right blue), used by the test scripts
        self.debug_image = np.zeros((self.CROP_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect the left and right lane boundary.

        Args:
            image: RGB camera image of the simulation, shape (96, 96, 3), uint8.

        Returns:
            (left_boundary, right_boundary): two (N, 2) float arrays with the pixel
            coordinates [x, y] of the boundary edge pixels (unordered). An empty
            (0, 2) array means the boundary is not visible. The points are ordered
            along the road later in the path planning.
        """
        image = np.copy(image)[:self.CROP_HEIGHT, :].astype(np.float32)
        car_mask = np.zeros(image.shape[:2], dtype=bool)
        car_mask[self.CAR_ROWS, self.CAR_COLS] = True

        # With domain_randomize the road, background and grass colors are random.
        # So instead of fixed color thresholds, estimate the road color in every frame
        # and classify road pixels by their color distance (all 3 channels).
        road_color, tol = self._estimate_road_color(image, car_mask)
        color_dist = np.linalg.norm(image - road_color, axis=2)
        road = (color_dist < tol) & ~car_mask

        # Fill small holes (anti-aliasing at tile borders), remove small blobs (false positives)
        road = self._remove_small(road, self.MIN_ROAD_BLOB)
        road = ~self._remove_small(~road, self.MIN_HOLE)
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
        left_label = right_label = 0
        cols = np.arange(road.shape[1])
        for row in range(self.SEARCH_START_ROW, -1, -1):
            segments, n_seg = ndimage.label(road[row])
            valid = [s for s in range(1, n_seg + 1)
                     if np.count_nonzero(segments == s) >= self.MIN_SEGMENT_WIDTH]
            if not valid:
                continue
            dist = {s: np.abs(cols[segments == s] - self.CAR_X).min() for s in valid}
            best = min(dist, key=dist.get)
            seg_cols = cols[segments == best]
            start, end = seg_cols[0], seg_cols[-1]
            # The edge is not always exactly on the last road pixel (anti-aliasing, e.g. 102 → 127 → 202),
            # so take the nearest edge within EDGE_SEARCH_RADIUS. At the image border the boundary is not visible.
            left_label  = self._nearest_label(labels[row], start) if start > 0 else 0
            right_label = self._nearest_label(labels[row], end) if end < road.shape[1] - 1 else 0
            break

        left_mask  = (labels == left_label)  & (left_label  > 0)   # label 0 = background, not a lane boundary
        right_mask = (labels == right_label) & (right_label > 0)

        left_boundary  = self._mask_to_points(left_mask)
        right_boundary = self._mask_to_points(right_mask)

        # Debug image (RGB): other edges gray, left red, right blue
        self.debug_image = np.zeros((*edges.shape, 3), dtype=np.uint8)
        self.debug_image[edges]      = [80, 80, 80]
        self.debug_image[left_mask]  = [255, 0, 0]
        self.debug_image[right_mask] = [0, 0, 255]

        return left_boundary, right_boundary

    def _estimate_road_color(self, image: np.ndarray, car_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Estimate the road color and color tolerance from the image.

        Grass and background form a checkerboard of two colors that differ in only ONE
        channel by about 20–26 (this is how CarRacing generates them, also with domain_randomize).
        This color pair is searched for and marked as "not road". The road is then the area
        directly in front of the car, or – if there is grass (car next to the road) – the most
        frequent remaining color. Curbs (red/white) are excluded.

        Returns:
            (road_color, tol): RGB road color as float array of shape (3,) and the max.
            color distance for a pixel to count as road.
        """
        pixels = np.round(image[~car_mask]).astype(np.int32)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        order = np.argsort(-counts)
        colors, counts = colors[order], counts[order]
        frequent = colors[counts >= self.MIN_COLOR_COUNT][:self.N_FREQUENT_COLORS].astype(np.float32)

        # Find the grass/background pair
        lo, hi = self.BG_CHANNEL_DIFF
        background = []
        for i in range(len(frequent)):
            for j in range(i + 1, len(frequent)):
                diff = np.abs(frequent[i] - frequent[j])
                if np.count_nonzero(diff > 2) == 1 and lo <= diff.max() <= hi:
                    background = [frequent[i], frequent[j]]
                    break
            if background:
                break

        def is_excluded(c):
            excl = background + list(self.CURB_COLORS)
            return min(np.linalg.norm(c - e) for e in excl) < self.EXCLUDE_DIST

        # First choice: color directly in front of the car
        front = np.median(image[self.FRONT_ROWS, self.FRONT_COLS].reshape(-1, 3), axis=0)
        if not is_excluded(front):
            road_color = front
        else:
            candidates = [c for c in frequent if not is_excluded(c)]
            road_color = candidates[0] if candidates else front

        # Tolerance: tiles vary by up to +5 per channel (≈ 9 distance). If the background
        # color is close to the road color, shrink the tolerance accordingly.
        tol = self.ROAD_COLOR_TOL
        if background:
            d_bg = min(np.linalg.norm(road_color - b) for b in background)
            tol = max(min(tol, 0.5 * d_bg), self.MIN_ROAD_COLOR_TOL)
        return road_color, tol

    @staticmethod
    def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove connected regions with fewer than min_size pixels from the mask."""
        labels, n = ndimage.label(mask)
        if n == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        return keep[labels]

    @staticmethod
    def _mask_to_points(mask: np.ndarray) -> np.ndarray:
        """Convert a boolean mask into an (N, 2) float array of pixel coordinates [x, y]."""
        y, x = np.nonzero(mask)
        return np.column_stack([x, y]).astype(float)

    def _nearest_label(self, label_row: np.ndarray, col: int) -> int:
        """Label of the edge closest to col in this row (0 if none within EDGE_SEARCH_RADIUS)."""
        r = self.EDGE_SEARCH_RADIUS
        lo, hi = max(col - r, 0), min(col + r + 1, label_row.size)
        candidates = np.nonzero(label_row[lo:hi])[0] + lo
        if candidates.size == 0:
            return 0
        return label_row[candidates[np.argmin(np.abs(candidates - col))]]
