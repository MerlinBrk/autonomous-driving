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
        e = edges.astype(np.uint8) * 255
        self.debug_image = np.stack([e, e, e], axis=2)         # für die Anzeige auf 3 Kanäle
        return np.array([]), np.array([])
