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
        self.debug_image = self.debug_image[..., 1]

        return np.array([]), np.array([])
