import cv2
import numpy as np
from matplotlib import pyplot as plt

class FeaturePreprocessing:
    def compute_difference(self, img1, img2, display=False):
        """
        Compute the absolute difference between two images.
        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.
            display (bool): If True, display the difference image.
        Returns:
            np.ndarray: The absolute difference image.
        """
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape.")
        diff = cv2.absdiff(img1, img2)
        if display:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
            plt.title("Image Difference")
            plt.axis('off')
            plt.show()
        return diff