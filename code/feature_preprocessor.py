# Description: Class for loading, resizing, and computing difference between images.

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import os
import config
from utils import get_image_paths_from_annotation # Import utils for testing

class FeaturePreprocessing:
    """
    Handles loading images, resizing them, computing the difference,
    and optionally saving intermediate steps for debugging.
    """
    def __init__(self, target_size=(224, 224), mean=None, std=None, output_dir="./debug_preprocessing"):
        """
        Initializes the preprocessor.

        Args:
            target_size (tuple): The target (width, height) to resize images to *before* diff.
            mean (list, optional): Mean for normalization (if using preprocess_diff_for_resnet). Defaults to ImageNet mean.
            std (list, optional): Standard deviation for normalization. Defaults to ImageNet std.
            output_dir (str): Directory to save debug images if display=True.
        """
        # Ensure target_size is (width, height) for cv2.resize
        self.target_size = target_size
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]
        self.output_dir = output_dir
        # Only create dir if debugging might be enabled later
        # os.makedirs(self.output_dir, exist_ok=True) # Moved creation to _save_image

        # This transform is specific to ResNet input, not directly used by DETR processor
        # but kept here for completeness of the original class.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        # print(f"FeaturePreprocessor initialized with target_size={self.target_size}, output_dir='{self.output_dir}'")


    def load(self, img_path, display=False):
        """
        Loads and resizes an image.

        Args:
            img_path (str): Path to the image file.
            display (bool): If True, save the resized image to output_dir.

        Returns:
            np.ndarray: The resized image (BGR format), or None if loading fails.
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Image not found or could not be read: {img_path}")
            return None

        # cv2.resize expects (width, height)
        # img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        if display:
            filename = os.path.basename(img_path).split('.')[0]
            # self._save_image(img_resized, f"{filename}_resized")
            self._save_image(img, f"{filename}_unresized")

        return img # Returns BGR numpy array

    def compute_difference(self, img1, img2, img1_path=None, img2_path=None, display=False):
        """
        Computes the absolute difference between two images.

        Args:
            img1 (np.ndarray): First image (BGR).
            img2 (np.ndarray): Second image (BGR).
            img1_path (str, optional): Path of first image (for debug filename).
            img2_path (str, optional): Path of second image (for debug filename).
            display (bool): If True, save the difference image.

        Returns:
            np.ndarray: The absolute difference image (BGR).

        Raises:
            ValueError: If input images are None or do not have the same shape.
        """
        if img1 is None or img2 is None:
             raise ValueError("Input images cannot be None.")
        if img1.shape != img2.shape:
            raise ValueError(f"Input images must have the same shape. Got {img1.shape} and {img2.shape}")

        diff = cv2.absdiff(img1, img2)

        if display:
            # Use filenames to generate diff name
            base1 = os.path.basename(img1_path).split('.')[0] if img1_path else "img1"
            base2 = os.path.basename(img2_path).split('.')[0] if img2_path else "img2"
            self._save_image(diff, f"{base1}_vs_{base2}_diff")

        return diff # Returns BGR numpy array

    def preprocess_diff_for_resnet(self, diff_img):
        """
        Applies normalization transforms suitable for a standard ResNet input.
        NOTE: This is likely NOT needed when using the Hugging Face DETR processor,
              as the processor handles its own normalization.

        Args:
            diff_img (np.ndarray): Difference image (BGR).

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        rgb_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
        return self.transform(rgb_img) # Returns torch Tensor

    def _save_image(self, img, filename):
        """Helper function to save an image using matplotlib."""
        try:
            # Create output dir only when needed
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, f"{filename}.png")
            plt.figure(figsize=(8, 6))
            # Ensure image is converted to RGB for matplotlib display
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(filename)
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            # print(f"Saved debug image: {save_path}")
        except Exception as e:
            print(f"Error saving debug image {filename}: {e}")

# --- Main block for testing FeaturePreprocessing class ---
if __name__ == "__main__":
    print("--- Testing FeaturePreprocessing class ---")

    # --- Setup for testing ---
    # Use settings from config.py
    test_target_size = config.PREPROCESSOR_TARGET_SIZE
    test_debug_dir = config.PREPROCESSOR_DEBUG_DIR
    # Set display=True for this test run to generate output
    test_display = True
    print(f"Using Target Size (W, H): {test_target_size}")
    print(f"Using Debug Output Dir: {test_debug_dir}")
    print(f"Saving debug images: {test_display}")

    # Instantiate the class
    try:
        feature_preprocessor = FeaturePreprocessing(
            target_size=test_target_size,
            output_dir=test_debug_dir
        )
    except Exception as e:
        print(f"Error initializing FeaturePreprocessing: {e}")
        exit() # Cannot proceed if initialization fails

    # --- Find a sample image pair ---
    img1_path_test = None
    img2_path_test = None
    test_ann_dir = config.ANNOTATION_DIR
    test_image_dir = config.IMAGE_DATA_DIR

    print(f"\nSearching for a sample image pair using:")
    print(f"  Annotation Dir: {test_ann_dir}")
    print(f"  Image Dir: {test_image_dir}")

    if os.path.isdir(test_ann_dir):
        try:
            all_ann_files = [f for f in os.listdir(test_ann_dir) if f.endswith('_match.txt')]
            if all_ann_files:
                # Try to find the first annotation file for which images exist
                for ann_file in all_ann_files:
                    p1, p2 = get_image_paths_from_annotation(ann_file, test_image_dir)
                    if p1 and p2 and os.path.exists(p1) and os.path.exists(p2):
                        img1_path_test = p1
                        img2_path_test = p2
                        print(f"Found sample pair from annotation: {ann_file}")
                        print(f"  Image 1: {img1_path_test}")
                        print(f"  Image 2: {img2_path_test}")
                        break # Stop after finding the first valid pair
                if not img1_path_test:
                    print("Found annotation files, but couldn't find corresponding image pairs in the expected location.")
            else:
                print(f"No annotation files ending with '_match.txt' found in {test_ann_dir}.")
        except Exception as e:
            print(f"Error searching for sample pair: {e}")
    else:
        print(f"Annotation directory '{test_ann_dir}' not found.")

    # --- Perform tests if a pair was found ---
    if img1_path_test and img2_path_test:
        print("\nRunning tests with the found image pair...")
        try:
            # Test load()
            print("Testing load()...")
            img1 = feature_preprocessor.load(img1_path_test, display=test_display)
            img2 = feature_preprocessor.load(img2_path_test, display=test_display)

            if img1 is not None and img2 is not None:
                print(f"  Loaded img1 shape: {img1.shape}")
                print(f"  Loaded img2 shape: {img2.shape}")
                assert img1.shape[:2] == (test_target_size[1], test_target_size[0]), "Image 1 not resized correctly"
                assert img2.shape[:2] == (test_target_size[1], test_target_size[0]), "Image 2 not resized correctly"

                # Test compute_difference()
                print("\nTesting compute_difference()...")
                diff_img = feature_preprocessor.compute_difference(
                    img1, img2,
                    img1_path=img1_path_test,
                    img2_path=img2_path_test,
                    display=test_display
                )
                print(f"  Computed difference image shape: {diff_img.shape}")
                assert diff_img.shape == img1.shape, "Difference image shape mismatch"

                # Test preprocess_diff_for_resnet() (optional, as it's not used by DETR pipeline)
                # print("\nTesting preprocess_diff_for_resnet()...")
                # preprocessed_tensor = feature_preprocessor.preprocess_diff_for_resnet(diff_img)
                # print(f"  Preprocessed tensor shape: {preprocessed_tensor.shape}")
                # # Expected shape: [Channels, Height, Width] e.g., [3, 480, 640]
                # assert preprocessed_tensor.shape == (3, test_target_size[1], test_target_size[0]), "Tensor shape mismatch"

                print("\nFeature Preprocessing tests completed successfully.")
                if test_display:
                    print(f"Check the '{test_debug_dir}' directory for output images.")

            else:
                 print("\nSkipping difference calculation due to image loading failure.")

        except FileNotFoundError as e:
            print(f"Error during test (FileNotFound): {e}")
        except ValueError as e:
            print(f"Error during test (ValueError): {e}")
        except AssertionError as e:
             print(f"Assertion Error during test: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}")
    else:
        print("\nCould not find a valid image pair to run tests. Please ensure:")
        print(f"  - Annotation directory '{test_ann_dir}' exists and contains '_match.txt' files.")
        print(f"  - Image directory '{test_image_dir}' exists.")
        print(f"  - Annotation filenames correctly correspond to image pair folders and image files within them.")

    print("\n--- FeaturePreprocessing Testing Complete ---")