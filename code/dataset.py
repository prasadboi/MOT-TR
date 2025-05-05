# Description: PyTorch Dataset and DataLoader definitions using FeaturePreprocessing.

import os
import cv2
import torch
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Added DataLoader for testing collate_fn

# Import local modules
from utils import get_image_paths_from_annotation, parse_annotation_file
from feature_preprocessor import FeaturePreprocessing  # Import the new class
import config  # Import configuration

# Add imports needed for __main__ block
from model import load_processor


class MovedObjectDataset(Dataset):
    """
    PyTorch Dataset for loading image pairs, computing difference using
    FeaturePreprocessing, and preparing data for DETR fine-tuning.
    """

    def __init__(self, annotation_files, annotation_dir, image_dir, image_processor):
        """
        Args:
            annotation_files (list): List of annotation filenames for this dataset split.
            annotation_dir (str): Directory containing all annotation files.
            image_dir (str): Base directory where image pair folders reside.
            image_processor: Hugging Face image processor instance for DETR.
        """
        self.annotation_files = annotation_files
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.image_processor = image_processor

        # Instantiate the custom feature preprocessor using settings from config
        self.feature_preprocessor = FeaturePreprocessing(
            target_size=config.PREPROCESSOR_TARGET_SIZE,  # Use (width, height) from config
            output_dir=config.PREPROCESSOR_DEBUG_DIR,
        )
        # Control debug image saving via config
        self.save_debug_images = config.PREPROCESSOR_SAVE_DEBUG_IMAGES

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        """
        Loads and processes a single data sample using FeaturePreprocessing.
        """
        ann_filename = self.annotation_files[idx]
        ann_path = os.path.join(self.annotation_dir, ann_filename)

        # 1. Parse Annotation File
        moved_objects = parse_annotation_file(ann_path)

        # 2. Get Image Paths
        img1_path, img2_path = get_image_paths_from_annotation(
            ann_filename, self.image_dir
        )

        if not img1_path or not img2_path:
            return self._get_dummy_item()

        try:
            img1_resized = self.feature_preprocessor.load(
                img1_path, display=self.save_debug_images
            )
            img2_resized = self.feature_preprocessor.load(
                img2_path, display=self.save_debug_images
            )

            if img1_resized is None or img2_resized is None:
                return self._get_dummy_item()

            diff_img = self.feature_preprocessor.compute_difference(
                img1_resized,
                img2_resized,
                img1_path=img1_path,
                img2_path=img2_path,
                display=self.save_debug_images,
            )

            diff_img_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
            diff_pil = Image.fromarray(diff_img_rgb)

            target = {"image_id": idx, "annotations": []}
            for obj in moved_objects:
                x, y, w, h = obj["bbox"]
                coco_bbox = [x, y, x + w, y + h]
                target["annotations"].append(
                    {
                        "bbox": coco_bbox,
                        "category_id": obj["label"],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

            encoding = self.image_processor(
                images=diff_pil, annotations=target, return_tensors="pt"
            )

            # # Extract class_labels and boxes from the processor output
            # if "labels" in encoding and encoding["labels"]:
            #     label_dict = encoding["labels"][0]
            #     labels = label_dict["class_labels"]  # shape: [num_objects]
            #     boxes = label_dict["boxes"]  # shape: [num_objects, 4]
            # else:
            #     labels = torch.tensor([], dtype=torch.long)
            #     boxes = torch.empty((0, 4), dtype=torch.float32)

            # pixel_values = encoding["pixel_values"].squeeze(0)

            # return {"pixel_values": pixel_values, "labels": labels, "boxes": boxes}
            label_dict = (
                encoding["labels"][0]
                if encoding.get("labels")
                else {
                    "class_labels": torch.tensor([], dtype=torch.long),
                    "boxes":        torch.empty((0, 4), dtype=torch.float32),
                }
            )

            pixel_values = encoding["pixel_values"].squeeze(0)

            return {
                "pixel_values": pixel_values,
                "labels":       label_dict,
            }

        except FileNotFoundError as e:
            return self._get_dummy_item()
        except ValueError as e:
            return self._get_dummy_item()
        except Exception as e:
            print(f"An unexpected error occurred processing {ann_filename}: {e}")
            return self._get_dummy_item()

    def _get_dummy_item(self):
        """Returns a dummy item structure for error cases."""
        # Ensure pixel_values is None to be filtered by collate_fn
        return {
            "pixel_values": None,
            "labels": torch.tensor([], dtype=torch.long),
            "boxes": torch.tensor([], dtype=torch.float32),
        }


# def collate_fn(batch):
#     """
#     Custom collate function to handle batching of DETR inputs.
#     Filters out None items resulting from errors in __getitem__.
#     """
#     # Filter out items where pixel_values is None (indicates loading error)
#     valid_batch = [
#         item
#         for item in batch
#         if item is not None and item.get("pixel_values") is not None
#     ]

#     if not valid_batch:
#         # If the entire batch is invalid, return an empty structure
#         # print("Warning: Entire batch is invalid, skipping.")
#         return {"pixel_values": torch.empty(0), "labels": [], "boxes": []}

#     try:
#         pixel_values = torch.stack([item["pixel_values"] for item in valid_batch])
#         # Labels and boxes are expected as lists of tensors for DETR
#         labels = [item["labels"] for item in valid_batch]
#         boxes = [item["boxes"] for item in valid_batch]

#         return {"pixel_values": pixel_values, "labels": labels, "boxes": boxes}
#     except Exception as e:
#         print(f"Error during collation: {e}. Returning empty batch.")
#         # Handle potential errors during stacking (e.g., inconsistent tensor sizes if processor failed)
#         return {"pixel_values": torch.empty(0), "labels": [], "boxes": []}

def collate_fn(batch):
    # filter out any None / invalid examples if you have that logic
    valid_batch = [b for b in batch if b is not None]

    pixel_values = torch.stack([item["pixel_values"] for item in valid_batch])

    # now a list of dicts, each with "class_labels" & "boxes"
    labels = [item["labels"] for item in valid_batch]

    return {
        "pixel_values": pixel_values,
        "labels":       labels,
    }


# --- Main block for testing Dataset and Collate Function ---
if __name__ == "__main__":
    print("--- Testing Dataset and Collate Function ---")

    # --- Setup ---
    print("\nSetting up for test...")
    # Load processor
    try:
        test_processor = load_processor()
    except Exception as e:
        print(f"Failed to load processor, cannot run dataset tests: {e}")
        exit()

    # Get list of annotation files
    test_ann_dir = config.ANNOTATION_DIR
    test_image_dir = config.IMAGE_DATA_DIR
    all_ann_files = []
    if os.path.isdir(test_ann_dir):
        try:
            all_ann_files = [
                f for f in os.listdir(test_ann_dir) if f.endswith("_match.txt")
            ]
            if not all_ann_files:
                print(f"Warning: No annotation files found in {test_ann_dir}.")
            else:
                print(f"Found {len(all_ann_files)} annotation files.")
        except Exception as e:
            print(f"Error listing annotation files: {e}")
    else:
        print(f"Error: Annotation directory not found: {test_ann_dir}")
        exit()  # Cannot proceed without annotation files

    if not all_ann_files:
        print("No annotation files available to create dataset. Exiting test.")
        exit()

    # Use a small subset for testing
    num_test_files = min(80, len(all_ann_files))
    test_files = all_ann_files[:num_test_files]
    print(f"Using {num_test_files} files for testing.")

    # Instantiate dataset
    try:
        test_dataset = MovedObjectDataset(
            annotation_files=test_files,
            annotation_dir=test_ann_dir,
            image_dir=test_image_dir,
            image_processor=test_processor,
        )
        print(f"Dataset instantiated with {len(test_dataset)} items.")
    except Exception as e:
        print(f"Error instantiating dataset: {e}")
        exit()

    # --- Test __getitem__ ---
    print("\nTesting __getitem__ (loading first item)...")
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        if item and item.get("pixel_values") is not None:
            print("Successfully loaded first item:")
            print(f"  pixel_values shape: {item['pixel_values'].shape}")
            print(f"  labels: {item['labels']}")
            print(f"  labels shape: {item['labels'].shape}")
            print(f"  boxes: {item['boxes']}")
            print(f"  boxes shape: {item['boxes'].shape}")
        elif item and item.get("pixel_values") is None:
            print(
                "First item loaded as a dummy item (pixel_values is None). Check logs for warnings."
            )
        else:
            print("Failed to load first item (returned None or unexpected structure).")
    else:
        print("Dataset is empty, cannot test __getitem__.")

    # --- Test collate_fn ---
    print("\nTesting collate_fn...")
    # Create a small batch manually by getting a few items
    batch_items = []
    num_collate_test = min(30, len(test_dataset))  # Test with up to 3 items
    print(f"Attempting to load {num_collate_test} items for batch...")
    items_loaded = 0
    for i in range(num_collate_test):
        try:
            item = test_dataset[i]
            batch_items.append(item)  # Append even if it's a dummy item
            if item and item.get("pixel_values") is not None:
                items_loaded += 1
        except Exception as e:
            print(f"Error loading item {i} for batch: {e}")

    print(
        f"Loaded {len(batch_items)} items ({items_loaded} valid, {len(batch_items) - items_loaded} dummy/error)."
    )

    if batch_items:
        try:
            collated_batch = collate_fn(batch_items)
            print("Collate function executed.")
            if collated_batch and collated_batch.get("pixel_values").numel() > 0:
                print("  Collated Batch Contents:")
                print(
                    f"    pixel_values shape: {collated_batch['pixel_values'].shape}"
                )  # Should be [Valid_Batch_Size, C, H, W]
                print(
                    f"    Number of label tensors: {len(collated_batch['labels'])}"
                )  # Should match Valid_Batch_Size
                print(
                    f"    Number of box tensors: {len(collated_batch['boxes'])}"
                )  # Should match Valid_Batch_Size
                # Print shape of first label/box tensor if available
                if collated_batch["labels"]:
                    print(
                        f"Shape of first labels tensor: {collated_batch['labels'][0]=}"
                    )
                if collated_batch["boxes"]:
                    print(f"Shape of first boxes tensor: {collated_batch['boxes'][0]}")
            elif collated_batch and collated_batch.get("pixel_values").numel() == 0:
                print(
                    "  Collate function returned an empty batch (likely all input items were invalid)."
                )
            else:
                print("  Collate function returned None or unexpected structure.")

        except Exception as e:
            print(f"Error executing collate_fn: {repr(e)}")
            # import traceback
            # traceback.print_exc()
    else:
        print("No items were loaded into the batch, cannot test collate_fn.")

    # --- Optional: Test with DataLoader ---
    print("\nTesting with DataLoader...")
    try:
        # Use a small batch size for testing
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 for easier debugging in main thread
        )
        print("DataLoader instantiated.")
        # Iterate over one batch
        first_dl_batch = next(iter(test_loader))
        print("Successfully retrieved first batch from DataLoader:")
        print(f"  pixel_values shape: {first_dl_batch['pixel_values'].shape}")
        print(f"  Number of label tensors: {len(first_dl_batch['labels'])}")
        print(f"  Number of box tensors: {len(first_dl_batch['boxes'])}")
    except StopIteration:
        print(
            "DataLoader iteration finished unexpectedly (maybe dataset was empty or all items failed)."
        )
    except Exception as e:
        print(f"Error testing with DataLoader: {e}")

    print("\n--- Dataset and Collate Function Testing Complete ---")
