# config.py
# Description: Configuration settings for the DETR fine-tuning pipeline.

import torch
import os # Added for main block

# --- Paths ---
ANNOTATION_DIR = '../data/matched_annotations' # Directory containing the GT annotation files
IMAGE_DATA_DIR = '../data/base/cv_data_hw2/data'    # Base directory where pair folders (e.g., Pair_S_...) reside
OUTPUT_DIR = '../output'     # Where the fine-tuned model and results will be saved
MODEL_SAVE_DIR = f'{OUTPUT_DIR}/models'
LOGGING_DIR = f"{OUTPUT_DIR}/logs"        # Directory for training logs
# Add path for debug images from feature_preprocessor.py
PREPROCESSOR_DEBUG_DIR = f"{OUTPUT_DIR}/debug_preprocessing"

# --- Model Configuration ---
CHECKPOINT = "facebook/detr-resnet-50"    # Pre-trained DETR model checkpoint
# CHECKPOINT = "facebook/detr-resnet-101" # Alternative backbone

# --- Feature Preprocessor Configuration ---
# Target size (width, height) for resizing images *before* computing difference
# Used by FeaturePreprocessing class in feature_preprocessor.py
PREPROCESSOR_TARGET_SIZE = (640, 480) 
PREPROCESSOR_SAVE_DEBUG_IMAGES = False # Set to True to save debug images during preprocessing

# --- Dataset Parameters ---
NUM_CLASSES = 6 # 0: Unknown, 1: Person, 2: Car, 3: Other Vehicle, 4: Other Object, 5: Bike
# Model head will have NUM_CLASSES + 1 outputs (extra for 'no object' which has been shown as Unknown in the dataset)

# --- Training Parameters ---
TRAIN_TEST_SPLIT = 0.95 # create a very small set of training samples for testing
# TRAIN_TEST_SPLIT = 0.20 # for main training run
BATCH_SIZE = 1 # For testing
# BATCH_SIZE = 4 # For main training run
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
GRAD_ACCUM_STEPS = 1 # For testing
# GRAD_ACCUM_STEPS = 2 # For main training run
RANDOM_SEED = 42
DATALOADER_NUM_WORKERS = 2 # Number of workers for DataLoader

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation ---
# Add any evaluation-specific configurations here if needed, e.g., IoU threshold
EVAL_THRESHOLD = 0.5 # Confidence threshold for post-processing detections


# --- Main block for testing configuration values ---
if __name__ == "__main__":
    print("--- Configuration Settings ---")
    print(f"Annotation Directory: {ANNOTATION_DIR}")
    print(f"Image Data Directory: {IMAGE_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Log Directory: {LOGGING_DIR}")
    print(f"Preprocessor Debug Directory: {PREPROCESSOR_DEBUG_DIR}")
    print("-" * 20)
    print(f"DETR Checkpoint: {CHECKPOINT}")
    print("-" * 20)
    print(f"Preprocessor Target Size (W, H): {PREPROCESSOR_TARGET_SIZE}")
    print(f"Save Preprocessor Debug Images: {PREPROCESSOR_SAVE_DEBUG_IMAGES}")
    print("-" * 20)
    print(f"Number of Classes: {NUM_CLASSES}")
    print("-" * 20)
    print(f"Train/Test Split: {TRAIN_TEST_SPLIT}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("-" * 20)
    print(f"Device: {DEVICE}")
    print(f"Evaluation Threshold: {EVAL_THRESHOLD}")
    print("-" * 20)

    # Check if directories exist (optional sanity check)
    print("Checking directory existence:")
    print(f"  Annotation Dir Exists: {os.path.isdir(ANNOTATION_DIR)}")
    print(f"  Image Data Dir Exists: {os.path.isdir(IMAGE_DATA_DIR)}")
    # Output directories might not exist before training
    print(f"  Output Dir Exists: {os.path.isdir(OUTPUT_DIR)}")