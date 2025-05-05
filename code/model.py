# model.py
# Description: Functions for loading the DETR model and processor.

from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Assuming config.py is in the same directory or Python path
import config
import torch  # Added for main block


def load_processor():
    """Loads the image processor from the specified checkpoint."""
    print(f"Loading image processor from checkpoint: {config.CHECKPOINT}")
    try:
        processor = AutoImageProcessor.from_pretrained(config.CHECKPOINT)
        print("Image processor loaded successfully.")
        return processor
    except Exception as e:
        print(f"Error loading image processor: {e}")
        raise  # Re-raise the exception to halt execution if processor fails


def load_model():
    """
    Loads the DETR model from the specified checkpoint and modifies the
    classification head for the number of classes in the custom dataset.

    Note:
    - Class label 0 is reserved for the 'unknown' class (not background).
    - Classes 1 to NUM_CLASSES-1 are the actual object classes.
    - The DETR model will add its own 'no object' (background) class internally.
    - The dataset should NOT use a label for 'no object' (background).
    """
    print(f"Loading DETR model from checkpoint: {config.CHECKPOINT}")

    # id2label: 0 -> 'unknown', 1 -> 'class_1', ..., N -> 'class_N'
    id2label = {0: "unknown"}
    id2label.update({i: f"class_{i}" for i in range(1, config.NUM_CLASSES)})

    label2id = {v: k for k, v in id2label.items()}

    try:
        model = AutoModelForObjectDetection.from_pretrained(
            config.CHECKPOINT,
            num_labels=config.NUM_CLASSES,  # No extra class; 0 is 'no object'
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        print("Model loaded. Classification head replaced.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise  # Re-raise the exception


# --- Main block for testing model loading ---
if __name__ == "__main__":
    print("--- Testing Model Loading Functions ---")

    # --- Test 1: Load Processor ---
    print("\nTesting load_processor()...")
    try:
        processor = load_processor()
        print("Processor details:")
        print(processor)
    except Exception as e:
        print(f"Failed to load processor: {e}")
        # Decide if you want to exit or continue to model loading test
        # exit()

    # --- Test 2: Load Model ---
    print("\nTesting load_model()...")
    try:
        model = load_model()
        print("Model details:")
        # print(model) # This can be very verbose, maybe print config or specific layers
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  Number of parameters: {model.num_parameters()}")
        # Check the classification head
        if hasattr(model, "class_labels_classifier"):
            print(
                f"  Classification head output size: {model.class_labels_classifier.out_features}"
            )
            expected_size = config.NUM_CLASSES + 1
            if model.class_labels_classifier.out_features == expected_size:
                print(f"  Classification head size matches config ({expected_size}).")
            else:
                print(
                    f"  WARNING: Classification head size ({model.class_labels_classifier.out_features}) does NOT match config ({expected_size})."
                )
        else:
            print(
                "  Could not find attribute 'class_labels_classifier' to check head size."
            )

        # Optional: Move model to device specified in config
        print(f"\nAttempting to move model to device: {config.DEVICE}")
        try:
            model.to(config.DEVICE)
            # Simple check: see if a parameter is on the expected device
            param_device = next(model.parameters()).device
            print(
                f"  Model successfully moved. Example parameter device: {param_device}"
            )
            if str(param_device) != str(config.DEVICE):
                print(
                    f"  WARNING: Parameter device ({param_device}) does not match config device ({config.DEVICE}). Check GPU availability/CUDA setup."
                )
        except Exception as e:
            print(f"  Error moving model to device {config.DEVICE}: {e}")

    except Exception as e:
        print(f"Failed to load model: {e}")

    print("\n--- Model Loading Testing Complete ---")
