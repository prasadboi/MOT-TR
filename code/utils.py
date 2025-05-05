# utils.py
# Description: Utility functions for file parsing and data handling.

import os
import config # Import config for testing paths
def get_image_paths_from_annotation(ann_filename, image_base_dir):
    """
    Derives the paths to the two image frames based on the annotation filename.
    Assumes annotation filename is like '{folder_name}-{img1_base}-{img2_base}_match.txt'
    and images are inside a folder named '{folder_name}'.

    Args:
        ann_filename (str): Name of the annotation file.
        image_base_dir (str): Base directory containing the pair folders.

    Returns:
        tuple: (img1_path, img2_path) or (None, None) if parsing fails.
    """
    try:
        base_name = ann_filename.replace('_match.txt', '')
        parts = base_name.split('-')
        if len(parts) < 3:
            print(f"Warning: Unexpected annotation filename format: {ann_filename}")
            return None, None

        folder_name = parts[0]
        img1_base = parts[1]
        img2_base = parts[2]

        pair_folder_path = os.path.join(image_base_dir, folder_name)
        img1_filename = f"{img1_base}.png"
        img2_filename = f"{img2_base}.png"

        img1_path = os.path.join(pair_folder_path, img1_filename)
        img2_path = os.path.join(pair_folder_path, img2_filename)

        return img1_path, img2_path
    except Exception as e:
        print(f"Error parsing annotation filename '{ann_filename}': {e}")
        return None, None

def parse_annotation_file(ann_path):
    """
    Parses the matched annotation file.
    Format: <object_id> <x> <y> <w> <h> <object_type>
    Each object appears twice (old and new). Outputs all lines as dictionaries.

    Args:
        ann_path (str): Path to the annotation file.

    Returns:
        list: List of dictionaries, each containing {'id': obj_id, 'bbox': [x, y, w, h], 'label': type}
              for every line in the file. Returns empty list on error or if no valid lines found.
    """
    objects = []
    try:
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                try:
                    obj_id = int(parts[0])
                    bbox = [float(p) for p in parts[1:5]] # [x, y, w, h]
                    label = int(parts[5])
                except ValueError:
                    continue # Skip lines with non-numeric data

                objects.append({
                    'id': obj_id,
                    'bbox': bbox,
                    'label': label
                })

        return objects
    except FileNotFoundError:
        print(f"Error: Annotation file not found {ann_path}")
        return []
    except Exception as e:
        print(f"Error reading or parsing annotation file {ann_path}: {e}")
        return []


# --- Main block for testing utility functions ---#

if __name__ == "__main__":
    print("--- Testing Utility Functions ---")

    # --- Test 1: get_image_paths_from_annotation ---
    print("\nTesting get_image_paths_from_annotation...")
    test_ann_filename = "S_000200_02_000479_000635_2160-S_000200_02_000479_000635_2400_match.txt"
    test_image_base_dir = config.IMAGE_DATA_DIR
    print(f"Input annotation filename: {test_ann_filename}")
    print(f"Input image base directory: {test_image_base_dir}")

    img1_path, img2_path = get_image_paths_from_annotation(test_ann_filename, test_image_base_dir)

    if img1_path and img2_path:
        print(f"  Expected Image 1 Path: {img1_path}")
        print(f"  Expected Image 2 Path: {img2_path}")
        pair_folder = os.path.dirname(img1_path)
        print(f"  Derived Pair Folder: {pair_folder}")
        print(f"  Pair Folder Exists: {os.path.isdir(pair_folder)}")
    else:
        print("  Failed to derive image paths.")

    print("\nTesting with invalid filename...")
    invalid_ann_filename = "Invalid_File_Name.txt"
    img1_path_invalid, img2_path_invalid = get_image_paths_from_annotation(invalid_ann_filename, test_image_base_dir)
    if not img1_path_invalid and not img2_path_invalid:
        print("  Correctly handled invalid filename (returned None).")
    else:
        print("  Incorrectly handled invalid filename.")

    # --- Test 2: parse_annotation_file ---
    print("\nTesting parse_annotation_file...")
    test_ann_dir = config.ANNOTATION_DIR
    print(f"Using annotation directory from config: {test_ann_dir}")

    first_ann_file = "S_000200_02_000479_000635_2160-S_000200_02_000479_000635_2400_match.txt"
    if os.path.isdir(test_ann_dir):
        try:
            all_files = [f for f in os.listdir(test_ann_dir) if f.endswith('_match.txt')]
            if all_files:
                first_ann_file = all_files[0]
                print(f"Found sample annotation file: {first_ann_file}")
            else:
                print(f"No annotation files ending with '_match.txt' found in {test_ann_dir}.")
        except Exception as e:
            print(f"Error listing files in {test_ann_dir}: {e}")
    else:
        print(f"Annotation directory '{test_ann_dir}' not found. Cannot perform parse test.")

    if first_ann_file:
        test_ann_path = os.path.join(test_ann_dir, first_ann_file)
        print(f"Parsing file: {test_ann_path}")
        parsed_objects = parse_annotation_file(test_ann_path)

        if parsed_objects:
            print(f"  Successfully parsed {len(parsed_objects)} moved objects.")
            for idx, obj in enumerate(parsed_objects):
                print(f"  Object {idx}:")
                print(f"    ID: {obj.get('id')}")
                print(f"    BBox: {obj.get('bbox')}")
                print(f"    Label: {obj.get('label')}")
        elif os.path.exists(test_ann_path):
            print("  Parsing completed, but no moved objects were extracted (or file was empty/malformed).")
        # else case handled by FileNotFoundError print within the function
    print("\n--- Utility Function Testing Complete ---")