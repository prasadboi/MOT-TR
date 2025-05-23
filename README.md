# DETR Fine-tuning Pipeline for Moved Object Detection

This repository provides a robust, reproducible pipeline for fine-tuning [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) on custom datasets for moved object detection. The pipeline covers dataset preparation, preprocessing (including image diffs), model adaptation, staged training, evaluation, and analysis.

For a comprehensive technical explanation, see [docs/detr_training_pipeline_report.md](docs/detr_training_pipeline_report.md).

---

## Table of Contents

- [DETR Fine-tuning Pipeline for Moved Object Detection](#detr-fine-tuning-pipeline-for-moved-object-detection)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Setup \& Requirements](#setup--requirements)
  - [Dataset \& Annotation Format](#dataset--annotation-format)
  - [Pipeline Usage](#pipeline-usage)
    - [1. Ground Truth Annotation](#1-ground-truth-annotation)
    - [2. Configure the Pipeline](#2-configure-the-pipeline)
    - [3. Training](#3-training)
    - [4. Evaluation \& Results](#4-evaluation--results)
  - [Key Features](#key-features)
  - [References](#references)

---

## Project Structure

```
.
├── code/
│   ├── train.py                # Main script to run DETR fine-tuning
│   ├── dataset.py              # PyTorch Dataset and DataLoader definitions
│   ├── config.py               # Configuration variables (paths, hyperparameters)
│   ├── utils.py                # Utility functions for image/annotation handling
│   ├── model.py                # Model and processor loading functions
│   ├── feature_preprocessor.py # Feature preprocessing class
│   └── ground_truth_labeller.py # Tool for creating bounding box annotations
├── data/
│   ├── matched_annotations/    # Annotation .txt files
│   └── base/cv_data_hw2/data/  # Raw images
├── output/
│   ├── models/                 # Saved fine-tuned models
│   ├── logs/                   # Training logs
│   └── ...                     # Other outputs
├── docs/
│   └── detr_training_pipeline_report.md # Full technical report
├── README.md
```

---

## Setup & Requirements

**Python version:** 3.8+

**Required packages:**

```bash
pip install torch torchvision transformers scikit-learn opencv-python pillow tqdm matplotlib
pip install torchmetrics  # Optional, for advanced evaluation metrics
```

---

## Dataset & Annotation Format

- **Images:** Place raw images in `data/base/cv_data_hw2/data/`.
- **Annotations:** Each image must have a corresponding `.txt` file in `data/matched_annotations/`.
- **Annotation file format:** Each line encodes one object:

  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```

  - All coordinates are normalized to [0, 1] relative to image dimensions.
  - Example:

    ```
    0 0.512 0.433 0.120 0.210
    2 0.700 0.600 0.100 0.150
    ```

---

## Pipeline Usage

### 1. Ground Truth Annotation

If you need to create bounding box labels:

```bash
python code/ground_truth_labeller.py
```

- Draw bounding boxes with your mouse
- Press `s` to save, `n` for next image, `q` to quit
- Annotations are saved as `.txt` files in the format above

### 2. Configure the Pipeline

Edit `code/config.py` to set:

- Data paths
- Model parameters (e.g., number of classes)
- Training hyperparameters (batch size, learning rate, epochs, etc.)
- Output/logging directories

### 3. Training

Run the training script:

```bash
python code/train.py
```

- The pipeline supports staged training (freezing/unfreezing layers, learning rate scheduling, gradient accumulation).
- Training and validation splits are handled automatically.

### 4. Evaluation & Results

- Evaluation metrics (loss, precision, recall, F1) are logged and visualized via TensorBoard.
- Model checkpoints are saved in `output/models/`.
- For detailed analysis and troubleshooting, see the [technical report](docs/detr_training_pipeline_report.md).

---

## Key Features

- **Image Diff Preprocessing:** Enhances detection of moved objects by using pixel-wise differences between "before" and "after" images.
- **Custom Dataset Loader:** Handles annotation parsing, image loading, and preprocessing.
- **Flexible Training:** Supports staged training, gradient accumulation, and learning rate scheduling.
- **Comprehensive Evaluation:** Quantitative (loss, precision, recall, F1) and qualitative (visualizations) analysis.
- **Reproducibility:** Configurable random seeds, clear directory structure, and logging.

---

## References

- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Full Technical Report](docs/detr_training_pipeline_report.md)

---
