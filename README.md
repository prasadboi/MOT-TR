# DETR Fine-tuning Pipeline for Moved Object Detection

This repository provides a complete pipeline for fine-tuning [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) on a custom dataset for moved object detection. The pipeline includes dataset preparation, feature preprocessing, model training, and evaluation.

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

## Usage

### 1. Ground Truth Annotation (if needed)

Run the annotation tool to create bounding box labels:
```bash
python code/ground_truth_labeller.py --image_dir /path/to/images --output_dir /path/to/save/annotations
```
- Draw bounding boxes with your mouse
- Press `s` to save, `n` for next image, `q` to quit
- Annotations are saved as `.txt` files with format: `x y w h 1` (coordinates and class label)

### 2. Configure the Pipeline

Edit `code/config.py` to set:
- Data paths
- Model parameters
- Training hyperparameters (batch size, learning rate, etc.)

### 3. Train the Model

```bash
python code/train.py
```

The script will:
- Load and preprocess the dataset
- Initialize the DETR model
- Train and evaluate the model
- Save checkpoints to `output/models/`

### 4. Evaluate & Visualize Results

```bash
python code/train.py --eval_only --model_path output/models/your_model.pth
```

This will:
- Load your trained model
- Run evaluation on the test set
- Calculate mAP metrics (if torchmetrics is installed)
- Display qualitative results with bounding box visualizations

---

## Advanced Usage

### Feature Preprocessing

The pipeline includes a `FeaturePreprocessing` class that:
- Performs image differencing between pairs
- Handles resizing and normalization
- Prepares features for the DETR model

### Custom Dataset

The `MovedObjectDataset` class:
- Loads image pairs and annotations
- Applies feature preprocessing
- Converts annotations to DETR format

### Training Configuration

Key parameters in `config.py`:
- `BATCH_SIZE`: Adjust based on your GPU memory
- `NUM_EPOCHS`: Training duration
- `LEARNING_RATE`: Default is 5e-5
- `WEIGHT_DECAY`: Default is 1e-4

---

## Troubleshooting

- **No annotation files found:**  
  Ensure your annotation `.txt` files are in the correct directory and match the expected format.

- **CUDA out of memory:**  
  Lower the batch size in `config.py`.

- **OpenCV window not showing:**  
  If running in a remote environment, OpenCV GUI windows may not work. Run the labeller locally.

---

## Acknowledgements

- [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/detr)
- [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)

---

**For questions or issues, please open an issue or contact the maintainer.**
