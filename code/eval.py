import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, EvalPrediction
# from tqdm.notebook import tqdm  # Use notebook version if in Colab/Jupyter
from tqdm import tqdm  # Use standard version otherwise
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from utils import get_image_paths_from_annotation

def visualize_predictions(model, processor, dataset, device, idx=0, threshold=0.5, save_dir=None):
    """
    Visualize (and optionally save) predictions for the idx-th sample in a dataset,
    and print out predicted labels with scores.
    """
    model.eval()
    item = dataset[idx]
    pixel_values = item["pixel_values"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    # Post-process to get boxes, scores, labels
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[dataset.feature_preprocessor.target_size[::-1]]  # (h, w)
    )
    result = results[0]

    # Print predicted labels and scores
    print(f"\nPredictions for sample {idx}:")
    if len(result["labels"]) == 0:
        print("  No predictions above threshold.")
    else:
        for label_id, score in zip(result["labels"], result["scores"]):
            class_name = model.config.id2label[label_id.item()]
            print(f"  - {class_name}: {score:.2f}")

    # Re-create the diff image for display (should match DETR input size)
    ann_filename = dataset.annotation_files[idx]
    img1_path, img2_path = get_image_paths_from_annotation(ann_filename, dataset.image_dir)
    img1 = dataset.feature_preprocessor.load(img1_path, display=False)
    img2 = dataset.feature_preprocessor.load(img2_path, display=False)
    diff = dataset.feature_preprocessor.compute_difference(img1, img2)
    diff_rgb = diff[..., ::-1]  # BGR -> RGB

    # If diff_rgb shape does not match DETR input size, resize for plotting
    target_h, target_w = dataset.feature_preprocessor.target_size[::-1]
    if diff_rgb.shape[0] != target_h or diff_rgb.shape[1] != target_w:
        import cv2
        diff_rgb = cv2.resize(diff_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(diff_rgb)
    for box, score, label_id in zip(result["boxes"], result["scores"], result["labels"]):
        x_min, y_min, x_max, y_max = box.tolist()
        w = x_max - x_min
        h = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        class_name = model.config.id2label[label_id.item()]
        ax.text(
            x_min, y_min - 5,
            f"{class_name}: {score:.2f}",
            fontsize=12,
            color='white',
            backgroundcolor='red'
        )

    ax.axis('off')

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"prediction_{idx}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved visualization to {out_path}")
    else:
        plt.show()
        
        
def evaluate_detection(model,
                       processor,
                       dataset,
                       collate_fn,
                       device,
                       iou_threshold: float = 0.5,
                       score_threshold: float = 0.5):
    """
    Runs object detection on every image in `dataset` and computes
    total TP, FP, FN at the given thresholds, then returns P, R, F1.
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)                # [1,3,H,W]
            gt = batch["labels"][0]                              # {class_labels: [N], boxes: [N,4]}

            # run DETR
            outputs = model(pixel_values=pv)
            # post-process into absolute corner boxes
            target_size = torch.tensor([[pv.shape[2], pv.shape[3]]], device=device)
            preds = processor.post_process_object_detection(
                outputs, threshold=score_threshold, target_sizes=target_size
            )[0]

            pred_boxes  = preds["boxes"]   # Tensor[K,4] (xmin,ymin,xmax,ymax)
            pred_labels = preds["labels"]  # Tensor[K]
            gt_boxes_n  = gt["boxes"]      # Tensor[N,4] in normalized [cx,cy,w,h]
            gt_labels   = gt["class_labels"]

            # convert GT from [cx,cy,w,h]→[xmin,ymin,xmax,ymax] in absolute coords
            H, W = pv.shape[2], pv.shape[3]
            cx, cy, w, h = gt_boxes_n.unbind(-1)
            gt_boxes = torch.stack([
                (cx - w/2) * W,
                (cy - h/2) * H,
                (cx + w/2) * W,
                (cy + h/2) * H,
            ], dim=-1).to(device)

            # matching preds→GT (one-to-one)
            used_gt = set()
            for pi, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
                # find GT of same class
                mask = (gt_labels.to(device) == pl).nonzero(as_tuple=False).squeeze(1)
                if mask.numel() == 0:
                    total_fp += 1
                    continue

                ious = box_iou(pb.unsqueeze(0), gt_boxes[mask])  # [1, M]
                max_iou, idx = ious.max(1)
                if max_iou >= iou_threshold and mask[idx.item()].item() not in used_gt:
                    total_tp += 1
                    used_gt.add(mask[idx.item()].item())
                else:
                    total_fp += 1

            # any GT boxes left unmatched are false negatives
            total_fn += (gt_boxes.shape[0] - len(used_gt))

    # compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }