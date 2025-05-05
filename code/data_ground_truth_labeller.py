import os
import numpy as np
import cv2
import random
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances

# Load ResNet model for feature extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def parse_annotation_file(annotation_path):
    objects = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            object_id = int(parts[0])  # Object ID
            duration = int(parts[1])   # Duration
            frame_id = int(parts[2])   # Frame number
            x, y, width, height = map(float, parts[3:7])  # Bounding box coordinates
            object_type = int(parts[7])  # Object type

            obj = {
                'id': object_id,
                'duration': duration,
                'frame': frame_id,
                'bbox': [x, y, width, height],
                'type': object_type,
                'bbox': [x, y, width, height],
                'centroid': [x + width / 2, y + height / 2]
            }
            objects.append(obj)
    return objects

def extract_features(img, objects):
    features = []
    for obj in objects:
        x, y, w, h = map(int, obj['bbox'])
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            features.append(np.zeros(512))
            continue
        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(inp).cpu().squeeze().numpy()
        features.append(feat)
    return np.array(features)

def compute_cost_matrix_centroid(objects1, objects2):
    n1, n2 = len(objects1), len(objects2)
    cost_matrix = np.zeros((n1, n2))
    max_distance = max(
        np.linalg.norm(np.array(o1['centroid']) - np.array(o2['centroid']))
        for o1 in objects1 for o2 in objects2
    ) if objects1 and objects2 else 1

    for i, o1 in enumerate(objects1):
        for j, o2 in enumerate(objects2):
            d = np.linalg.norm(np.array(o1['centroid']) - np.array(o2['centroid'])) / max_distance
            type_penalty = 0 if o1['type'] == o2['type'] else 1
            area1 = o1['bbox'][2] * o1['bbox'][3]
            area2 = o2['bbox'][2] * o2['bbox'][3]
            size_penalty = 1 - (min(area1, area2) / max(area1, area2)) if max(area1, area2) > 0 else 1
            cost_matrix[i, j] = 0.5 * d + 0.3 * type_penalty + 0.2 * size_penalty
    return cost_matrix
    
def compute_cost_matrix_feat(img1, img2, objects1, objects2):
    feats1 = extract_features(img1, objects1)
    feats2 = extract_features(img2, objects2)
    return cosine_distances(feats1, feats2) + compute_cost_matrix_centroid(objects1, objects2)

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def draw_matches(img, objects1, objects2, matched_pairs):
    img_draw = img.copy()
    for i, j in matched_pairs:
        color = tuple(random.randint(0, 255) for _ in range(3))

        # Draw old (previous frame) bbox
        x1, y1, w1, h1 = map(int, objects1[i]['bbox'])
        cv2.rectangle(img_draw, (x1, y1), (x1 + w1, y1 + h1), color, 2)

        # Draw new (current frame) bbox
        x2, y2, w2, h2 = map(int, objects2[j]['bbox'])
        cv2.rectangle(img_draw, (x2, y2), (x2 + w2, y2 + h2), color, 2)

    return img_draw

def process_pair(img1_path, ann1_path, img2_path, ann2_path, folder_name, output_ann_dir, visual_dir=None, visualize=False, iou_threshold=0.1):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    objects1 = parse_annotation_file(ann1_path)
    objects2 = parse_annotation_file(ann2_path)

    if not objects1 or not objects2:
        return

    cost_matrix = compute_cost_matrix_feat(img1, img2, objects1, objects2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_objects = []
    for i, j in zip(row_ind, col_ind):
        iou = compute_iou(objects1[i]['bbox'], objects2[j]['bbox'])
        if iou < iou_threshold:
            matched_objects.append((objects1[i], objects2[j]))

    if matched_objects:
        filename = f"{folder_name}-{os.path.splitext(os.path.basename(img1_path))[0]}-{os.path.splitext(os.path.basename(img2_path))[0]}_match.txt"
        out_path = os.path.join(output_ann_dir, filename)
        with open(out_path, 'w') as f:
            for match_id, (obj1, obj2) in enumerate(matched_objects):
                for obj in [obj1, obj2]:
                    x, y, w, h = map(int, obj['bbox'])
                    duration = 1  # Placeholder; replace with actual if available
                    line = f"{match_id} {x} {y} {w} {h} {obj['type']}\n"
                    f.write(line)

        if visualize and visual_dir:
            idx_pairs = [(objects1.index(obj1), objects2.index(obj2)) for obj1, obj2 in matched_objects if obj1 in objects1 and obj2 in objects2]
            if idx_pairs:
                img = draw_matches(img2, objects1, objects2, idx_pairs)
                vis_path = os.path.join(visual_dir, filename.replace(".txt", ".png"))
                cv2.imwrite(vis_path, img)


def run_pipeline(data_dir, output_ann_dir, visual_dir, batch_size=30, visualize_samples=5):
    os.makedirs(output_ann_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    index_file = os.path.join(data_dir, 'index.txt')
    with open(index_file, 'r') as f:
        lines = [line.strip().split(',') for line in f if len(line.strip().split(',')) == 4]

    total_batches = (len(lines) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(lines))
        batch = lines[start:end]
        visual_batch = random.sample(batch, min(visualize_samples, len(batch)))

        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} [{start}:{end}]")

        for pair in batch:
            img1_path = os.path.join(data_dir, pair[0])
            ann1_path = os.path.join(data_dir, pair[1])
            img2_path = os.path.join(data_dir, pair[2])
            ann2_path = os.path.join(data_dir, pair[3])
            visualize = pair in visual_batch
            folder_name = os.path.basename(os.path.dirname(img1_path))
            # print(f"Processing pair in folder '{folder_name}': {img1_path} and {img2_path}")
            process_pair(img1_path, ann1_path, img2_path, ann2_path, folder_name, output_ann_dir, visual_dir, visualize)

    print(f"\nProcessed {len(lines)} image pairs in {total_batches} batches.")

if __name__ == "__main__":
    base_dir = '../data/base/cv_data_hw2'
    output_ann_dir = '../data/matched_annotations'
    visual_dir = '../data/visual_matches'
    run_pipeline(base_dir, output_ann_dir, visual_dir)
    print("Pipeline completed.")