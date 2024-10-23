import cv2
import numpy as np
import torch

def extract_box(image_np, box):
    """Extract a bounding box from an image."""
    x1, y1, x2, y2 = [int(coord) for coord in box]
    return image_np[y1:y2, x1:x2]

def extract_features_for_svm(image_np):
    """Extract features for SVM classification."""
    image_resized = cv2.resize(image_np, (64, 64))
    return image_resized.flatten()

def extract_global_features(image_np):
    """Extract global features from the image."""
    global_features = cv2.calcHist([image_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(global_features, global_features).flatten()

def compute_relationships(detections):
    """Compute relationships between detected objects."""
    relationships = []
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i != j:
                relationship = {
                    'subject': detections[i]['class'],
                    'object': detections[j]['class'],
                    'relationship': 'interacts_with'
                }
                relationships.append(relationship)
    return relationships

def preprocess_image_for_faster_rcnn(image):
    # Resize and normalize image for Faster R-CNN
    image_resized = cv2.resize(image, (800, 800))  # Example size, adjust as needed
    image_normalized = image_resized / 255.0  # Normalize to [0, 1]
    return torch.tensor(image_normalized).permute(2, 0, 1)  # Change to CxHxW format