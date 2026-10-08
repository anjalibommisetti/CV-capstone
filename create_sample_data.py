"""
create_sample_data.py
Utility to generate synthetic demonstration leaf images and structured sample datasets.

Allows college students and reviewers to run:
1. python train.py (verifying pipeline, baseline CNN, MobileNetV2, metrics, plots)
2. python app.py (test image upload, segmentation, disease spot highlighting, recommendations)
instantly on a laptop without waiting hours for the 2GB PlantVillage download!
"""

import os
import cv2
import numpy as np

SAMPLE_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Apple___Apple_scab",
    "Apple___healthy"
]


def generate_leaf_image(class_name: str, width: int = 256, height: int = 256, variant: int = 0) -> np.ndarray:
    """
    Generates a realistic synthetic leaf image on a neutral background
    with distinct healthy vs diseased visual characteristics.
    """
    # Background: off-white or soft studio gray with slight texture
    bg_color = (228 + (variant * 3) % 15, 230 + (variant * 2) % 12, 232)
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    center_x, center_y = width // 2, height // 2
    
    # Base green leaf blade
    is_healthy = "healthy" in class_name.lower()
    
    if "Tomato" in class_name:
        # Serrated compound leaf lobe shape
        pts = np.array([
            [center_x, center_y - 100],
            [center_x + 35, center_y - 70],
            [center_x + 75, center_y - 30],
            [center_x + 85, center_y + 20],
            [center_x + 60, center_y + 70],
            [center_x + 20, center_y + 95],
            [center_x, center_y + 110],
            [center_x - 20, center_y + 95],
            [center_x - 60, center_y + 70],
            [center_x - 85, center_y + 20],
            [center_x - 75, center_y - 30],
            [center_x - 35, center_y - 70]
        ], np.int32)
        leaf_color = (34, 139, 34) if is_healthy else (46, 125, 50)
    else:  # Apple / oval leaf
        pts = np.array([
            [center_x, center_y - 95],
            [center_x + 55, center_y - 50],
            [center_x + 75, center_y + 10],
            [center_x + 50, center_y + 70],
            [center_x, center_y + 105],
            [center_x - 50, center_y + 70],
            [center_x - 75, center_y + 10],
            [center_x - 55, center_y - 50]
        ], np.int32)
        leaf_color = (40, 160, 45) if is_healthy else (50, 130, 40)
        
    cv2.fillPoly(img, [pts], leaf_color)
    
    # Draw central vein and lateral veins
    vein_color = (70, 180, 70) if is_healthy else (80, 150, 60)
    cv2.line(img, (center_x, center_y - 90), (center_x, center_y + 100), vein_color, 2)
    for offset_y in [-50, -20, 15, 50]:
        cv2.line(img, (center_x, center_y + offset_y), (center_x + 45, center_y + offset_y - 15), vein_color, 1)
        cv2.line(img, (center_x, center_y + offset_y), (center_x - 45, center_y + offset_y - 15), vein_color, 1)
        
    # If diseased, draw characteristic pathological lesion spots
    if not is_healthy:
        if "Early_blight" in class_name:
            # Concentric target-board brown rings with yellow halo
            spots = [
                (center_x - 25, center_y - 20, 18),
                (center_x + 30, center_y + 25, 22),
                (center_x - 10, center_y + 45, 14)
            ]
            for sx, sy, r in spots:
                # Chlorotic yellow outer halo
                cv2.circle(img, (sx, sy), r + 7, (60, 195, 215), -1)
                # Outer brown lesion ring
                cv2.circle(img, (sx, sy), r, (19, 69, 139), -1)
                # Middle concentric ring
                cv2.circle(img, (sx, sy), int(r * 0.65), (28, 85, 160), -1)
                # Inner dark necrotic center
                cv2.circle(img, (sx, sy), int(r * 0.35), (10, 40, 90), -1)
        elif "scab" in class_name.lower():
            # Velvety dark olive-black irregular spots
            scab_spots = [
                (center_x - 30, center_y - 15, 12),
                (center_x + 20, center_y - 30, 15),
                (center_x + 15, center_y + 35, 10),
                (center_x - 20, center_y + 30, 14)
            ]
            for sx, sy, r in scab_spots:
                cv2.circle(img, (sx, sy), r, (15, 45, 50), -1)
                cv2.circle(img, (sx + 2, sy - 1), max(r - 4, 2), (20, 30, 35), -1)
                
    # Add slight natural Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def create_demo_dataset(base_dir: str = "dataset", samples_per_class: int = 15):
    """
    Creates an 80/10/10 split sample dataset for quick testing and verification.
    """
    splits = {
        "train": int(samples_per_class * 0.8),
        "val": max(1, int(samples_per_class * 0.1)),
        "test": max(1, int(samples_per_class * 0.1))
    }
    
    print(f"\n[INFO] Generating sample dataset in '{base_dir}'...")
    for class_name in SAMPLE_CLASSES:
        for split_name, count in splits.items():
            folder = os.path.join(base_dir, split_name, class_name)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                img = generate_leaf_image(class_name, variant=i)
                filepath = os.path.join(folder, f"{class_name}_{i:03d}.jpg")
                # Save in BGR for OpenCV imwrite
                cv2.imwrite(filepath, img)
        print(f"  Created sample images for: {class_name}")
        
    # Also save dedicated test sample images for web UI upload testing
    samples_dir = os.path.join("static", "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    tomato_eb = generate_leaf_image("Tomato___Early_blight", variant=99)
    cv2.imwrite(os.path.join(samples_dir, "tomato_early_blight.jpg"), tomato_eb)
    
    tomato_h = generate_leaf_image("Tomato___healthy", variant=88)
    cv2.imwrite(os.path.join(samples_dir, "tomato_healthy.jpg"), tomato_h)
    
    apple_s = generate_leaf_image("Apple___Apple_scab", variant=77)
    cv2.imwrite(os.path.join(samples_dir, "apple_scab.jpg"), apple_s)
    
    print(f"[SUCCESS] Sample dataset & test images generated in '{base_dir}' and '{samples_dir}'!\n")


if __name__ == "__main__":
    create_demo_dataset()
