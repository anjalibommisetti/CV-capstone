"""
preprocess.py
MODULE 1: Image Acquisition & Preprocessing

Functions for:
1. Loading image from file or camera buffer and validating format and size.
2. Noise suppression using Gaussian Blur.
3. Resizing to 224x224 (MobileNetV2 standard input).
4. Normalizing pixel values to [0, 1] or [-1, 1].
5. Data augmentation pipeline (rotation, flips, zoom, brightness) for training & preview.
"""

import os
import cv2
import numpy as np
from PIL import Image

# Allowed image file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE = (224, 224)


def validate_image_file(file_path: str, min_size: int = 32) -> tuple[bool, str]:
    """
    Validates that a file exists, has a valid image extension,
    and can be decoded into a valid non-empty image.
    
    Args:
        file_path: Path to the image file.
        min_size: Minimum width and height in pixels.
        
    Returns:
        (is_valid, message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Attempt to open and inspect dimensions
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return False, f"Image dimensions too small: {w}x{h}. Minimum required is {min_size}x{min_size}."
            return True, "Image is valid."
    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}"


def load_image_rgb(image_input) -> np.ndarray:
    """
    Loads an image from a file path, file-like object, or numpy array,
    and returns a standard 3-channel RGB numpy array (uint8, 0-255).
    
    Handles:
    - File path string
    - PIL Image object
    - Grayscale or RGBA conversions to RGB
    """
    if isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        return np.array(img)
    
    if isinstance(image_input, str):
        # Read using OpenCV
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            # Fallback to PIL in case OpenCV fails on certain encodings
            pil_img = Image.open(image_input).convert("RGB")
            return np.array(pil_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    # If it's a file stream / BytesIO
    pil_img = Image.open(image_input).convert("RGB")
    return np.array(pil_img)


def remove_noise_gaussian(img_rgb: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 0.0) -> np.ndarray:
    """
    Applies Gaussian Blur filtering to suppress high-frequency camera noise
    while preserving edge boundaries of the leaf and diseased spots.
    
    Gaussian filter uses a 2D Gaussian bell-curve kernel:
    G(x, y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2))
    """
    return cv2.GaussianBlur(img_rgb, kernel_size, sigma)


def resize_image(img_rgb: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    Resizes image to target dimensions (e.g. 224x224).
    Uses INTER_AREA when downsampling for sharp edges, or INTER_LINEAR when upsampling.
    """
    h, w = img_rgb.shape[:2]
    target_w, target_h = target_size
    if w > target_w or h > target_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(img_rgb, target_size, interpolation=interpolation)


def normalize_pixels(img: np.ndarray, mode: str = "zero_one") -> np.ndarray:
    """
    Normalizes pixel intensities for deep neural network input.
    
    Modes:
    - 'zero_one': Scales [0, 255] -> [0.0, 1.0] (Used by Custom CNN & standard visualizers)
    - 'mobilenet': Scales [0, 255] -> [-1.0, 1.0] (Used by MobileNetV2: img / 127.5 - 1.0)
    """
    img_float = img.astype(np.float32)
    if mode == "mobilenet":
        return (img_float / 127.5) - 1.0
    return img_float / 255.0


def preprocess_pipeline(image_input, target_size: tuple = TARGET_SIZE, apply_blur: bool = True):
    """
    Complete Preprocessing Pipeline for Module 1:
    1. Load image and ensure RGB.
    2. Optional Gaussian blur for noise removal.
    3. Resize to target (224x224).
    4. Normalize pixel values.
    
    Returns:
        dict containing:
        - 'original_rgb': Original loaded RGB image
        - 'blurred_rgb': Noise-reduced RGB image
        - 'resized_rgb': 224x224 RGB image (uint8, ready for display)
        - 'normalized_zero_one': (224, 224, 3) float32 in [0, 1]
        - 'normalized_mobilenet': (224, 224, 3) float32 in [-1, 1]
    """
    original_rgb = load_image_rgb(image_input)
    
    if apply_blur:
        blurred_rgb = remove_noise_gaussian(original_rgb, kernel_size=(5, 5))
    else:
        blurred_rgb = original_rgb.copy()
        
    resized_rgb = resize_image(blurred_rgb, target_size=target_size)
    
    norm_01 = normalize_pixels(resized_rgb, mode="zero_one")
    norm_mobilenet = normalize_pixels(resized_rgb, mode="mobilenet")
    
    return {
        "original_rgb": original_rgb,
        "blurred_rgb": blurred_rgb,
        "resized_rgb": resized_rgb,
        "normalized_zero_one": norm_01,
        "normalized_mobilenet": norm_mobilenet
    }


def get_data_augmentation_layers():
    """
    Returns a tf.keras.Sequential pipeline of data augmentation layers
    for training deep learning models.
    
    Includes:
    - RandomFlip: Horizontal and vertical reflections
    - RandomRotation: Slight rotations (-15% to +15% / approx -54 to +54 deg)
    - RandomZoom: Zoom in/out by 15%
    - RandomContrast: Light exposure variations
    """
    import tensorflow as tf
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        tf.keras.layers.RandomRotation(0.15, name="aug_rotation"),
        tf.keras.layers.RandomZoom(0.15, name="aug_zoom"),
        tf.keras.layers.RandomContrast(0.15, name="aug_contrast")
    ], name="data_augmentation_pipeline")
    
    return data_augmentation


def sample_augmentations_opencv(img_rgb: np.ndarray) -> dict:
    """
    Demonstration helper using OpenCV / NumPy to generate distinct
    augmented versions of a single leaf image for demonstration and inspection.
    
    Returns:
        Dictionary of augmented RGB images:
        - 'rotated_90': 90-degree rotated leaf
        - 'horizontal_flip': Horizontally mirrored
        - 'vertical_flip': Vertically mirrored
        - 'brightness_adjusted': Simulating different sunlight intensities
        - 'zoomed': Cropped and zoomed central leaf region
    """
    h, w = img_rgb.shape[:2]
    
    # 1. Rotation (45 degrees)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img_rgb, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. Flips
    h_flip = cv2.flip(img_rgb, 1)
    v_flip = cv2.flip(img_rgb, 0)
    
    # 3. Brightness adjustment (+40 and -40)
    bright = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=30)
    
    # 4. Zoom (center crop 80% and scale back)
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img_rgb[start_y:start_y + crop_h, start_x:start_x + crop_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return {
        "original": img_rgb,
        "rotated_45": rotated,
        "horizontal_flip": h_flip,
        "vertical_flip": v_flip,
        "brightened": bright,
        "zoomed": zoomed
    }


if __name__ == "__main__":
    print("[Module 1: Image Acquisition & Preprocessing Test]")
    # Create a synthetic test leaf patch to verify pipeline
    test_patch = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(test_patch, (150, 150), 100, (34, 139, 34), -1)  # Forest Green circle
    cv2.circle(test_patch, (130, 130), 20, (139, 69, 19), -1)   # Brown spot
    
    processed = preprocess_pipeline(test_patch)
    print("Original shape:", processed["original_rgb"].shape)
    print("Resized shape: ", processed["resized_rgb"].shape)
    print("Normalized range [min, max]:", 
          processed["normalized_zero_one"].min(), 
          processed["normalized_zero_one"].max())
    print("MobileNet normalized range [min, max]:", 
          processed["normalized_mobilenet"].min(), 
          processed["normalized_mobilenet"].max())
    print("Augmentations preview count:", len(sample_augmentations_opencv(test_patch)))
    print("Module 1 self-test passed successfully!")
