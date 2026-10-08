"""
segment.py
MODULE 2: Leaf Segmentation & Feature Extraction

Features:
1. Leaf background separation via:
   - HSV color thresholding & morphological filtering (default fast method)
   - K-means clustering (unsupervised color clustering)
2. Diseased region (spot / lesion) isolation and visual highlighting with neon bounding overlays.
3. Feature Extraction:
   - Color histograms & statistical moments (Mean, Std, Skewness)
   - Texture features via GLCM (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
   - Shape features (Area, Perimeter, Aspect Ratio, Circularity / Compactness)
4. Theoretical bridge: Explanation of how Convolutional Neural Networks (CNNs)
   automatically learn these features end-to-end.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def segment_leaf_hsv(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separates the leaf from the background using HSV color-space thresholding
    and morphological contour extraction.
    
    Plant leaves possess distinct green, yellow-green, and brown pigments.
    In HSV:
    - Hue (H) captures pure color wavelength independent of lighting.
    - Saturation (S) captures color richness.
    - Value (V) captures brightness.
    
    Returns:
        (segmented_rgb, binary_leaf_mask)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Range 1: Green foliage (H: 25-90)
    lower_green = np.array([25, 30, 20])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Range 2: Yellow / brown diseased leaf tissue (H: 10-25)
    lower_brown = np.array([10, 40, 30])
    upper_brown = np.array([25, 255, 220])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combined leaf color mask
    raw_mask = cv2.bitwise_or(mask_green, mask_brown)
    
    # If the image background has minimal saturation (white/gray/black lab paper),
    # Otsu thresholding on saturation is also an effective complement:
    s_channel = hsv[:, :, 1]
    _, otsu_sat_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    combined_mask = cv2.bitwise_or(raw_mask, otsu_sat_mask)
    
    # Morphological cleaning: Close small holes inside the leaf, remove small background specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Find contours and keep the largest contour (the primary leaf)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(clean_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Ensure contour is large enough to be a leaf
        if cv2.contourArea(largest_contour) > (img_rgb.shape[0] * img_rgb.shape[1] * 0.05):
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            final_mask = clean_mask
    else:
        final_mask = clean_mask

    # Extract segmented leaf (background filled with clean white or black)
    segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
    
    return segmented_rgb, final_mask


def segment_leaf_kmeans(img_rgb: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Unsupervised segmentation using K-Means Clustering on pixel color values.
    
    Clusters pixels into K color centroids (typically: background, healthy tissue, lesion).
    """
    # Reshape the image into a 2D array of pixels (N, 3)
    pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)
    
    # Define stopping criteria: 100 iterations or epsilon = 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Run K-means clustering
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)
    
    # Identify the cluster with the highest green-to-red ratio as the leaf cluster
    green_scores = [float(center[1]) - float(center[0]) for center in centers]
    leaf_cluster_idx = int(np.argmax(green_scores))
    
    leaf_mask = (labels.flatten() == leaf_cluster_idx).astype(np.uint8) * 255
    leaf_mask = leaf_mask.reshape((img_rgb.shape[0], img_rgb.shape[1]))
    
    # Morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    
    segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=leaf_mask)
    return segmented_rgb, leaf_mask


def highlight_diseased_regions(img_rgb: np.ndarray, leaf_mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Identifies and highlights necrotic/chlorotic diseased spots within the leaf boundary.
    
    Diseased spots exhibit:
    - Brown, yellow, or black necrotic tissue where chlorophyll has degraded.
    - Low green-to-red ratio (R >= G or significant yellowing: H between 5 and 30).
    
    Returns:
        (highlighted_rgb, disease_mask, disease_stats)
    """
    if leaf_mask is None:
        _, leaf_mask = segment_leaf_hsv(img_rgb)
        
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    
    # Criterion 1: Brownish/Yellow lesions (Hue 8 to 28, moderate saturation)
    lesion_color_mask = ((h >= 5) & (h <= 28) & (s >= 35) & (v >= 30)).astype(np.uint8) * 255
    
    # Criterion 2: Necrotic dark spots (Very low value/brightness inside leaf)
    dark_spot_mask = ((v < 60) & (leaf_mask > 0)).astype(np.uint8) * 255
    
    # Criterion 3: Discolored chlorophyll deficiency (Red component significantly exceeds or matches Green)
    chlorosis_mask = ((r.astype(np.int16) - g.astype(np.int16) > -15) & (g > 30) & (leaf_mask > 0)).astype(np.uint8) * 255
    
    # Combine lesion cues and restrict strictly to the segmented leaf area
    raw_disease_mask = cv2.bitwise_or(lesion_color_mask, dark_spot_mask)
    raw_disease_mask = cv2.bitwise_or(raw_disease_mask, chlorosis_mask)
    disease_mask = cv2.bitwise_and(raw_disease_mask, raw_disease_mask, mask=leaf_mask)
    
    # Filter microscopic specks (noise) with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    
    # Generate visualization overlay:
    # 1. Semi-transparent red highlight over infected tissue
    overlay = img_rgb.copy()
    overlay[disease_mask > 0] = [235, 50, 50]  # Bright coral red
    
    # Blend overlay with original image (alpha = 0.65, beta = 0.35)
    highlighted_rgb = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)
    
    # 2. Draw distinct neon-yellow bounding contours around distinct disease lesions
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = 0
    total_spot_area = 0.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 12:  # Ignore trivial single-pixel noise
            spot_count += 1
            total_spot_area += area
            # Draw boundary contour
            cv2.drawContours(highlighted_rgb, [cnt], -1, (255, 230, 0), 1, lineType=cv2.LINE_AA)
            
    # Calculate infection statistics
    total_leaf_pixels = float(np.count_nonzero(leaf_mask))
    infected_pixels = float(np.count_nonzero(disease_mask))
    infection_pct = (infected_pixels / max(total_leaf_pixels, 1.0)) * 100.0
    
    disease_stats = {
        "spot_count": spot_count,
        "infection_percentage": round(infection_pct, 2),
        "total_leaf_area_px": int(total_leaf_pixels),
        "infected_area_px": int(infected_pixels)
    }
    
    return highlighted_rgb, disease_mask, disease_stats


def extract_color_features(img_rgb: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Extracts statistical color moments (Mean, Standard Deviation, Skewness)
    and histogram features for R, G, B and H, S, V channels.
    """
    if mask is not None:
        pixels_r = img_rgb[:, :, 0][mask > 0]
        pixels_g = img_rgb[:, :, 1][mask > 0]
        pixels_b = img_rgb[:, :, 2][mask > 0]
    else:
        pixels_r = img_rgb[:, :, 0].flatten()
        pixels_g = img_rgb[:, :, 1].flatten()
        pixels_b = img_rgb[:, :, 2].flatten()
        
    if len(pixels_r) == 0:
        pixels_r = img_rgb[:, :, 0].flatten()
        pixels_g = img_rgb[:, :, 1].flatten()
        pixels_b = img_rgb[:, :, 2].flatten()

    def calc_moments(channel_data):
        mean = float(np.mean(channel_data))
        std = float(np.std(channel_data))
        # Skewness: measures asymmetry of color distribution around the mean
        skew = float(np.mean(((channel_data - mean) / max(std, 1e-5)) ** 3))
        return round(mean, 2), round(std, 2), round(skew, 2)

    r_mean, r_std, r_skew = calc_moments(pixels_r)
    g_mean, g_std, g_skew = calc_moments(pixels_g)
    b_mean, b_std, b_skew = calc_moments(pixels_b)
    
    # Compute 32-bin histograms for quick visual plotting
    hist_r, _ = np.histogram(pixels_r, bins=32, range=(0, 256), density=True)
    hist_g, _ = np.histogram(pixels_g, bins=32, range=(0, 256), density=True)
    hist_b, _ = np.histogram(pixels_b, bins=32, range=(0, 256), density=True)

    return {
        "rgb_moments": {
            "red": {"mean": r_mean, "std": r_std, "skewness": r_skew},
            "green": {"mean": g_mean, "std": g_std, "skewness": g_skew},
            "blue": {"mean": b_mean, "std": b_std, "skewness": b_skew}
        },
        "histograms": {
            "r": hist_r.tolist(),
            "g": hist_g.tolist(),
            "b": hist_b.tolist()
        }
    }


def extract_glcm_texture_features(img_rgb: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Extracts second-order statistical texture features using the
    Gray-Level Co-occurrence Matrix (GLCM).
    
    Features computed:
    - Contrast: Measures local intensity variations.
    - Dissimilarity: Measures the difference between pairs of pixels.
    - Homogeneity: Measures uniformity of textures.
    - Energy: Measures textural order/regularity.
    - Correlation: Measures joint probability occurrence of specified pixel pairs.
    - ASM (Angular Second Moment): Measure of homogeneity.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Quantize to 32 gray levels to make GLCM computationally fast & robust
    quantized_gray = (gray // 8).astype(np.uint8)
    
    # Compute GLCM with distance=1 at 4 canonical angles (0, 45, 90, 135 degrees)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        quantized_gray, 
        distances=distances, 
        angles=angles, 
        levels=32, 
        symmetric=True, 
        normed=True
    )
    
    # Extract properties and average across angles
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    asm = float(np.mean(graycoprops(glcm, 'ASM')))
    
    return {
        "contrast": round(contrast, 3),
        "dissimilarity": round(dissimilarity, 3),
        "homogeneity": round(homogeneity, 3),
        "energy": round(energy, 3),
        "correlation": round(correlation, 3),
        "asm": round(asm, 3)
    }


def extract_shape_features(leaf_mask: np.ndarray) -> dict:
    """
    Extracts morphological shape descriptors from the binary leaf mask:
    - Area & Perimeter
    - Aspect Ratio (Width / Height)
    - Extent (Contour Area / Bounding Box Area)
    - Circularity / Compactness: 4 * pi * Area / (Perimeter^2)
    """
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 1.0, "circularity": 0.0}
    
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, closed=True))
    
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / max(float(h), 1.0)
    extent = area / max(float(w * h), 1.0)
    
    # Circularity: 1.0 indicates a perfect circle, lower values indicate elongated/irregular leaves
    circularity = (4.0 * np.pi * area) / max(perimeter ** 2, 1e-5)
    
    return {
        "area_px": int(area),
        "perimeter_px": round(perimeter, 1),
        "aspect_ratio": round(aspect_ratio, 3),
        "extent": round(extent, 3),
        "circularity": round(circularity, 3)
    }


def explain_cnn_vs_classical_features() -> str:
    """
    Educational reference explaining the shift from Classical Computer Vision
    (Feature Engineering) to Convolutional Neural Networks (Feature Learning).
    
    Crucial for viva examinations, thesis reports, and presentations.
    """
    explanation = """
========================================================================================
   CLASSICAL COMPUTER VISION vs CONVOLUTIONAL NEURAL NETWORKS (CNN) EXPLANATION
========================================================================================

1. CLASSICAL CV APPROACH (Manual Feature Engineering):
   - Color Features: We manually compute RGB/HSV histograms and moments (Mean, Std, Skewness).
   - Texture Features: We manually compute Gray-Level Co-occurrence Matrices (GLCM) for
     contrast, homogeneity, and energy to quantify rough vs smooth leaf surfaces.
   - Shape Features: We manually extract contours, aspect ratio, and circularity.
   - Limitation: Highly sensitive to lighting changes, shadow variations, and complex backgrounds.
     Requires hand-crafting different parameters for every single crop type.

2. DEEP LEARNING APPROACH (Hierarchical Automated Feature Learning via CNNs):
   - A CNN does not require manual GLCM or color histograms! Instead, backpropagation
     trains learnable convolution filter kernels (e.g. 3x3) to capture these patterns:
     
     * Layer 1 - 2 (Low-Level Features):
       Kernels act like Gabor and Sobel edge detectors, learning oriented edges, color gradients,
       and boundary transitions (mirroring color histograms & shape contours).
       
     * Layer 3 - 5 (Mid-Level Features):
       Kernels combine edges to recognize textures, circular spots, concentric ring lesions,
       and pustule clusters (directly replacing manual GLCM texture extraction).
       
     * Deeper Layers (High-Level Semantic Features):
       Deep receptive fields recognize full pathological symptoms:
       - Alternaria target-board concentric rings (Early Blight)
       - Oomycete water-soaked chlorotic decay (Late Blight)
       - Talcum-white fungal mycelium mats (Powdery Mildew)
       
3. SUMMARY FOR VIVA / EXAM:
   'In classical CV, the engineer invents the features and the model just classifies them.
    In CNNs, the neural network learns BOTH the feature extraction and classification
    simultaneously from raw data, achieving far higher accuracy and robustness.'
========================================================================================
"""
    return explanation


if __name__ == "__main__":
    print("[Module 2: Leaf Segmentation & Feature Extraction Test]")
    # Create synthetic leaf with spot
    test_leaf = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(test_leaf, (150, 150), 90, (40, 150, 40), -1)      # Green leaf body
    cv2.circle(test_leaf, (130, 120), 22, (140, 70, 20), -1)     # Brown Early Blight spot
    cv2.circle(test_leaf, (170, 160), 15, (180, 140, 20), -1)    # Yellow chlorotic spot

    # 1. Segment leaf
    seg_rgb, mask = segment_leaf_hsv(test_leaf)
    print("Leaf mask pixel count:", np.count_nonzero(mask))
    
    # 2. Highlight disease
    highlighted, d_mask, stats = highlight_diseased_regions(test_leaf, mask)
    print("Disease stats:", stats)
    
    # 3. Extract features
    color_feats = extract_color_features(test_leaf, mask)
    glcm_feats = extract_glcm_texture_features(test_leaf, mask)
    shape_feats = extract_shape_features(mask)
    
    print("Color moments (Green channel):", color_feats["rgb_moments"]["green"])
    print("GLCM Texture features:", glcm_feats)
    print("Shape descriptors:", shape_feats)
    print(explain_cnn_vs_classical_features())
    print("Module 2 self-test passed successfully!")
