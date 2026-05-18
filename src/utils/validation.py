"""
SPC-GAN – Preprocessing Validation Pipeline
Provides robust checks to verify that uploaded images are valid, uncorrupted,
in-focus, and contain plant leaves before running disease classification.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image

_IMAGENET_MODEL = None
_IMAGENET_WEIGHTS = None

# Comprehensive plant, leaf, and agricultural keywords
PLANT_KEYWORDS = {
    "leaf", "foliage", "plant", "tree", "flower", "grass", "herb", "shrub", "fern", "moss", 
    "vegetable", "fruit", "crop", "greenhouse", "daisy", "buckeye", "acorn", "artichoke", 
    "cardoon", "cabbage", "broccoli", "cauliflower", "zucchini", "squash", "cucumber", 
    "banana", "corn", "pot", "spider web", "parasol", "head cabbage", "fig", "pineapple", 
    "custard apple", "pomegranate", "jackfruit", "orange", "lemon", "strawberry", "grape",
    "apple", "pear", "peach", "plum", "cherry", "nectarine", "apricot", "bell pepper",
    "eggplant", "potato", "onion", "garlic", "tomato", "chili", "pepper"
}

def get_imagenet_model():
    """Lazily loads a lightweight pretrained MobileNet-V3-Small for ImageNet classification."""
    global _IMAGENET_MODEL, _IMAGENET_WEIGHTS
    if _IMAGENET_MODEL is None:
        import torchvision.models as models
        # Lightweight weights (~10MB)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        _IMAGENET_MODEL = model
        _IMAGENET_WEIGHTS = weights
    return _IMAGENET_MODEL, _IMAGENET_WEIGHTS

def check_corruption_and_empty(uploaded_file):
    """
    Checks if the uploaded file is empty or corrupted.
    Returns:
        (True, pil_image) if valid.
        (False, error_msg) if empty or corrupted.
    """
    if uploaded_file is None:
        return False, "No file uploaded. Please upload a tomato leaf image."
    
    # Check if empty (handles Streamlit UploadedFile, BytesIO, and generic streams)
    is_empty = False
    if hasattr(uploaded_file, 'size'):
        if uploaded_file.size == 0:
            is_empty = True
    elif hasattr(uploaded_file, 'getbuffer'):
        if uploaded_file.getbuffer().nbytes == 0:
            is_empty = True
    else:
        try:
            if hasattr(uploaded_file, 'seek') and hasattr(uploaded_file, 'tell'):
                current_pos = uploaded_file.tell()
                uploaded_file.seek(0, os.SEEK_END)
                size = uploaded_file.tell()
                uploaded_file.seek(current_pos)
                if size == 0:
                    is_empty = True
        except Exception:
            pass
            
    if is_empty:
        return False, "The uploaded file is empty (0 bytes). Please upload a valid image."
        
    try:
        # Seek back to 0 just in case it has been read before
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            
        pil_image = Image.open(uploaded_file)
        pil_image.verify()  # Fast structural verification without decoding
        
        # Re-open and convert to RGB (since verify() closes the stream or leaves it at end)
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        pil_image = Image.open(uploaded_file).convert("RGB")
        return True, pil_image
    except Exception as e:
        return False, f"The uploaded file is corrupted or not a valid image. Details: {str(e)}"

def check_blurriness(pil_image, threshold=80.0):
    """
    Measures the sharpness of the image using Laplacian variance.
    Returns:
        (True, variance) if sharp enough.
        (False, error_msg) if too blurry or out-of-focus.
    """
    try:
        img_np = np.array(pil_image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < threshold:
            return False, f"The uploaded image is too blurry or lacks detail (sharpness score: {variance:.1f}). Please upload a clearer, in-focus photo of a tomato leaf."
        return True, variance
    except Exception as e:
        # If OpenCV operations fail, log and proceed (fallback-friendly)
        return True, 100.0

def check_organic_colors(pil_image, min_percentage=0.08):
    """
    Validates if the image contains sufficient plant-like colors (greens, yellows, browns).
    Returns:
        (True, ratio) if plant colors are present.
        (False, error_msg) if colors match non-leaf objects (laptops, walls, etc.).
    """
    try:
        img_np = np.array(pil_image)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Plant colors: Hue in [10, 85], Saturation/Value in [25, 255]
        # This covers bright greens, healthy leaves, yellow spots, and dry brown spots
        lower_bound = np.array([10, 25, 25])
        upper_bound = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        organic_ratio = np.sum(mask > 0) / mask.size
        
        if organic_ratio < min_percentage:
            return False, f"No plant leaf colors detected (leaf pixel ratio: {organic_ratio:.1%}). Please ensure a tomato leaf is clearly visible under good lighting."
        return True, organic_ratio
    except Exception as e:
        return True, 1.0

def check_imagenet_class(pil_image):
    """
    Uses a pretrained ImageNet model to verify if the image is a plant or leaf.
    Returns:
        (True, detected_class) if validation passes or fallback is active.
        (False, error_msg) if a non-leaf object is detected with high confidence.
    """
    try:
        model, weights = get_imagenet_model()
    except Exception as e:
        # Graceful fallback: If offline/firewalled on Streamlit Cloud, log and pass.
        # This ensures we don't break the application when PyTorch weights download fails.
        return True, "skipped (offline/no-internet)"
        
    try:
        preprocess = weights.transforms()
        input_tensor = preprocess(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
            
        categories = weights.meta["categories"]
        top3_prob, top3_idx = torch.topk(probs, 3)
        
        top3_classes = [categories[idx.item()] for idx in top3_idx]
        
        # Check if any of the top 3 classes are plant-related
        is_plant = False
        matched_class = None
        for cat in top3_classes:
            cat_lower = cat.lower()
            if any(kw in cat_lower for kw in PLANT_KEYWORDS):
                is_plant = True
                matched_class = cat
                break
                
        if not is_plant:
            top_class = top3_classes[0]
            # Replace underscores and tidy up the class name
            top_class_clean = top_class.replace("_", " ").title()
            return False, f"The uploaded image does not appear to be a plant leaf (detected: '{top_class_clean}'). Please upload a clear photo of a tomato leaf."
            
        return True, matched_class
    except Exception as e:
        # Any unexpected runtime error during PyTorch inference will gracefully pass
        return True, f"skipped (runtime error: {str(e)})"

def validate_uploaded_image(uploaded_file):
    """
    Orchestrates the full preprocessing validation pipeline.
    Returns:
        (True, pil_image) if all checks pass.
        (False, error_msg) if any check fails.
    """
    # 1. Corruption and empty checks
    ok, res = check_corruption_and_empty(uploaded_file)
    if not ok:
        return False, res
    pil_image = res
    
    # 2. Blurriness check
    ok, res = check_blurriness(pil_image)
    if not ok:
        return False, res
        
    # 3. Organic colors check
    ok, res = check_organic_colors(pil_image)
    if not ok:
        return False, res
        
    # 4. Pretrained ImageNet model validation
    ok, res = check_imagenet_class(pil_image)
    if not ok:
        return False, res
        
    return True, pil_image
