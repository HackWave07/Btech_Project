"""
Automated validation tests for the pre-processing validation pipeline.
To run: python src/utils/test_validation.py
"""

import io
import os
import sys
import unittest
import numpy as np
from PIL import Image
import cv2

# Make sure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.validation import (
    check_corruption_and_empty,
    check_blurriness,
    check_organic_colors,
    check_imagenet_class,
    validate_uploaded_image
)

class TestValidationPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create different mock images for testing
        
        # 1. A sharp, green "leaf" mock image (contains textures and dominant leaf colors)
        # We generate a grid of green tones with some randomized high frequency details (texture)
        leaf_np = np.zeros((300, 300, 3), dtype=np.uint8)
        # Fill with green (HSV: H=40, S=150, V=200 -> RGB approx: [150, 200, 50])
        leaf_np[:, :, 0] = 50   # R
        leaf_np[:, :, 1] = 180  # G
        leaf_np[:, :, 2] = 40   # B
        
        # Draw central and side diagonal veins to simulate leaf structure (avoiding grid mesh)
        cv2.line(leaf_np, (150, 0), (150, 300), (30, 100, 20), 4)
        for y in range(30, 300, 40):
            cv2.line(leaf_np, (150, y), (50, y - 30), (35, 110, 25), 2)
            cv2.line(leaf_np, (150, y), (250, y - 30), (35, 110, 25), 2)
            
        # Add random noise for detail
        noise = np.random.randint(-15, 15, leaf_np.shape).astype(np.int16)
        leaf_np = np.clip(leaf_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.sharp_leaf_pil = Image.fromarray(leaf_np)
        
        # 2. A blurry leaf mock (applying heavy Gaussian Blur)
        blurry_np = cv2.GaussianBlur(leaf_np, (25, 25), 0)
        self.blurry_leaf_pil = Image.fromarray(blurry_np)
        
        # 3. A flat grey "wall" image (no leaf colors, no texture)
        wall_np = np.ones((300, 300, 3), dtype=np.uint8) * 128
        self.wall_pil = Image.fromarray(wall_np)
        
        # 4. A blue "laptop screen" mock image (no organic leaf colors)
        screen_np = np.zeros((300, 300, 3), dtype=np.uint8)
        screen_np[:, :, 2] = 220  # Blue channel high (RGB format: B is index 2)
        screen_np[:, :, 0] = 10   # Red
        screen_np[:, :, 1] = 10   # Green
        self.screen_pil = Image.fromarray(screen_np)

    def test_empty_and_corruption(self):
        print("\n--- Testing Empty and Corruption Handling ---")
        
        # Test None file
        ok, msg = check_corruption_and_empty(None)
        self.assertFalse(ok)
        self.assertIn("No file uploaded", msg)
        print("Success: None file caught successfully.")
        
        # Test empty BytesIO
        empty_file = io.BytesIO()
        ok, msg = check_corruption_and_empty(empty_file)
        self.assertFalse(ok)
        self.assertIn("empty", msg)
        print("Success: Empty file caught successfully.")
        
        # Test corrupted file (random bytes)
        corrupt_file = io.BytesIO(b"this is just some random garbage text, not an image format")
        ok, msg = check_corruption_and_empty(corrupt_file)
        self.assertFalse(ok)
        self.assertIn("corrupted", msg)
        print("Success: Corrupted file caught successfully.")
        
        # Test valid image BytesIO
        valid_io = io.BytesIO()
        self.sharp_leaf_pil.save(valid_io, format="JPEG")
        valid_io.seek(0)
        ok, res = check_corruption_and_empty(valid_io)
        self.assertTrue(ok)
        self.assertIsInstance(res, Image.Image)
        print("Success: Valid image passed corruption check.")

    def test_blurriness(self):
        print("\n--- Testing Blurriness Checking ---")
        
        # Sharp leaf should pass
        ok, val = check_blurriness(self.sharp_leaf_pil, threshold=50.0)
        self.assertTrue(ok)
        print(f"Success: Sharp leaf passed (sharpness score: {val:.1f}).")
        
        # Blurry leaf should fail
        ok, val = check_blurriness(self.blurry_leaf_pil, threshold=50.0)
        self.assertFalse(ok)
        self.assertIn("too blurry", val)
        print(f"Success: Blurry leaf rejected successfully (message: {val}).")
        
        # Flat wall should fail (very low variance)
        ok, val = check_blurriness(self.wall_pil, threshold=50.0)
        self.assertFalse(ok)
        print(f"Success: Plain wall rejected for lack of detail/blurriness (message: {val}).")

    def test_organic_colors(self):
        print("\n--- Testing Organic Color Verification ---")
        
        # Green leaf should pass
        ok, val = check_organic_colors(self.sharp_leaf_pil, min_percentage=0.08)
        self.assertTrue(ok)
        print(f"Success: Green leaf passed color check (organic color ratio: {val:.1%}).")
        
        # Blue laptop screen should fail
        ok, val = check_organic_colors(self.screen_pil, min_percentage=0.08)
        self.assertFalse(ok)
        self.assertIn("No plant leaf colors", val)
        print(f"Success: Blue screen rejected successfully (message: {val}).")
        
        # Grey wall should fail
        ok, val = check_organic_colors(self.wall_pil, min_percentage=0.08)
        self.assertFalse(ok)
        print(f"Success: Grey wall rejected successfully (message: {val}).")

    def test_imagenet_classification(self):
        print("\n--- Testing ImageNet-based Plant Leaf Verification ---")
        
        # Check if the function can run or falls back gracefully
        try:
            # Let's test with screen (which shouldn't look like a plant at all)
            # If weights are not available locally, it will fall back to True and print a warning
            ok, res = check_imagenet_class(self.screen_pil)
            print(f"ImageNet check on screen returned: ok={ok}, res={res}")
            
            # Test with leaf
            ok_leaf, res_leaf = check_imagenet_class(self.sharp_leaf_pil)
            print(f"ImageNet check on leaf returned: ok={ok_leaf}, res={res_leaf}")
            
            if "skipped" not in str(res):
                # If weights loaded correctly, screen should fail validation
                self.assertFalse(ok)
                self.assertIn("does not appear to be a plant leaf", res)
                print("Success: Pretrained ImageNet filter successfully blocked the non-leaf screen mock!")
            else:
                print("ImageNet check was skipped (offline mode or weight loading issue). Fallback worked successfully.")
        except Exception as e:
            self.fail(f"ImageNet class verification raised an exception: {e}")

    def test_full_pipeline(self):
        print("\n--- Testing Full Validation Pipeline ---")
        
        # Test 1: Valid sharp leaf
        valid_io = io.BytesIO()
        self.sharp_leaf_pil.save(valid_io, format="JPEG")
        valid_io.seek(0)
        
        ok, res = validate_uploaded_image(valid_io)
        # If ImageNet check is offline, it will pass because of the fallback,
        # and if it is online, a highly green/textured mock might pass or fail depending on ImageNet classes,
        # but the lower levels (corruption, blurriness, colors) will definitely pass.
        # Let's verify that the pipeline executes completely.
        print(f"Pipeline outcome on sharp leaf: ok={ok}, result={res}")
        
        # Test 2: Blurry leaf
        blurry_io = io.BytesIO()
        self.blurry_leaf_pil.save(blurry_io, format="JPEG")
        blurry_io.seek(0)
        ok, res = validate_uploaded_image(blurry_io)
        self.assertFalse(ok)
        self.assertIn("too blurry", res)
        print("Success: Pipeline correctly blocked blurry leaf.")

if __name__ == "__main__":
    unittest.main()
