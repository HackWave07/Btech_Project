import sys
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

from config.config import Config
from src.models.classifier import ResNetClassifier

def predict(image_path, model_path):
    device = Config.DEVICE
    
    model = ResNetClassifier(len(Config.ALLOWED_FOLDERS), len(Config.SEVERITY_LEVELS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cls_preds, sev_preds = model(input_tensor)
        
        cls_probs = torch.nn.functional.softmax(cls_preds, dim=1)
        sev_probs = torch.nn.functional.softmax(sev_preds, dim=1)
        
        cls_conf, cls_idx = torch.max(cls_probs, 1)
        sev_conf, sev_idx = torch.max(sev_probs, 1)
        
    predicted_class = Config.ALLOWED_FOLDERS[cls_idx.item()]
    predicted_severity = Config.SEVERITY_LEVELS[sev_idx.item()]
    
    print(f"Predicted Class: {predicted_class} (Confidence: {cls_conf.item():.4f})")
    print(f"Predicted Severity: {predicted_severity} (Confidence: {sev_conf.item():.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <path_to_image> <path_to_model>")
    else:
        predict(sys.argv[1], sys.argv[2])
