import os
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from config.config import Config
from src.models.classifier import ResNetClassifier
from src.utils.utils import GradCAM

@st.cache_resource
def load_model(model_path):
    device = Config.DEVICE
    model = ResNetClassifier(len(Config.ALLOWED_FOLDERS), len(Config.SEVERITY_LEVELS)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception:
        return None

def main():
    st.set_page_config(page_title="SPC-GAN Disease Detection", layout="wide")
    st.title("Tomato Leaf Disease Detection & Severity Estimation")
    
    model = None
    for ckpt_name in ["best_classifier.pth", "baseline_classifier.pth", "demo_classifier.pth"]:
        model_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(model_path):
            model = load_model(model_path)
            if model is not None:
                st.success(f"Loaded model from {ckpt_name}")
                break
                
    if model is None:
        st.warning("Classifier checkpoint not found. The app is running in UI-only mode. Please train the model to enable predictions.")
        
    uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if model is None:
            st.error("Predictions are unavailable without a trained model.")
            return
        
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            cls_preds, sev_preds = model(input_tensor)
            cls_probs = torch.nn.functional.softmax(cls_preds, dim=1)
            sev_probs = torch.nn.functional.softmax(sev_preds, dim=1)
            
            cls_conf, cls_idx = torch.max(cls_probs, 1)
            sev_conf, sev_idx = torch.max(sev_probs, 1)
            
        predicted_class = Config.ALLOWED_FOLDERS[cls_idx.item()]
        predicted_severity = Config.SEVERITY_LEVELS[sev_idx.item()]
        
        st.subheader("Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Disease Class", predicted_class.replace("Tomato_", "").replace("_", " "))
            st.text(f"Confidence: {cls_conf.item():.2%}")
        with col2:
            st.metric("Severity", predicted_severity.capitalize())
            st.text(f"Confidence: {sev_conf.item():.2%}")
            
        if st.button("Generate Grad-CAM Visualization"):
            grad_cam = GradCAM(model, model.get_target_layer())
            cam = grad_cam(input_tensor, class_idx=cls_idx)
            
            cam_img = cam[0, 0].cpu().numpy()
            
            orig_img = np.array(image.resize((Config.IMG_SIZE, Config.IMG_SIZE))) / 255.0
            
            heatmap = plt.get_cmap('jet')(cam_img)[:, :, :3]
            overlay = 0.5 * heatmap + 0.5 * orig_img
            
            st.image(overlay, caption="Grad-CAM Activation", use_column_width=True)

if __name__ == "__main__":
    main()
