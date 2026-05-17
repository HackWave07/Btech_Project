"""
SPC-GAN – Tomato Leaf Disease Detection
Entry point: streamlit run app.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# ── Make sure project root is on sys.path ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config.config import Config
from src.models.classifier import ResNetClassifier
from src.utils.visualization import GradCAM

# ── Disease information lookup ────────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "desc": "Small, dark, water-soaked spots on leaves and fruit caused by *Xanthomonas* bacteria.",
        "treatment": "Copper-based bactericides; avoid overhead irrigation.",
        "severity_tip": "Higher severity indicates widespread spotting across the leaf surface.",
    },
    "Tomato_Early_blight": {
        "desc": "Concentric ring lesions ('target spots') caused by *Alternaria solani* fungus.",
        "treatment": "Chlorothalonil or mancozeb fungicide; crop rotation.",
        "severity_tip": "Severe cases show full leaf yellowing and defoliation.",
    },
    "Tomato_Late_blight": {
        "desc": "Water-soaked lesions that rapidly turn brown/black, caused by *Phytophthora infestans*.",
        "treatment": "Metalaxyl fungicides; remove and destroy infected tissue immediately.",
        "severity_tip": "Late blight spreads very quickly—act on even mild symptoms.",
    },
    "Tomato_Leaf_Mold": {
        "desc": "Yellow patches on the upper leaf surface with olive-green mold beneath.",
        "treatment": "Improve air circulation; apply copper or sulfur fungicides.",
        "severity_tip": "Severe cases collapse entire leaflets.",
    },
    "Tomato_Septoria_leaf_spot": {
        "desc": "Small, circular spots with dark borders and light centers caused by *Septoria lycopersici*.",
        "treatment": "Remove affected leaves; apply mancozeb or chlorothalonil.",
        "severity_tip": "Disease starts on lower leaves and moves upward.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "desc": "Tiny arachnid pests that cause stippled, yellowing, or bronzed foliage.",
        "treatment": "Miticides or neem oil; increase humidity; introduce predatory mites.",
        "severity_tip": "Webbing on leaves indicates severe infestation.",
    },
    "Tomato_Target_Spot": {
        "desc": "Brown lesions with concentric rings resembling a target, caused by *Corynespora cassiicola*.",
        "treatment": "Tebuconazole or azoxystrobin fungicides; remove infected debris.",
        "severity_tip": "Defoliation accelerates under warm and humid conditions.",
    },
    "Tomato_Tomato_YellowLeafCurlVirus": {
        "desc": "Viral disease transmitted by whiteflies causing leaf curling and yellowing.",
        "treatment": "Control whitefly populations; use resistant varieties; remove infected plants.",
        "severity_tip": "No chemical cure exists—prevention is critical.",
    },
    "Tomato_Tomato_mosaic_virus": {
        "desc": "Mosaic patterns on leaves caused by *Tomato mosaic virus* (ToMV).",
        "treatment": "Use certified virus-free seeds; sanitize tools; remove infected plants.",
        "severity_tip": "Mild symptoms can still reduce yield significantly.",
    },
    "Tomato_healthy": {
        "desc": "The leaf shows no signs of disease. 🌿",
        "treatment": "Maintain regular fertilization, irrigation, and pest monitoring.",
        "severity_tip": "Continue regular crop monitoring to catch issues early.",
    },
}

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = Config.DEVICE
    model = ResNetClassifier(
        len(Config.ALLOWED_FOLDERS), len(Config.SEVERITY_LEVELS)
    ).to(device)
    for ckpt_name in ["best_classifier.pth", "baseline_classifier.pth", "demo_classifier.pth"]:
        model_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                return model, ckpt_name
            except Exception:
                continue
    return None, None


# ── Image transform ───────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ── Confidence bar chart ──────────────────────────────────────────────────────
def confidence_chart(labels, probs, title, color):
    fig, ax = plt.subplots(figsize=(5, max(2, len(labels) * 0.35)))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, probs, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.replace("Tomato_", "").replace("_", " ") for l in labels],
                       fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.bar_label(bars, fmt="%.1%%", padding=3, fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SPC-GAN | Tomato Disease Detection",
        page_icon="🍅",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg",
                 use_column_width=True)
        st.title("🍅 SPC-GAN")
        st.caption("Tomato Leaf Disease Detection & Severity Estimation")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🔍 Diagnose Leaf", "📊 Model Info", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.divider()

        model, ckpt_name = load_model()
        if model:
            st.success(f"✅ Model loaded  \n`{ckpt_name}`")
            st.caption(f"Device: `{Config.DEVICE.upper()}`")
        else:
            st.warning("⚠️ No trained model found.  \nRun training first.")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Diagnose Leaf
    # ═════════════════════════════════════════════════════════════════════════
    if page == "🔍 Diagnose Leaf":
        st.title("Tomato Leaf Disease Diagnosis")
        st.markdown(
            "Upload a clear photo of a tomato leaf to detect disease and estimate severity."
        )

        uploaded = st.file_uploader(
            "📁 Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="visible"
        )

        if uploaded is None:
            st.info("Upload an image to get started.")
            return

        image = Image.open(uploaded).convert("RGB")

        col_img, col_results = st.columns([1, 2], gap="large")

        with col_img:
            st.image(image, caption="Uploaded Leaf", use_column_width=True)

        with col_results:
            if model is None:
                st.error("Predictions unavailable — please train the model first.")
                return

            transform = get_transform()
            input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

            with torch.no_grad():
                cls_preds, sev_preds = model(input_tensor)
                cls_probs = F.softmax(cls_preds, dim=1).squeeze().cpu().numpy()
                sev_probs = F.softmax(sev_preds, dim=1).squeeze().cpu().numpy()

            cls_idx   = int(np.argmax(cls_probs))
            sev_idx   = int(np.argmax(sev_probs))
            cls_label = Config.ALLOWED_FOLDERS[cls_idx]
            sev_label = Config.SEVERITY_LEVELS[sev_idx]

            # ── Headline metrics ──────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("🌿 Disease",  cls_label.replace("Tomato_", "").replace("_", " "))
            m2.metric("⚠️ Severity", sev_label.capitalize())
            m3.metric("🎯 Confidence", f"{cls_probs[cls_idx]:.1%}")

            st.divider()

            # ── Confidence charts ─────────────────────────────────────────────
            cc1, cc2 = st.columns(2)
            with cc1:
                st.pyplot(
                    confidence_chart(
                        Config.ALLOWED_FOLDERS, cls_probs,
                        "Disease Probabilities", "#2ecc71"
                    )
                )
            with cc2:
                st.pyplot(
                    confidence_chart(
                        Config.SEVERITY_LEVELS, sev_probs,
                        "Severity Probabilities", "#e67e22"
                    )
                )

            st.divider()

            # ── Disease information panel ─────────────────────────────────────
            info = DISEASE_INFO.get(cls_label, {})
            if info:
                st.subheader("📋 Disease Information")
                st.markdown(f"**Description:** {info['desc']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
                st.info(f"**Severity note:** {info['severity_tip']}")

        # ── Grad-CAM section ──────────────────────────────────────────────────
        st.divider()
        if st.button("🔥 Generate Grad-CAM Heatmap", use_container_width=True):
            with st.spinner("Computing activation map…"):
                try:
                    grad_cam = GradCAM(model, model.get_target_layer())
                    cam = grad_cam(input_tensor, class_idx=cls_idx)
                    cam_np = cam[0, 0].cpu().numpy()

                    orig_np = np.array(
                        image.resize((Config.IMG_SIZE, Config.IMG_SIZE))
                    ) / 255.0
                    heatmap = plt.get_cmap("jet")(cam_np)[:, :, :3]
                    overlay = np.clip(0.55 * heatmap + 0.45 * orig_np, 0, 1)

                    gc1, gc2 = st.columns(2)
                    gc1.image(orig_np, caption="Original", use_column_width=True)
                    gc2.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Model Info
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📊 Model Info":
        st.title("Model Architecture")
        st.markdown("""
### SPC-GAN Pipeline

| Component | Architecture | Purpose |
|-----------|-------------|---------|
| **Generator** | Conditional GAN with attention | Synthesise disease-specific leaf images |
| **Discriminator** | Multi-output PatchGAN | Distinguish real vs fake; class & severity |
| **Classifier** | ResNet-18 (dual-head) | Predict disease class + severity level |

### Training Strategy
1. **Step 1 – Baseline Classifier**: Train ResNet-18 on real PlantVillage data.  
2. **Step 2 – SPC-GAN Training**: Train Generator/Discriminator with Grad-CAM consistency loss guided by the pretrained classifier.  
3. **Step 3 – Augmented Classifier**: Retrain classifier on real + synthetic data for improved generalisation.

### Key Innovations
- **Severity-conditioned generation** – GAN conditioned on both disease *class* and *severity level*.  
- **Grad-CAM Consistency Loss** – forces the generator to produce images where discriminative regions align with classifier attention maps.  
- **Attention Regularisation** – prevents attention collapse during GAN training.
        """)

        st.subheader("Supported Classes")
        for cls in Config.ALLOWED_FOLDERS:
            is_healthy = "healthy" in cls
            st.markdown(f"{'🟢' if is_healthy else '🔴'} `{cls}`")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: About
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "ℹ️ About":
        st.title("About SPC-GAN")
        st.markdown("""
**SPC-GAN** (Severity-conditioned Progressive Conditional GAN) is a B.Tech
research project for automated tomato leaf disease detection and severity
estimation using deep learning.

### Highlights
- Detects **10 tomato disease / health categories**  
- Estimates **3 severity levels** (mild · moderate · severe)  
- GAN-generated synthetic data reduces class imbalance  
- Grad-CAM heatmaps provide visual explainability  
- Fully deployable on **Streamlit Community Cloud**

### Dataset
[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — open-source
plant disease image dataset.

### Tech Stack
`PyTorch` · `torchvision` · `Streamlit` · `scikit-learn` · `OpenCV` · `Matplotlib`
        """)


if __name__ == "__main__":
    main()
