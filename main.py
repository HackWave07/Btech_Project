import argparse
import sys
import os
import torch

from src.config.config import Config
from src.data.dataset import get_dataloaders
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.classifier import ResNetClassifier
from src.training.train_classifier import train_classifier
from src.training.train_gan import train_gan
from src.evaluation.evaluate import evaluate_model, compare_models
from src.utils.helpers import set_seed

def verify_environment():
    missing_packages = []
    try:
        import torch
        import torchvision
    except ImportError:
        missing_packages.extend(['torch', 'torchvision'])
        
    try:
        import cv2
    except ImportError:
        missing_packages.append('opencv-python')
        
    try:
        import PIL
    except ImportError:
        missing_packages.append('Pillow')
        
    try:
        import sklearn
    except ImportError:
        missing_packages.append('scikit-learn')
        
    try:
        import pandas
    except ImportError:
        missing_packages.append('pandas')
        
    try:
        import matplotlib
        import seaborn
    except ImportError:
        missing_packages.extend(['matplotlib', 'seaborn'])
        
    try:
        import streamlit
    except ImportError:
        missing_packages.append('streamlit')
        
    if missing_packages:
        print("ERROR: Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)

def verify_dataset():
    if not os.path.exists(Config.DATA_DIR):
        print(f"ERROR: Dataset directory not found at {Config.DATA_DIR}")
        print("Please ensure the PlantVillage dataset is placed in the project root.")
        sys.exit(1)
        
    found_folders = [f for f in Config.ALLOWED_FOLDERS if os.path.isdir(os.path.join(Config.DATA_DIR, f))]
    if len(found_folders) == 0:
        print(f"ERROR: No valid tomato dataset folders found in {Config.DATA_DIR}.")
        print("Expected folders like: Tomato_Bacterial_spot, Tomato_healthy, etc.")
        sys.exit(1)
        
    return found_folders

def main():
    parser = argparse.ArgumentParser(description="SPC-GAN Pipeline for Tomato Leaf Disease Detection")
    parser.add_argument('--mode', type=str, required=True, choices=['prepare', 'train_classifier', 'train_gan', 'evaluate', 'demo', 'generate', 'inference'], help="Pipeline mode to run")
    parser.add_argument('--image_path', type=str, help="Path to image for inference")
    parser.add_argument('--model_path', type=str, help="Path to model for inference")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of synthetic samples per class to generate")
    args = parser.parse_args()
    
    verify_environment()
    Config.setup_dirs()
    set_seed(Config.SEED)
    
    if args.mode == 'prepare':
        verify_dataset()
        import scripts.prepare_dataset as prep
        prep.main()
        
    elif args.mode == 'train_classifier':
        verify_dataset()
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(Config)
        model = ResNetClassifier(num_classes, len(Config.SEVERITY_LEVELS))
        print("Training baseline classifier...")
        train_classifier(Config, model, train_loader, val_loader, save_name="baseline_classifier.pth")
        
    elif args.mode == 'train_gan':
        verify_dataset()
        train_loader, val_loader, _, num_classes = get_dataloaders(Config)
        
        classifier = ResNetClassifier(num_classes, len(Config.SEVERITY_LEVELS)).to(Config.DEVICE)
        clf_path = os.path.join(Config.CHECKPOINT_DIR, "baseline_classifier.pth")
        if os.path.exists(clf_path):
            classifier.load_state_dict(torch.load(clf_path, map_location=Config.DEVICE))
        else:
            print("Warning: Baseline classifier not found. Training GAN without pretrained classifier guidance might be suboptimal.")
            
        generator = Generator(Config.Z_DIM, num_classes, len(Config.SEVERITY_LEVELS), Config.IMG_SIZE)
        discriminator = Discriminator(num_classes, len(Config.SEVERITY_LEVELS), Config.IMG_SIZE)
        
        print("Training SPC-GAN...")
        train_gan(Config, generator, discriminator, classifier, train_loader)
        
    elif args.mode == 'evaluate':
        verify_dataset()
        _, _, test_loader, num_classes = get_dataloaders(Config)
        
        metrics_dict = {}
        
        base_model = ResNetClassifier(num_classes, len(Config.SEVERITY_LEVELS))
        base_path = os.path.join(Config.CHECKPOINT_DIR, "baseline_classifier.pth")
        if os.path.exists(base_path):
            base_model.load_state_dict(torch.load(base_path, map_location=Config.DEVICE))
            metrics_dict['Baseline'] = evaluate_model(Config, base_model, test_loader, "baseline")
            
        # For simplicity, if augmented model is trained, it would be loaded here.
        
        if metrics_dict:
            compare_models(Config, metrics_dict)
            print("Evaluation complete. Check plots/ and metrics/ directories.")
        else:
            print("No models found to evaluate.")
            
    elif args.mode == 'demo':
        verify_dataset()
        Config.CLS_EPOCHS = 1
        Config.GAN_EPOCHS = 1
        Config.BATCH_SIZE = 8
        print("Running Demo Mode (1 epoch, batch size 8)")
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(Config)
        
        print("Training demo classifier...")
        classifier = ResNetClassifier(num_classes, len(Config.SEVERITY_LEVELS))
        train_classifier(Config, classifier, train_loader, val_loader, save_name="demo_classifier.pth")
        
        print("Training demo GAN...")
        generator = Generator(Config.Z_DIM, num_classes, len(Config.SEVERITY_LEVELS), Config.IMG_SIZE)
        discriminator = Discriminator(num_classes, len(Config.SEVERITY_LEVELS), Config.IMG_SIZE)
        train_gan(Config, generator, discriminator, classifier, train_loader)
        
        print("Demo evaluation...")
        metrics = evaluate_model(Config, classifier, test_loader, "demo_classifier")
        print("Demo complete. Check generated_images/ and plots/ directories.")
        
    elif args.mode == 'generate':
        import scripts.generate_synthetic as gen
        gen.main(args.num_samples)
        
    elif args.mode == 'inference':
        if not args.image_path or not args.model_path:
            print("ERROR: --image_path and --model_path are required for inference mode.")
            sys.exit(1)
        from src.inference.predict import predict
        predict(args.image_path, args.model_path)

if __name__ == "__main__":
    main()
