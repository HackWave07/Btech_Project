"""
scripts/export_model.py
Export a trained ResNetClassifier checkpoint to TorchScript.

Usage:
    python scripts/export_model.py --checkpoint models/checkpoints/best_classifier.pth \
                                   --output     models/trained/classifier_scripted.pt
"""

import os
import sys
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config.config import Config
from src.models.classifier import ResNetClassifier


def export(checkpoint_path: str, output_path: str) -> None:
    device = torch.device("cpu")  # Export on CPU for portability

    num_classes  = len(Config.ALLOWED_FOLDERS)
    num_severity = len(Config.SEVERITY_LEVELS)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = ResNetClassifier(num_classes, num_severity)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Trace with a dummy input
    dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    traced = torch.jit.trace(model, dummy)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)
    print(f"TorchScript model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export classifier to TorchScript")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(Config.CHECKPOINT_DIR, "best_classifier.pth"),
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(Config.TRAINED_DIR, "classifier_scripted.pt"),
        help="Output path for the TorchScript .pt file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
