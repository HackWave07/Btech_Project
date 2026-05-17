import os
import torch

class Config:
    # Root of the repository (two levels up from src/config/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "PlantVillage")
    
    ALLOWED_FOLDERS = [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_YellowLeafCurlVirus",
        "Tomato_Tomato_mosaic_virus",
        "Tomato_healthy"
    ]
    
    SEVERITY_LEVELS = ["mild", "moderate", "severe"]
    
    # Storage directories
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
    TRAINED_DIR   = os.path.join(BASE_DIR, "models", "trained")
    LOG_DIR        = os.path.join(BASE_DIR, "logs")
    PLOT_DIR       = os.path.join(BASE_DIR, "assets", "sample_outputs")
    METRICS_DIR    = os.path.join(BASE_DIR, "metrics")
    GEN_IMG_DIR    = os.path.join(BASE_DIR, "data", "synthetic")
    
    IMG_SIZE = 256
    CHANNELS = 3
    Z_DIM = 100
    
    BATCH_SIZE = 32
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    
    GAN_EPOCHS = 200
    LR_G = 0.0002
    LR_D = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    
    CLS_EPOCHS = 50
    LR_CLS = 0.001
    
    LAMBDA_ADV = 1.0
    LAMBDA_CLS = 10.0
    LAMBDA_ATT = 5.0
    LAMBDA_GP = 10.0
    
    SEED = 42
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    MIXED_PRECISION = True if DEVICE == "cuda" else False
    
    @classmethod
    def setup_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.TRAINED_DIR, cls.LOG_DIR,
                  cls.PLOT_DIR, cls.METRICS_DIR, cls.GEN_IMG_DIR]:
            os.makedirs(d, exist_ok=True)
