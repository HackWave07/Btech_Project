import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from src.config.config import Config
from src.models.generator import Generator

def main(num_samples_per_class=100):
    Config.setup_dirs()
    
    device = Config.DEVICE
    generator = Generator(Config.Z_DIM, len(Config.ALLOWED_FOLDERS), len(Config.SEVERITY_LEVELS), Config.IMG_SIZE).to(device)
    
    gen_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best.pth")
    if not os.path.exists(gen_path):
        print("Generator checkpoint not found. Train the GAN first.")
        return
        
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    generator.eval()
    
    syn_dir = os.path.join(Config.DATA_DIR, "Synthetic_Augmented")
    os.makedirs(syn_dir, exist_ok=True)
    
    transform = T.ToPILImage()
    
    with torch.no_grad():
        for c_idx, class_name in enumerate(Config.ALLOWED_FOLDERS):
            class_dir = os.path.join(syn_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for s_idx, severity_name in enumerate(Config.SEVERITY_LEVELS):
                print(f"Generating for {class_name} - {severity_name}")
                
                for i in tqdm(range(0, num_samples_per_class, Config.BATCH_SIZE)):
                    batch_size = min(Config.BATCH_SIZE, num_samples_per_class - i)
                    z = torch.randn(batch_size, Config.Z_DIM, device=device)
                    labels = torch.full((batch_size,), c_idx, dtype=torch.long, device=device)
                    severities = torch.full((batch_size,), s_idx, dtype=torch.long, device=device)
                    
                    fake_imgs, _ = generator(z, labels, severities)
                    fake_imgs = (fake_imgs + 1) / 2.0
                    
                    for j in range(batch_size):
                        img = transform(fake_imgs[j].cpu())
                        img.save(os.path.join(class_dir, f"syn_{severity_name}_{i+j}.png"))
                        
if __name__ == "__main__":
    main()
