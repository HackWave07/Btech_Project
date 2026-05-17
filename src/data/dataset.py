import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class TomatoDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        self.class_to_idx = {c: i for i, c in enumerate(sorted(df['class'].unique()))}
        self.severity_to_idx = {'mild': 0, 'moderate': 1, 'severe': 2}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['class'], row['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (256, 256), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        class_label = self.class_to_idx[row['class']]
        severity_label = self.severity_to_idx[row['severity']]
        
        return image, class_label, severity_label

def estimate_severity(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 'mild'
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        total_pixels = img.shape[0] * img.shape[1]
        brown_pixels = cv2.countNonZero(mask_brown)
        green_pixels = cv2.countNonZero(mask_green)
        leaf_pixels = brown_pixels + green_pixels
        
        if leaf_pixels == 0:
            return 'mild'
            
        ratio = brown_pixels / leaf_pixels
        
        if ratio < 0.1:
            return 'mild'
        elif ratio < 0.3:
            return 'moderate'
        else:
            return 'severe'
    except Exception:
        return 'mild'

def prepare_dataset_csv(data_dir, allowed_folders, out_csv):
    data = []
    for cls in allowed_folders:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
            
        for f in os.listdir(cls_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_path, f)
                severity = estimate_severity(img_path) if "healthy" not in cls else "mild"
                data.append({'filename': f, 'class': cls, 'severity': severity})
                
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    return df

def get_dataloaders(config):
    csv_path = os.path.join(config.BASE_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        df = prepare_dataset_csv(config.DATA_DIR, config.ALLOWED_FOLDERS, csv_path)
    else:
        df = pd.read_csv(csv_path)
        
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=config.SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=config.SEED)
    
    transform_train = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = TomatoDataset(train_df, config.DATA_DIR, transform=transform_train)
    val_dataset = TomatoDataset(val_df, config.DATA_DIR, transform=transform_val)
    test_dataset = TomatoDataset(test_df, config.DATA_DIR, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader, len(train_dataset.class_to_idx)
