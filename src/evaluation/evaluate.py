import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import json

def evaluate_model(config, model, test_loader, model_name="baseline"):
    model = model.to(config.DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, severities in test_loader:
            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            cls_preds, _ = model(imgs)
            _, predicted = torch.max(cls_preds.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    with open(os.path.join(config.METRICS_DIR, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.ALLOWED_FOLDERS, yticklabels=config.ALLOWED_FOLDERS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{model_name}_cm.png"))
    plt.close()
    
    return metrics

def compare_models(config, metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(os.path.join(config.METRICS_DIR, "model_comparison.csv"))
    
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "model_comparison.png"))
    plt.close()
