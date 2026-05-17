import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def train_classifier(config, model, train_loader, val_loader, save_name="best_classifier.pth"):
    model = model.to(config.DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LR_CLS)
    
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for imgs, labels, severities in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLS_EPOCHS}"):
            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            severities = severities.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    cls_preds, sev_preds = model(imgs)
                    loss_cls = criterion_cls(cls_preds, labels)
                    loss_sev = criterion_sev(sev_preds, severities)
                    loss = loss_cls + 0.5 * loss_sev
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_preds, sev_preds = model(imgs)
                loss_cls = criterion_cls(cls_preds, labels)
                loss_sev = criterion_sev(sev_preds, severities)
                loss = loss_cls + 0.5 * loss_sev
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels, severities in val_loader:
                imgs = imgs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                severities = severities.to(config.DEVICE)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        cls_preds, sev_preds = model(imgs)
                        loss_cls = criterion_cls(cls_preds, labels)
                        loss_sev = criterion_sev(sev_preds, severities)
                        loss = loss_cls + 0.5 * loss_sev
                else:
                    cls_preds, sev_preds = model(imgs)
                    loss_cls = criterion_cls(cls_preds, labels)
                    loss_sev = criterion_sev(sev_preds, severities)
                    loss = loss_cls + 0.5 * loss_sev
                    
                val_loss += loss.item()
                _, predicted = torch.max(cls_preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, save_name))
            
    with open(os.path.join(config.LOG_DIR, f"{save_name}_history.json"), "w") as f:
        json.dump(history, f)
        
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{save_name}_loss.png"))
    plt.close()
    
    return history
