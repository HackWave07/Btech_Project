import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, num_severities):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()
        
        self.cls_head = nn.Linear(num_ftrs, num_classes)
        self.sev_head = nn.Linear(num_ftrs, num_severities)
        
    def forward(self, x):
        features = self.backbone(x)
        cls_preds = self.cls_head(features)
        sev_preds = self.sev_head(features)
        return cls_preds, sev_preds
    
    def get_target_layer(self):
        return self.backbone.layer4[-1].conv2
