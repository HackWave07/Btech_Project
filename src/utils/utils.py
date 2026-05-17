import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_image_grid(tensor, path, nrow=8, normalize=True):
    vutils.save_image(tensor, path, nrow=nrow, normalize=normalize)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        b, c, h, w = x.size()
        out = self.model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
            
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            
        score = logits[torch.arange(b), class_idx]
        
        self.model.zero_grad()
        score.backward(torch.ones_like(score), retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam
