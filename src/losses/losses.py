import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAMConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, attn_map, grad_cam_map):
        b, c, h, w = attn_map.size()
        attn_map = F.interpolate(attn_map, size=(grad_cam_map.shape[2], grad_cam_map.shape[3]), mode='bilinear', align_corners=False)
        loss = F.mse_loss(attn_map, grad_cam_map)
        return loss

class AttentionRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, attn_map):
        return torch.mean(torch.abs(attn_map))

def compute_gradient_penalty(D, real_samples, fake_samples, labels, severities, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _, _ = D(interpolates, labels, severities)
    fake = torch.ones(real_samples.size(0), 1, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
