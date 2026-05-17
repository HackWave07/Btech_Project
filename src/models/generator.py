import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, width, height = x.size()
        
        proj_query = self.query(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, width * height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        
        out = self.gamma * out + x
        
        # Convert N*N attention to 1*W*H spatial map for Grad-CAM consistency
        attn_spatial = attention.mean(dim=1).view(batch, 1, width, height)
        return out, attn_spatial

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, num_severities, img_size=256):
        super().__init__()
        self.z_dim = z_dim
        
        self.class_emb = nn.Embedding(num_classes, 50)
        self.sev_emb = nn.Embedding(num_severities, 10)
        
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(nn.Linear(z_dim + 60, 512 * self.init_size ** 2))
        
        self.conv_blocks1 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.attn = SelfAttention(128)
        
        self.conv_blocks2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, labels, severities):
        c_emb = self.class_emb(labels)
        s_emb = self.sev_emb(severities)
        
        gen_input = torch.cat((c_emb, s_emb, noise), -1)
        
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        
        out = self.conv_blocks1(out)
        out, attn_map = self.attn(out)
        out = self.conv_blocks2(out)
        img = self.final_block(out)
        
        return img, attn_map
