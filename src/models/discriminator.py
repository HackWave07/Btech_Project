import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes, num_severities, img_size=256):
        super().__init__()
        
        self.class_emb = nn.Embedding(num_classes, img_size * img_size)
        self.sev_emb = nn.Embedding(num_severities, img_size * img_size)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(5, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        
        ds_size = img_size // 16
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1))
        self.aux_cls = nn.Sequential(nn.Linear(512 * ds_size ** 2, num_classes))
        self.aux_sev = nn.Sequential(nn.Linear(512 * ds_size ** 2, num_severities))
        
    def forward(self, img, labels, severities):
        c_emb = self.class_emb(labels).view(labels.shape[0], 1, img.shape[2], img.shape[3])
        s_emb = self.sev_emb(severities).view(severities.shape[0], 1, img.shape[2], img.shape[3])
        
        d_in = torch.cat((img, c_emb, s_emb), 1)
        
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        
        validity = self.adv_layer(out)
        cls_pred = self.aux_cls(out)
        sev_pred = self.aux_sev(out)
        
        return validity, cls_pred, sev_pred
