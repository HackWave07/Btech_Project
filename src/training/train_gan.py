import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from src.losses.losses import compute_gradient_penalty, GradCAMConsistencyLoss, AttentionRegularizationLoss
from src.utils.utils import save_image_grid, GradCAM

def train_gan(config, generator, discriminator, classifier, train_loader):
    generator = generator.to(config.DEVICE)
    discriminator = discriminator.to(config.DEVICE)
    classifier = classifier.to(config.DEVICE)
    classifier.eval()
    
    opt_G = Adam(generator.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = Adam(discriminator.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sev = nn.CrossEntropyLoss()
    grad_cam_loss = GradCAMConsistencyLoss()
    attn_loss = AttentionRegularizationLoss()
    
    grad_cam = GradCAM(classifier, classifier.get_target_layer())
    
    scaler_G = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None
    scaler_D = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None
    
    history = {'d_loss': [], 'g_loss': []}
    
    fixed_noise = torch.randn(64, config.Z_DIM, device=config.DEVICE)
    fixed_labels = torch.randint(0, len(config.ALLOWED_FOLDERS), (64,), device=config.DEVICE)
    fixed_severities = torch.randint(0, len(config.SEVERITY_LEVELS), (64,), device=config.DEVICE)
    
    for epoch in range(config.GAN_EPOCHS):
        generator.train()
        discriminator.train()
        
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        gen_steps = 0
        
        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{config.GAN_EPOCHS}")
        for imgs, labels, severities in pbar:
            batch_size = imgs.size(0)
            real_imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            severities = severities.to(config.DEVICE)
            
            # Train Discriminator
            opt_D.zero_grad()
            z = torch.randn(batch_size, config.Z_DIM, device=config.DEVICE)
            
            with torch.no_grad():
                fake_imgs, _ = generator(z, labels, severities)
                
            if config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    real_validity, real_cls, real_sev = discriminator(real_imgs, labels, severities)
                    fake_validity, fake_cls, fake_sev = discriminator(fake_imgs.detach(), labels, severities)
                    
                    d_adv_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                    d_cls_loss = criterion_cls(real_cls, labels) + criterion_cls(fake_cls, labels)
                    d_sev_loss = criterion_sev(real_sev, severities) + criterion_sev(fake_sev, severities)
                    
                    gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels, severities, config.DEVICE)
                    
                    d_loss = d_adv_loss + config.LAMBDA_GP * gp + config.LAMBDA_CLS * (d_cls_loss + 0.5 * d_sev_loss)
                    
                scaler_D.scale(d_loss).backward()
                scaler_D.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler_D.step(opt_D)
                scaler_D.update()
            else:
                real_validity, real_cls, real_sev = discriminator(real_imgs, labels, severities)
                fake_validity, fake_cls, fake_sev = discriminator(fake_imgs.detach(), labels, severities)
                
                d_adv_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_cls_loss = criterion_cls(real_cls, labels) + criterion_cls(fake_cls, labels)
                d_sev_loss = criterion_sev(real_sev, severities) + criterion_sev(fake_sev, severities)
                
                gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels, severities, config.DEVICE)
                
                d_loss = d_adv_loss + config.LAMBDA_GP * gp + config.LAMBDA_CLS * (d_cls_loss + 0.5 * d_sev_loss)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_D.step()
                
            # Train Generator
            if pbar.n % 5 == 0:
                gen_steps += 1
                opt_G.zero_grad()
                
                if config.MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        gen_imgs, attn_map = generator(z, labels, severities)
                        fake_validity, fake_cls, fake_sev = discriminator(gen_imgs, labels, severities)
                        
                        g_adv_loss = -torch.mean(fake_validity)
                        g_cls_loss = criterion_cls(fake_cls, labels) + 0.5 * criterion_sev(fake_sev, severities)
                        
                        cam_map = grad_cam(gen_imgs, class_idx=labels)
                        g_gc_loss = grad_cam_loss(attn_map, cam_map)
                        g_attn_loss = attn_loss(attn_map)
                        
                        g_loss = g_adv_loss + config.LAMBDA_CLS * g_cls_loss + config.LAMBDA_ATT * (g_gc_loss + 0.1 * g_attn_loss)
                        
                    scaler_G.scale(g_loss).backward()
                    scaler_G.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    scaler_G.step(opt_G)
                    scaler_G.update()
                else:
                    gen_imgs, attn_map = generator(z, labels, severities)
                    fake_validity, fake_cls, fake_sev = discriminator(gen_imgs, labels, severities)
                    
                    g_adv_loss = -torch.mean(fake_validity)
                    g_cls_loss = criterion_cls(fake_cls, labels) + 0.5 * criterion_sev(fake_sev, severities)
                    
                    cam_map = grad_cam(gen_imgs, class_idx=labels)
                    g_gc_loss = grad_cam_loss(attn_map, cam_map)
                    g_attn_loss = attn_loss(attn_map)
                    
                    g_loss = g_adv_loss + config.LAMBDA_CLS * g_cls_loss + config.LAMBDA_ATT * (g_gc_loss + 0.1 * g_attn_loss)
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    opt_G.step()
                    
                g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            
            pbar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item() if 'g_loss' in locals() else 0)
            
        d_loss_epoch /= len(train_loader)
        g_loss_epoch = g_loss_epoch / gen_steps if gen_steps > 0 else 0.0
        
        history['d_loss'].append(d_loss_epoch)
        history['g_loss'].append(g_loss_epoch)
        
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(config.CHECKPOINT_DIR, "generator_latest.pth"))
            torch.save(discriminator.state_dict(), os.path.join(config.CHECKPOINT_DIR, "discriminator_latest.pth"))
            
            with torch.no_grad():
                generator.eval()
                sample_imgs, _ = generator(fixed_noise, fixed_labels, fixed_severities)
                save_image_grid(sample_imgs, os.path.join(config.GEN_IMG_DIR, f"epoch_{epoch+1}.png"))
                
    torch.save(generator.state_dict(), os.path.join(config.CHECKPOINT_DIR, "generator_best.pth"))
    
    with open(os.path.join(config.LOG_DIR, "gan_history.json"), "w") as f:
        json.dump(history, f)
        
    plt.figure()
    plt.plot(history['d_loss'], label='D Loss')
    plt.plot(history['g_loss'], label='G Loss')
    plt.legend()
    plt.savefig(os.path.join(config.PLOT_DIR, "gan_loss.png"))
    plt.close()
    
    return history
