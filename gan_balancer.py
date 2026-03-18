import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import GAN_CONFIG, OUTPUT_DIR
import matplotlib.pyplot as plt


# GAN MODEL DEFINITIONS

class GANGenerator(nn.Module):

    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),       nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, feature_dim), nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)


class GANDiscriminator(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256),         nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 1),           nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


# GAN TRAINING FUNCTION

def run_gan_balancing(X_minority, n_to_generate, 
                      noise_dim=GAN_CONFIG['noise_dim'],
                      epochs=GAN_CONFIG['epochs'],
                      batch_size=GAN_CONFIG['batch_size'],
                      lr=GAN_CONFIG['learning_rate'],
                      verbose=True):
  
    feature_dim = X_minority.shape[1]
    device = torch.device('cpu')
    
    # Initialize networks
    G = GANGenerator(noise_dim, feature_dim).to(device)
    D = GANDiscriminator(feature_dim).to(device)
    
    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    
    # Data loader
    X_tensor = torch.from_numpy(X_minority.astype(np.float32)).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    if verbose:
        print(f"\n  GAN training: feature_dim={feature_dim}, "
              f"minority={len(X_minority):,}, to_generate={n_to_generate:,}")
    
    # Training loop
    for epoch in range(epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        
        for (real_batch,) in loader:
            bs = real_batch.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)
            
            # Train Discriminator
            opt_D.zero_grad()
            d_real = D(real_batch)
            loss_dr = bce(d_real, real_lbl)
            
            z = torch.randn(bs, noise_dim, device=device)
            fake = G(z).detach()
            d_fake = D(fake)
            loss_df = bce(d_fake, fake_lbl)
            
            loss_D = (loss_dr + loss_df) / 2
            loss_D.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(bs, noise_dim, device=device)
            fake = G(z)
            g_pred = D(fake)
            loss_G = bce(g_pred, real_lbl)
            loss_G.backward()
            opt_G.step()
            
            d_loss_epoch += loss_D.item()
            g_loss_epoch += loss_G.item()
        
        # Progress logging
        if verbose and (epoch + 1) % 100 == 0:
            avg_d = d_loss_epoch / max(len(loader), 1)
            avg_g = g_loss_epoch / max(len(loader), 1)
            print(f"    Epoch [{epoch+1:4d}/{epochs}]  "
                  f"D_loss={avg_d:.4f}  G_loss={avg_g:.4f}")
    
    # Generate synthetic samples
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_to_generate, noise_dim, device=device)
        synthetic = G(z).cpu().numpy()
    
    return synthetic


# BALANCING PIPELINE

def balance_dataset_with_gan(X, y, target_ratio=1.0, verbose=True):
    
    # Analyze imbalance
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    n_to_gen = int(n_majority * target_ratio) - n_minority
    
    if verbose:
        print(f"\n── Analyzing class imbalance ──")
        print(f"  Majority (Class 0): {n_majority:,}")
        print(f"  Minority (Class 1): {n_minority:,}")
        print(f"  Samples to generate: {n_to_gen:,}")
    
    if n_to_gen <= 0:
        print("  Dataset already balanced or majority is minority. Skipping GAN.")
        return X, y
    
    # Generate synthetic samples
    synthetic_samples = run_gan_balancing(
        X_minority, n_to_generate=n_to_gen, 
        verbose=verbose
    )
    
    # Assemble balanced dataset
    X_balanced = np.vstack([X, synthetic_samples])
    y_balanced = np.concatenate([y, np.ones(n_to_gen, dtype=np.int64)])
    
    if verbose:
        print(f"\n   After GAN balancing:")
        print(f"     Total samples: {len(X_balanced):,}")
        print(f"     Class 0      : {(y_balanced==0).sum():,}")
        print(f"     Class 1      : {(y_balanced==1).sum():,}")
    
    return X_balanced, y_balanced


def plot_balancing_results(y_before, y_after, save_path=None):

    neg_before = int((y_before == 0).sum())
    pos_before = int((y_before == 1).sum())
    neg_after = int((y_after == 0).sum())
    pos_after = int((y_after == 1).sum())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (neg, pos, title) in zip(axes, [
        (neg_before, pos_before, 'BEFORE GAN\n(Imbalanced)'),
        (neg_after, pos_after, 'AFTER GAN\n(Balanced, 1:1 ratio)'),
    ]):
        bars = ax.bar(['Class 0\n(No)', 'Class 1\n(Yes)'],
                      [neg, pos], color=['#d62728', '#1f77b4'],
                      edgecolor='white', width=0.45)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(neg, pos) * 1.18)
        for bar, v in zip(bars, [neg, pos]):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max(neg,pos)*0.01,
                   f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('BindingDB-Kd: Class Distribution Before vs After GAN Balancing',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.savefig(f"{OUTPUT_DIR}/Kd_class_distribution_before_after_GAN.png",
                   dpi=150, bbox_inches='tight')
        plt.close()