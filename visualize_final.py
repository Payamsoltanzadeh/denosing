#!/usr/bin/env python3
"""
visualize_final.py - Professional publication-quality visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.gridspec import GridSpec
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 150

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def find_best_slice(volume, axis=2):
    """Find slice with maximum dose."""
    if axis == 2:
        sums = [volume[:, :, i].sum() for i in range(volume.shape[2])]
    return np.argmax(sums)


def plot_publication_heatmap(hp, lp, pred, output_path='fig_heatmap.png'):
    """
    Publication-quality heatmap with proper log scale
    """
    # Find best slice (maximum dose)
    slice_idx = find_best_slice(hp, axis=2)
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx] if pred is not None else None
    
    print(f"\nUsing slice {slice_idx} (max dose slice)")
    print(f"  HP slice range: [{hp_slice.min():.6f}, {hp_slice.max():.6f}]")
    
    # Calculate gaussian baseline
    if HAS_SCIPY:
        lp_norm = lp_slice / 0.02
        gaussian_slice = gaussian_filter(lp_norm, sigma=0.8) * 0.02
    else:
        gaussian_slice = None
    
    # Setup for log scale
    epsilon = 1e-7
    vmin = hp_slice[hp_slice > 0].min() * 0.5 if np.any(hp_slice > 0) else 1e-7
    vmax = hp_slice.max()
    
    print(f"  Log scale: vmin={vmin:.2e}, vmax={vmax:.2e}")
    
    # Create figure
    fig = plt.figure(figsize=(15, 4))
    
    n_cols = 4 if gaussian_slice is not None else 3
    gs = GridSpec(1, n_cols + 1, width_ratios=[1]*n_cols + [0.05], 
                  wspace=0.25, left=0.05, right=0.95)
    
    # Color settings
    cmap = 'hot'
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # Prepare data
    data_list = [
        (lp_slice + epsilon, 'LP Input\n(N=100 photons, noisy)'),
        (gaussian_slice + epsilon if gaussian_slice is not None else None, 'Gaussian Filter\n(σ=0.8, baseline)'),
        (pred_slice + epsilon if pred_slice is not None else None, 'U-Net Prediction\n(Our method)'),
        (hp_slice + epsilon, 'Ground Truth\n(Reference, N=10⁶)')
    ]
    
    axes = []
    im = None
    col_idx = 0
    
    for data, title in data_list:
        if data is None:
            continue
            
        ax = fig.add_subplot(gs[0, col_idx])
        axes.append(ax)
        
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='equal', 
                      interpolation='bilinear', origin='lower')
        
        # Title color based on content
        title_color = 'darkgreen' if 'U-Net' in title else 'black'
        ax.set_title(title, fontweight='bold', color=title_color, pad=8)
        
        ax.set_xlabel('X (voxels)', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Y (voxels)', fontsize=9)
        else:
            ax.set_yticks([])
        
        col_idx += 1
    
    # Colorbar
    cax = fig.add_subplot(gs[0, -1])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Dose (Gy)\nLog Scale', fontsize=9, rotation=90, labelpad=10)
    
    # Main title
    fig.suptitle(f'MC Dose Denoising Results (Z-slice {slice_idx}, max dose region)', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_publication_lineprofile(hp, lp, pred, output_path='fig_lineprofile.png'):
    """
    Publication-quality line profile with proper scaling
    """
    # Find best slice
    slice_idx = find_best_slice(hp, axis=2)
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx] if pred is not None else None
    
    # Find beam center (max dose location)
    max_pos = np.unravel_index(hp_slice.argmax(), hp_slice.shape)
    row_idx = max_pos[0]
    
    print(f"\nLine profile through row {row_idx} (beam center)")
    
    # Extract profiles
    hp_line = hp_slice[row_idx, :]
    lp_line = lp_slice[row_idx, :]
    pred_line = pred_slice[row_idx, :] if pred_slice is not None else None
    
    # Gaussian baseline
    if HAS_SCIPY:
        lp_norm = lp_slice / 0.02
        gaussian_slice = gaussian_filter(lp_norm, sigma=0.8) * 0.02
        gaussian_line = gaussian_slice[row_idx, :]
    else:
        gaussian_line = None
    
    x = np.arange(len(hp_line))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), 
                                     gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.35})
    
    # ---- Top panel: Log scale ----
    epsilon = hp_line[hp_line > 0].min() * 0.01 if np.any(hp_line > 0) else 1e-8
    
    # Plot with proper styling
    ax1.plot(x, lp_line + epsilon, color='#999999', linewidth=1.5, 
            label='LP Input (noisy)', alpha=0.6, zorder=1)
    
    if gaussian_line is not None:
        ax1.plot(x, gaussian_line + epsilon, color='#2196F3', linewidth=2, 
                linestyle='--', label='Gaussian Filter', alpha=0.8, zorder=2)
    
    if pred_line is not None:
        ax1.plot(x, pred_line + epsilon, color='#F44336', linewidth=2.5, 
                linestyle=':', label='U-Net Prediction', alpha=0.95, zorder=4)
    
    ax1.plot(x, hp_line + epsilon, color='#4CAF50', linewidth=2.5, 
            label='Ground Truth (HP)', alpha=0.95, zorder=3)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Dose (Gy) — Log Scale', fontweight='bold', fontsize=11)
    ax1.set_title(f'Dose Profile Through Beam Center (Z-slice {slice_idx}, Row {row_idx})', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
    ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.5, len(hp_line) - 0.5)
    
    # ---- Bottom panel: Linear scale (zoomed) ----
    # Focus on high-dose region
    high_dose_mask = hp_line > hp_line.max() * 0.1
    if np.any(high_dose_mask):
        indices = np.where(high_dose_mask)[0]
        x_min = max(0, indices.min() - 2)
        x_max = min(len(hp_line), indices.max() + 3)
        
        ax2.plot(x, lp_line, color='#999999', linewidth=1.5, 
                label='LP Input', alpha=0.6)
        
        if gaussian_line is not None:
            ax2.plot(x, gaussian_line, color='#2196F3', linewidth=2, 
                    linestyle='--', label='Gaussian', alpha=0.8)
        
        if pred_line is not None:
            ax2.plot(x, pred_line, color='#F44336', linewidth=2.5, 
                    linestyle=':', label='U-Net', alpha=0.95)
        
        ax2.plot(x, hp_line, color='#4CAF50', linewidth=2.5, 
                label='Ground Truth', alpha=0.95)
        
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(0, hp_line.max() * 1.1)
        ax2.set_xlabel('Voxel Position (X)', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Dose (Gy) — Linear Scale', fontweight='bold', fontsize=11)
        ax2.set_title('High-Dose Region (Zoomed)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, ncol=4)
        ax2.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_publication_metrics(hp, lp, pred, output_path='fig_metrics.png'):
    """
    Metrics comparison figure
    """
    # Calculate metrics
    lp_rmse = np.sqrt(np.mean((lp - hp)**2))
    lp_mae = np.mean(np.abs(lp - hp))
    lp_psnr = 20 * np.log10(hp.max() / lp_rmse) if lp_rmse > 0 else 100
    
    pred_rmse = np.sqrt(np.mean((pred - hp)**2)) if pred is not None else 0
    pred_mae = np.mean(np.abs(pred - hp)) if pred is not None else 0
    pred_psnr = 20 * np.log10(hp.max() / pred_rmse) if pred_rmse > 0 else 100
    
    # Gaussian baseline
    if HAS_SCIPY:
        lp_norm = lp / 0.02
        gaussian = gaussian_filter(lp_norm, sigma=0.8) * 0.02
        gauss_rmse = np.sqrt(np.mean((gaussian - hp)**2))
        gauss_mae = np.mean(np.abs(gaussian - hp))
        gauss_psnr = 20 * np.log10(hp.max() / gauss_rmse) if gauss_rmse > 0 else 100
    else:
        gauss_rmse = gauss_mae = gauss_psnr = 0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    methods = ['LP\nInput', 'Gaussian\nBaseline', 'U-Net\n(Ours)']
    
    # RMSE
    rmse_vals = [lp_rmse, gauss_rmse, pred_rmse]
    colors = ['#FF9800', '#2196F3', '#4CAF50']
    bars1 = axes[0].bar(methods, rmse_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('RMSE (Gy)', fontweight='bold')
    axes[0].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    # Add values on bars
    for bar, val in zip(bars1, rmse_vals):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # PSNR
    psnr_vals = [lp_psnr, gauss_psnr, pred_psnr]
    bars2 = axes[1].bar(methods, psnr_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('PSNR (dB)', fontweight='bold')
    axes[1].set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=lp_psnr, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for bar, val in zip(bars2, psnr_vals):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} dB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Improvement
    rmse_improve = [(lp_rmse - pred_rmse) / lp_rmse * 100,
                    (gauss_rmse - pred_rmse) / gauss_rmse * 100] if pred is not None else [0, 0]
    psnr_improve = [pred_psnr - lp_psnr, pred_psnr - gauss_psnr] if pred is not None else [0, 0]
    
    x_pos = np.arange(2)
    width = 0.35
    
    bars3a = axes[2].bar(x_pos - width/2, rmse_improve, width, 
                         label='RMSE Reduction (%)', color='#E91E63', alpha=0.8, edgecolor='black')
    bars3b = axes[2].bar(x_pos + width/2, psnr_improve, width,
                         label='PSNR Gain (dB)', color='#9C27B0', alpha=0.8, edgecolor='black')
    
    axes[2].set_ylabel('Improvement', fontweight='bold')
    axes[2].set_title('U-Net Improvement vs Baselines', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(['vs LP', 'vs Gaussian'])
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=0, color='black', linewidth=1)
    
    # Add values
    for bars in [bars3a, bars3b]:
        for bar in bars:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=8, fontweight='bold')
    
    fig.suptitle('Quantitative Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    
    # Print summary
    print(f"\n📊 METRICS SUMMARY:")
    print(f"  LP Input:     RMSE={lp_rmse:.2e}, PSNR={lp_psnr:.2f} dB")
    if HAS_SCIPY:
        print(f"  Gaussian:     RMSE={gauss_rmse:.2e}, PSNR={gauss_psnr:.2f} dB")
    if pred is not None:
        print(f"  U-Net:        RMSE={pred_rmse:.2e}, PSNR={pred_psnr:.2f} dB")
        print(f"  Improvement:  {rmse_improve[0]:.1f}% RMSE reduction, +{psnr_improve[0]:.1f} dB PSNR")


def run_final_visualization():
    """Generate all publication-quality figures"""
    import torch
    import sys
    sys.path.insert(0, '/workspace')
    
    from models.simple_unet_denoiser import SimpleUNetDenoiser
    
    # Paths
    dataset_dir = '/workspace/dataset_5_patients'
    model_path = '/workspace/results/simple_unet_5patients/best_model.pth'
    output_dir = '/workspace/results/publication_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample
    patients = [d for d in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, d))]
    patient_path = os.path.realpath(os.path.join(dataset_dir, patients[0]))
    
    hp_dir = os.path.join(patient_path, 'output_cubes')
    lp_dir = os.path.join(patient_path, 'lp_cubes_100')
    hp_files = sorted([f for f in os.listdir(hp_dir) if f.endswith('.npy')])
    
    # Load middle cube
    cube_file = hp_files[len(hp_files)//2]
    hp = np.load(os.path.join(hp_dir, cube_file))
    lp = np.load(os.path.join(lp_dir, cube_file))
    
    print(f"\n{'='*70}")
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print(f"{'='*70}")
    print(f"Sample: {cube_file}")
    print(f"Volume shape: {hp.shape}")
    print(f"HP range: [{hp.min():.6f}, {hp.max():.6f}]")
    print(f"LP range: [{lp.min():.6f}, {lp.max():.6f}]")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNetDenoiser(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Predict
    lp_norm = lp / 0.02
    gaussian_baseline = gaussian_filter(lp_norm, sigma=0.8)
    
    lp_t = torch.from_numpy(lp_norm).float().unsqueeze(0).unsqueeze(0)
    g_t = torch.from_numpy(gaussian_baseline).float().unsqueeze(0).unsqueeze(0)
    input_t = torch.cat([lp_t, g_t], dim=1).to(device)
    
    with torch.no_grad():
        pred_res = model(input_t).squeeze().cpu().numpy()
    
    pred = np.clip((gaussian_baseline + pred_res / 1000.0) * 0.02, 0, None)
    print(f"Pred range: [{pred.min():.6f}, {pred.max():.6f}]")
    
    # Generate figures
    print(f"\n📊 Generating figures...")
    
    plot_publication_heatmap(hp, lp, pred, 
        os.path.join(output_dir, 'Figure1_Heatmap.png'))
    
    plot_publication_lineprofile(hp, lp, pred,
        os.path.join(output_dir, 'Figure2_LineProfile.png'))
    
    plot_publication_metrics(hp, lp, pred,
        os.path.join(output_dir, 'Figure3_Metrics.png'))
    
    print(f"\n{'='*70}")
    print(f"✅ ALL FIGURES SAVED TO: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_final_visualization()
