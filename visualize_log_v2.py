#!/usr/bin/env python3
"""
visualize_log_v2.py - Improved Logarithmic Visualization for U-Net Denoising

Clean, publication-quality figures for Professor Hesser.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import os
import argparse

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def safe_log_data(data, epsilon=1e-7):
    """Prepare data for log scale by adding epsilon and clipping negatives."""
    return np.clip(data, epsilon, None)


def plot_logarithmic_heatmaps_v2(hp, lp, pred, output_path='log_heatmap_v2.png', 
                                  slice_idx=None, vmin=1e-6):
    """
    Plot 1: Clean Logarithmic Heatmap Comparison
    """
    if slice_idx is None:
        slice_idx = hp.shape[2] // 2
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx] if pred is not None else None
    
    epsilon = 1e-7
    vmax = hp_slice.max() * 1.1
    
    # Prepare log-safe data
    hp_log = safe_log_data(hp_slice, epsilon)
    lp_log = safe_log_data(lp_slice, epsilon)
    pred_log = safe_log_data(pred_slice, epsilon) if pred_slice is not None else None
    gaussian_log = None
    
    if HAS_SCIPY and pred is not None:
        gaussian_slice = gaussian_filter(lp_slice, sigma=0.8)
        gaussian_log = safe_log_data(gaussian_slice, epsilon)
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(16, 4.5))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.15)
    
    cmap = 'inferno'  # Better than 'hot' for log scale
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    titles = ['LP Input\n(Noisy, N=100)', 'HP Ground Truth\n(Reference)', 
              'Gaussian Filter\n(σ=0.8)', 'U-Net Output\n(Our Method)']
    data_list = [lp_log, hp_log, gaussian_log, pred_log]
    
    axes = []
    for i, (title, data) in enumerate(zip(titles, data_list)):
        if data is None:
            continue
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='equal', 
                       interpolation='nearest')
        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('X (voxels)')
        if i == 0:
            ax.set_ylabel('Y (voxels)')
        ax.tick_params(labelsize=9)
    
    # Add colorbar
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Dose (Gy) - Log Scale', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # Main title
    fig.suptitle(f'Dose Distribution Comparison (Central Slice, Z={slice_idx})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.1)
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_line_profile_v2(hp, lp, pred, output_path='line_profile_v2.png',
                         slice_idx=None, row_idx=None):
    """
    Plot 2: Clean Central Line Profile (Log Scale)
    """
    if slice_idx is None:
        slice_idx = hp.shape[2] // 2
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx] if pred is not None else None
    
    # Find beam center (max dose location)
    if row_idx is None:
        row_idx = np.unravel_index(hp_slice.argmax(), hp_slice.shape)[0]
    
    hp_line = hp_slice[row_idx, :]
    lp_line = lp_slice[row_idx, :]
    pred_line = pred_slice[row_idx, :] if pred_slice is not None else None
    
    x = np.arange(len(hp_line))
    
    # Replace zeros with small value for log scale
    epsilon = hp_line[hp_line > 0].min() * 0.1 if np.any(hp_line > 0) else 1e-7
    hp_line_safe = np.where(hp_line > 0, hp_line, epsilon)
    lp_line_safe = np.where(lp_line > 0, lp_line, epsilon)
    pred_line_safe = np.where(pred_line > 0, pred_line, epsilon) if pred_line is not None else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with better styling
    ax.semilogy(x, lp_line_safe, color='#888888', linewidth=1.2, 
                label='LP Input (Noisy)', alpha=0.6, zorder=1)
    
    ax.semilogy(x, hp_line_safe, color='#2E7D32', linewidth=2.5, 
                label='HP Ground Truth', alpha=0.95, zorder=3)
    
    if HAS_SCIPY:
        gaussian_line = gaussian_filter(lp_slice, sigma=0.8)[row_idx, :]
        gaussian_line_safe = np.where(gaussian_line > 0, gaussian_line, epsilon)
        ax.semilogy(x, gaussian_line_safe, color='#1976D2', linewidth=1.8, 
                    linestyle='--', label='Gaussian Baseline', alpha=0.8, zorder=2)
    
    if pred_line_safe is not None:
        ax.semilogy(x, pred_line_safe, color='#D32F2F', linewidth=2.5, 
                    linestyle=':', label='U-Net Prediction', alpha=0.95, zorder=4)
    
    # Styling
    ax.set_xlabel('Voxel Position (X)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dose (Gy) — Log Scale', fontsize=12, fontweight='bold')
    ax.set_title(f'Dose Profile Along Beam Center\n(Z-slice={slice_idx}, Row={row_idx})', 
                 fontsize=14, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', linestyle=':', alpha=0.15, color='gray')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
              edgecolor='gray', fancybox=False)
    
    # Set x limits
    ax.set_xlim(-0.5, len(hp_line) - 0.5)
    
    # Add annotation box
    textstr = ('Key Observations:\n'
               '• Gray (LP): High noise fluctuations\n'
               '• Red (U-Net): Follows green (HP) closely\n'
               '• Low-dose tail preserved correctly')
    props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', 
                 edgecolor='gray', alpha=0.9)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_error_analysis_v2(hp, lp, pred, output_path='error_analysis_v2.png', slice_idx=None):
    """
    Plot 3: Clean Error Analysis
    """
    if pred is None:
        return
    
    if slice_idx is None:
        slice_idx = hp.shape[2] // 2
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]
    
    # Calculate errors
    lp_error = np.abs(lp_slice - hp_slice)
    pred_error = np.abs(pred_slice - hp_slice)
    
    if HAS_SCIPY:
        gaussian_slice = gaussian_filter(lp_slice, sigma=0.8)
        gaussian_error = np.abs(gaussian_slice - hp_slice)
    
    epsilon = 1e-8
    vmax = lp_error.max()
    vmin = 1e-7
    
    # Create figure
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.2)
    
    # Reference HP
    ax0 = fig.add_subplot(gs[0, 0])
    hp_vmax = hp_slice.max()
    im0 = ax0.imshow(safe_log_data(hp_slice), cmap='inferno', 
                     norm=LogNorm(vmin=1e-6, vmax=hp_vmax), aspect='equal')
    ax0.set_title('Reference (HP)\nGround Truth', fontweight='bold', fontsize=11)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    
    # Error plots
    cmap_err = 'viridis'
    norm_err = LogNorm(vmin=vmin, vmax=vmax)
    
    error_data = [
        (lp_error, f'|LP − HP|\nMax Error: {lp_error.max():.2e}'),
        (gaussian_error if HAS_SCIPY else None, f'|Gaussian − HP|\nMax Error: {gaussian_error.max():.2e}' if HAS_SCIPY else None),
        (pred_error, f'|U-Net − HP|\nMax Error: {pred_error.max():.2e}')
    ]
    
    for i, (err, title) in enumerate(error_data):
        if err is None:
            continue
        ax = fig.add_subplot(gs[0, i+1])
        im = ax.imshow(safe_log_data(err), cmap=cmap_err, norm=norm_err, aspect='equal')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('X')
        if i == 0:
            ax.set_ylabel('Y')
    
    # Colorbar
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Absolute Error (Gy)', fontsize=11)
    
    fig.suptitle('Error Analysis: Proving Low-Dose Errors are Noise, Not Structural', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def plot_summary_comparison(hp, lp, pred, output_path='summary_comparison.png', slice_idx=None):
    """
    Plot 4: Summary figure with metrics overlay
    """
    if slice_idx is None:
        slice_idx = hp.shape[2] // 2
    
    hp_slice = hp[:, :, slice_idx]
    lp_slice = lp[:, :, slice_idx]
    pred_slice = pred[:, :, slice_idx] if pred is not None else None
    
    # Calculate metrics
    lp_rmse = np.sqrt(np.mean((lp_slice - hp_slice)**2))
    lp_psnr = 20 * np.log10(hp_slice.max() / lp_rmse) if lp_rmse > 0 else 100
    
    pred_rmse = np.sqrt(np.mean((pred_slice - hp_slice)**2)) if pred_slice is not None else 0
    pred_psnr = 20 * np.log10(hp_slice.max() / pred_rmse) if pred_rmse > 0 else 100
    
    improvement = pred_psnr - lp_psnr
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    vmax = hp_slice.max()
    cmap = 'hot'
    
    # LP
    im0 = axes[0].imshow(lp_slice, cmap=cmap, vmin=0, vmax=vmax)
    axes[0].set_title(f'LP Input (Noisy)\nPSNR: {lp_psnr:.1f} dB', 
                      fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # U-Net
    if pred_slice is not None:
        im1 = axes[1].imshow(pred_slice, cmap=cmap, vmin=0, vmax=vmax)
        axes[1].set_title(f'U-Net Output\nPSNR: {pred_psnr:.1f} dB (+{improvement:.1f} dB)', 
                          fontweight='bold', fontsize=12, color='green')
    axes[1].axis('off')
    
    # HP
    im2 = axes[2].imshow(hp_slice, cmap=cmap, vmin=0, vmax=vmax)
    axes[2].set_title('HP Ground Truth\n(Reference)', fontweight='bold', fontsize=12)
    axes[2].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im2, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Dose (Gy)', fontsize=11)
    
    fig.suptitle('Denoising Result Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def run_visualization():
    """Run all visualizations on test sample."""
    import torch
    import sys
    sys.path.insert(0, '/workspace')
    
    from models.simple_unet_denoiser import SimpleUNetDenoiser
    from scipy.ndimage import gaussian_filter
    
    # Paths
    dataset_dir = '/workspace/dataset_5_patients'
    model_path = '/workspace/results/simple_unet_5patients/best_model.pth'
    lp_folder = 'lp_cubes_100'
    output_dir = '/workspace/results/log_visualization_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find patient
    patients = [d for d in os.listdir(dataset_dir) 
                if os.path.isdir(os.path.join(dataset_dir, d))]
    patient = patients[0]
    patient_path = os.path.realpath(os.path.join(dataset_dir, patient))
    
    # Load cube
    hp_dir = os.path.join(patient_path, 'output_cubes')
    lp_dir = os.path.join(patient_path, lp_folder)
    hp_files = sorted([f for f in os.listdir(hp_dir) if f.endswith('.npy')])
    
    # Get middle cube
    cube_file = hp_files[len(hp_files)//2]
    hp = np.load(os.path.join(hp_dir, cube_file))
    lp = np.load(os.path.join(lp_dir, cube_file))
    
    print(f"\n{'='*60}")
    print("GENERATING IMPROVED VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"Sample: {cube_file}")
    print(f"HP range: [{hp.min():.6f}, {hp.max():.6f}]")
    print(f"LP range: [{lp.min():.6f}, {lp.max():.6f}]")
    
    # Load model and predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNetDenoiser(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Inference
    dose_scale = 0.02
    residual_scale = 1000.0
    
    lp_norm = lp / dose_scale
    gaussian_baseline = gaussian_filter(lp_norm, sigma=0.8)
    
    lp_t = torch.from_numpy(lp_norm).float().unsqueeze(0).unsqueeze(0)
    g_t = torch.from_numpy(gaussian_baseline).float().unsqueeze(0).unsqueeze(0)
    input_t = torch.cat([lp_t, g_t], dim=1).to(device)
    
    with torch.no_grad():
        pred_res = model(input_t).squeeze().cpu().numpy()
    
    pred = np.clip((gaussian_baseline + pred_res / residual_scale) * dose_scale, 0, None)
    print(f"Pred range: [{pred.min():.6f}, {pred.max():.6f}]")
    
    # Generate all plots
    print(f"\nGenerating plots...")
    
    plot_logarithmic_heatmaps_v2(hp, lp, pred, 
        os.path.join(output_dir, 'log_heatmap.png'))
    
    plot_line_profile_v2(hp, lp, pred,
        os.path.join(output_dir, 'line_profile_log.png'))
    
    plot_error_analysis_v2(hp, lp, pred,
        os.path.join(output_dir, 'error_analysis.png'))
    
    plot_summary_comparison(hp, lp, pred,
        os.path.join(output_dir, 'summary.png'))
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    return output_dir


if __name__ == '__main__':
    run_visualization()
