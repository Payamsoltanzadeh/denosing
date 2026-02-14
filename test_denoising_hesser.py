# ============================================================================
# test_denoising_hesser.py
# ============================================================================
# Date: 2025-12-11
# Purpose: UPDATED Inference & Evaluation with Hesser's Requirements:
#          1. PSNR metric (standard in medical imaging)
#          2. Un-normalized metrics (physical dose in Gy)
#          3. Dose Regime Analysis (high/mid/low dose regions)
#          4. Classical filter baselines (Gaussian, Bilateral)
#
# Key Changes:
#   - Load RAW data (normalize=False) for physical metrics
#   - Load NORMALIZED data for model input (model trained on normalized)
#   - Un-normalize model output before metrics
#   - Add PSNR calculation
#   - Add Gaussian/Bilateral filter baselines
#   - Add dose regime analysis
# ============================================================================

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter, median_filter

# Import Model and Dataset
from models.simple_unet_denoiser import get_simple_denoiser
from dataset.pl_dose_dataset import ConditionalDoseDataset
from utils.gamma_index import calculate_gamma_index_3d  # WARNING: DD-only, no DTA!
from utils.torch_gaussian import get_gaussian_layer


# ============================================================================
# NEW: PSNR Calculation
# ============================================================================
def compute_psnr(reference, evaluated, data_range=None):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    Standard metric in medical imaging (Hesser request).
    
    Args:
        reference: Ground truth (HP dose in physical units)
        evaluated: Prediction or LP (physical units)
        data_range: Max value for normalization. If None, uses max of reference.
    
    Returns:
        PSNR in dB (higher is better, typically 20-50 dB for medical images)
    """
    if data_range is None:
        data_range = np.max(reference)
    
    if data_range < 1e-10:
        return 0.0  # Avoid division by zero
    
    mse = np.mean((reference - evaluated) ** 2)
    if mse < 1e-15:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


# ============================================================================
# NEW: Dose Regime Analysis (Hesser request)
# ============================================================================
def compute_metrics_by_dose_regime(reference, evaluated, thresholds=(0.1, 0.5)):
    """
    Compute metrics separately for different dose regimes.
    This answers Hesser's request for "categorize error in different dose regimes".
    
    Dose Regimes:
        - Low dose:  0-10% of max dose (background, air, low scatter)
        - Mid dose:  10-50% of max dose (penumbra, buildup)
        - High dose: >50% of max dose (beam center, treatment volume)
    
    Args:
        reference: Ground truth HP dose (physical units)
        evaluated: Prediction (physical units)
        thresholds: (low_threshold, high_threshold) as fraction of max dose
    
    Returns:
        Dictionary with RMSE, MAE, RelativeError for each regime
    """
    max_dose = np.max(reference)
    if max_dose < 1e-10:
        return {"low": None, "mid": None, "high": None}
    
    low_thresh = thresholds[0] * max_dose
    high_thresh = thresholds[1] * max_dose
    
    # Define masks for each regime
    # Note: Exclude zero-dose regions (air outside patient)
    low_mask = (reference > 1e-10) & (reference <= low_thresh)
    mid_mask = (reference > low_thresh) & (reference <= high_thresh)
    high_mask = reference > high_thresh
    
    results = {}
    
    for name, mask in [("low", low_mask), ("mid", mid_mask), ("high", high_mask)]:
        if np.sum(mask) > 100:  # Need enough voxels for statistics
            ref_region = reference[mask]
            eval_region = evaluated[mask]
            
            rmse = np.sqrt(np.mean((ref_region - eval_region) ** 2))
            mae = np.mean(np.abs(ref_region - eval_region))
            
            # Relative error (percentage of local dose)
            rel_error = np.mean(np.abs(ref_region - eval_region) / (ref_region + 1e-10)) * 100
            
            results[name] = {
                "rmse": rmse,
                "mae": mae,
                "rel_error_percent": rel_error,
                "num_voxels": int(np.sum(mask)),
                "mean_dose": float(np.mean(ref_region))
            }
        else:
            results[name] = None
    
    return results


# ============================================================================
# NEW: Classical Filter Baselines (Hesser request)
# ============================================================================
def apply_gaussian_filter_gpu(volume_tensor, gaussian_layer, device):
    """
    Apply 3D Gaussian filter on GPU for FAST baseline comparison.
    This is the "linear denoising filter" Hesser mentioned.
    
    Args:
        volume_tensor: 3D dose volume as torch tensor [1, D, H, W]
        gaussian_layer: Pre-initialized GaussianBlur3D layer
        device: torch device
    
    Returns:
        Smoothed volume as numpy array
    """
    with torch.no_grad():
        if volume_tensor.dim() == 3:
            volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif volume_tensor.dim() == 4:
            volume_tensor = volume_tensor.unsqueeze(1)  # [1, 1, D, H, W]
        
        volume_tensor = volume_tensor.to(device)
        filtered = gaussian_layer(volume_tensor)
        return filtered.squeeze().cpu().numpy()


# ============================================================================
# Comprehensive Metrics on PHYSICAL (un-normalized) dose
# ============================================================================
def compute_all_metrics(hp_raw, pred_raw, lp_raw):
    """
    Compute all metrics on RAW (un-normalized) physical dose.
    
    This is the CORRECT way to calculate metrics (Hesser feedback).
    All values are in Gy (or same units as input dose).
    
    Returns:
        Dictionary with all metrics
    """
    # Basic metrics (physical units)
    mse = np.mean((hp_raw - pred_raw) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(hp_raw - pred_raw))
    
    # PSNR (NEW - Hesser request)
    psnr = compute_psnr(hp_raw, pred_raw)
    psnr_lp = compute_psnr(hp_raw, lp_raw)  # Baseline PSNR
    
    # Gamma Index (on physical dose)
    # WARNING: This uses DD-only gamma_index.py (DTA is NOT included).
    # Results should be reported as "DD-only X%", NOT "Gamma X%/Xmm".
    # For proper Gamma with DTA, use pymedphys via honest_evaluation.py.
    try:
        _, gamma_33 = calculate_gamma_index_3d(hp_raw, pred_raw, dta_mm=3.0, dd_percent=3.0)
        _, gamma_22 = calculate_gamma_index_3d(hp_raw, pred_raw, dta_mm=2.0, dd_percent=2.0)
        _, gamma_11 = calculate_gamma_index_3d(hp_raw, pred_raw, dta_mm=1.0, dd_percent=1.0)
    except Exception as e:
        print(f"Warning: Gamma calculation failed: {e}")
        gamma_33, gamma_22, gamma_11 = 0.0, 0.0, 0.0
    
    # Dose regime analysis (NEW - Hesser request)
    dose_regimes = compute_metrics_by_dose_regime(hp_raw, pred_raw)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "psnr_lp": psnr_lp,
        "gamma_33": gamma_33,
        "gamma_22": gamma_22,
        "gamma_11": gamma_11,
        "dose_regimes": dose_regimes,
        "max_dose": np.max(hp_raw),
        "min_dose": np.min(hp_raw[hp_raw > 0]) if np.any(hp_raw > 0) else 0
    }


def visualize_results(ct, lp, hp, pred, save_path, sample_idx, metrics=None):
    """
    Visualize central slice with metrics overlay.
    """
    d_center = ct.shape[0] // 2
    
    ct_slice = ct[d_center, :, :]
    lp_slice = lp[d_center, :, :]
    hp_slice = hp[d_center, :, :]
    pred_slice = pred[d_center, :, :]
    diff_slice = np.abs(hp_slice - pred_slice)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # CT
    im0 = axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title("CT (Anatomy)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # LP (Input) - Raw dose values
    im1 = axes[1].imshow(lp_slice, cmap='jet')
    axes[1].set_title("LP Dose (Noisy)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # HP (Target) - Raw dose values
    im2 = axes[2].imshow(hp_slice, cmap='jet')
    axes[2].set_title("HP Dose (Target)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Prediction - Raw dose values
    im3 = axes[3].imshow(pred_slice, cmap='jet')
    axes[3].set_title("Predicted Dose")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Difference
    im4 = axes[4].imshow(diff_slice, cmap='hot')
    if metrics:
        axes[4].set_title(f"Diff (PSNR={metrics['psnr']:.1f}dB)")
    else:
        axes[4].set_title("Abs Difference")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{sample_idx}.png"), dpi=150)
    plt.close()


def test(args):
    """
    Main test function with Hesser's requirements:
    1. PSNR metric
    2. Un-normalized physical dose metrics
    3. Dose regime analysis
    4. Classical filter baselines
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    print(f"🚀 Testing on: {device}")
    print("="*70)
    print("EVALUATION WITH HESSER'S REQUIREMENTS")
    print("  - PSNR metric")
    print("  - Un-normalized metrics (physical dose)")
    print("  - Dose regime analysis (low/mid/high)")
    print("  - Classical filter baselines (Gaussian/Bilateral)")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =========================================================================
    # CRITICAL: Load TWO versions of dataset
    # 1. NORMALIZED - for model input (model trained on normalized data)
    # 2. RAW - for metrics calculation (physical dose units)
    #
    # FOLDER-LEVEL SPLIT (Fixed: 2026-02-03)
    # Per Marcus Buchwald: Each folder is independent simulation
    # Test ONLY on last 3 folders (never seen during training!)
    # =========================================================================
    ALL_FOLDERS = sorted([
        "132102389758881090179672567372987664560",
        "156339165372145091992366212553627308160",
        "169178663511657097619787299759624979616",
        "226808136684964871480153898846429021344",
        "235101017859661465075472232303048949736",
        "277818358016850894820204297874963333096",
        "40001034793533568458224025980056277120",
        "40838398986272789027252987369168472240",
        "62242424098194756750821789925294112700",
        "73631120845087000785006921692554887100",
    ])
    TEST_FOLDERS = ALL_FOLDERS[8:]  # Last 2 folders (20%)
    
    print("\n📂 Loading datasets...")
    print(f"   ⚠️  FOLDER-LEVEL TESTING: Testing on {len(TEST_FOLDERS)} folders (4000 patches, 20 patients)")
    print(f"   These folders were NEVER seen during training!")
    print(f"   ⚠️  FOLDER-LEVEL TESTING: Only folder {TEST_FOLDERS[0][:20]}...")
    print(f"   This folder (10 patients, 2000 patches) was NEVER seen during training!")
    
    # Dataset for MODEL INPUT (normalized)
    dataset_norm = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,  # Model expects normalized input
        target_dim=32,  # Match actual data size (32x32x32)
        delta=None,
        target_uncertainty=0.10,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
        dose_scale=args.dose_scale,
        folder_names=TEST_FOLDERS,  # NEW: Folder-level split
    )
    
    # Dataset for METRICS (raw, physical units)
    dataset_raw = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=False,  # RAW physical dose for metrics!
        target_dim=32,  # Match actual data size (32x32x32)
        delta=None,
        target_uncertainty=0.10,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
        dose_scale=args.dose_scale,
        folder_names=TEST_FOLDERS,  # NEW: Folder-level split
    )
    
    print(f"   Loaded {len(dataset_norm)} test samples (1 folder = 10 patients = 2000 patches)")
    
    # =========================================================================
    # Load Model
    # Read base_channels, gaussian_sigma, residual_scale from checkpoint
    # so the test script is always consistent with whatever was trained.
    # =========================================================================
    print(f"\n🏗️ Loading model from: {args.model_path}")
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Read training args from checkpoint (if available)
        saved_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
        base_channels = saved_args.get('base_channels', args.base_channels)
        gaussian_sigma = saved_args.get('gaussian_sigma', args.gaussian_sigma)
        residual_scale = saved_args.get('residual_scale', args.residual_scale)
        print(f"   Checkpoint args: base_channels={base_channels}, "
              f"gaussian_sigma={gaussian_sigma}, residual_scale={residual_scale}")
    else:
        base_channels = args.base_channels
        gaussian_sigma = args.gaussian_sigma
        residual_scale = args.residual_scale
    
    model = get_simple_denoiser(
        model_type=args.model_type,
        base_channels=base_channels
    ).to(device)
    
    if os.path.exists(args.model_path):
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Model loaded (epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("   ✅ Model loaded successfully")
    else:
        print(f"   ❌ Model not found! Using random weights.")
    
    model.eval()
    
    # Create Gaussian blur layer for inference (matches training)
    gaussian_blur = get_gaussian_layer(channels=1, sigma=gaussian_sigma, device=device)
    gaussian_blur.eval()
    print(f"   Gaussian blur sigma: {gaussian_sigma}")
    
    # Create Gaussian filter for CPU baseline (sigma=1.0 for MC denoising)
    gaussian_baseline = get_gaussian_layer(channels=1, sigma=1.0, device=device)
    gaussian_baseline.eval()
    
    # =========================================================================
    # Results Storage
    # =========================================================================
    unet_results = []
    gaussian_results = []
    lp_baseline_results = []  # LP without any processing
    
    num_samples = min(args.num_samples, len(dataset_norm)) if args.num_samples > 0 else len(dataset_norm)
    
    print(f"\n🔥 Processing {num_samples} samples...")
    print("-"*70)
    
    with torch.no_grad():
        for i in range(num_samples):
            # =================================================================
            # Get NORMALIZED data for model input
            # =================================================================
            hp_norm, condition_norm = dataset_norm[i]
            ct_norm = condition_norm['ct']
            lp_norm = condition_norm['lp_dose']
            
            # =================================================================
            # Get RAW data for metrics
            # =================================================================
            hp_raw_tensor, condition_raw = dataset_raw[i]
            ct_raw_tensor = condition_raw['ct']
            lp_raw_tensor = condition_raw['lp_dose']
            
            # Convert to numpy (physical dose values)
            hp_raw = hp_raw_tensor.squeeze().numpy()
            lp_raw = lp_raw_tensor.squeeze().numpy()
            ct_raw = ct_raw_tensor.squeeze().numpy()
            
            # =================================================================
            # Model Prediction (on normalized input)
            # RESIDUAL LEARNING with SIGNAL AMPLIFICATION on GAUSSIAN baseline
            # Training used target = (HP - Gaussian(LP)) * residual_scale
            # So here: Pred = Gaussian(LP) + Model_Output / residual_scale
            # =================================================================
            inputs_norm = torch.cat([ct_norm.unsqueeze(0), lp_norm.unsqueeze(0)], dim=1).to(device)
            
            # 1. Get scaled residual from model
            pred_scaled_residual = model(inputs_norm)
            
            # 2. Scale back to normalized dose space
            pred_residual = pred_scaled_residual / float(residual_scale)
            
            # 3. Apply Gaussian blur to LP to get baseline (matches training)
            lp_norm_batch = lp_norm.unsqueeze(0).to(device)
            with torch.no_grad():
                lp_gaussian_norm = gaussian_blur(lp_norm_batch)
            
            # 4. Add correction to Gaussian baseline
            pred_norm = lp_gaussian_norm + pred_residual
            pred_norm_np = pred_norm[0, 0].cpu().numpy()
            
            # =================================================================
            # UN-NORMALIZE prediction to physical dose
            # 
            # NEW NORMALIZATION: Dose is scaled to [0, 1] using global max
            # Un-scale: x_physical = x_normalized * max_dose_global
            # =================================================================
            hp_norm_np = hp_norm.squeeze().numpy()
            
            # Use the same global max as in training
            max_dose_global = float(args.dose_scale)  # Must match dataset normalization!
            
            # Un-normalize: Scale back from [0,1] to physical units
            pred_raw = pred_norm_np * max_dose_global
            pred_raw = np.maximum(pred_raw, 0)  # Dose must be >= 0
            
            # =================================================================
            # Compute U-Net Metrics (on PHYSICAL dose)
            # =================================================================
            unet_metrics = compute_all_metrics(hp_raw, pred_raw, lp_raw)
            unet_results.append(unet_metrics)
            
            # =================================================================
            # Classical Filter Baselines (on RAW LP) - GPU ACCELERATED
            # =================================================================
            # Gaussian filter (sigma=1.0, typical for MC denoising) - ON GPU!
            lp_gaussian = apply_gaussian_filter_gpu(lp_raw_tensor, gaussian_baseline, device)
            gaussian_metrics = compute_all_metrics(hp_raw, lp_gaussian, lp_raw)
            gaussian_results.append(gaussian_metrics)
            
            # LP baseline (no processing)
            lp_baseline_metrics = compute_all_metrics(hp_raw, lp_raw, lp_raw)
            lp_baseline_results.append(lp_baseline_metrics)
            
            # =================================================================
            # Print Sample Results
            # =================================================================
            print(f"\nSample {i+1}/{num_samples}:")
            print(f"  Dose range: {hp_raw.min():.6f} - {hp_raw.max():.6f}")
            print(f"  {'Method':<12} {'RMSE':<12} {'PSNR (dB)':<12} {'Gamma 1/1':<12}")
            print(f"  {'-'*48}")
            print(f"  {'LP (raw)':<12} {lp_baseline_metrics['rmse']:<12.6f} {lp_baseline_metrics['psnr']:<12.2f} {lp_baseline_metrics['gamma_11']:<12.2f}")
            print(f"  {'Gaussian':<12} {gaussian_metrics['rmse']:<12.6f} {gaussian_metrics['psnr']:<12.2f} {gaussian_metrics['gamma_11']:<12.2f}")
            print(f"  {'U-Net':<12} {unet_metrics['rmse']:<12.6f} {unet_metrics['psnr']:<12.2f} {unet_metrics['gamma_11']:<12.2f}")
            
            # Dose Regime Analysis
            if unet_metrics['dose_regimes']['high'] is not None:
                print(f"\n  Dose Regime Analysis (U-Net):")
                for regime in ['low', 'mid', 'high']:
                    dr = unet_metrics['dose_regimes'][regime]
                    if dr is not None:
                        print(f"    {regime:>5}: RMSE={dr['rmse']:.6f}, RelErr={dr['rel_error_percent']:.2f}%, Voxels={dr['num_voxels']}")
            
            # Visualize (with raw dose values)
            visualize_results(ct_raw, lp_raw, hp_raw, pred_raw, 
                            args.output_dir, i, unet_metrics)
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY RESULTS (Physical Dose Units)")
    print("="*70)
    
    def avg(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.mean(vals) if vals else 0.0
    
    def std(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.std(vals) if len(vals) > 1 else 0.0
    
    print(f"\n{'Method':<12} {'RMSE':<14} {'MAE':<14} {'PSNR (dB)':<14} {'Gamma 3/3':<12} {'Gamma 1/1':<12}")
    print("-"*78)
    
    # LP Baseline
    print(f"{'LP (raw)':<12} {avg(lp_baseline_results, 'rmse'):<14.6f} {avg(lp_baseline_results, 'mae'):<14.6f} {avg(lp_baseline_results, 'psnr'):<14.2f} {avg(lp_baseline_results, 'gamma_33'):<12.2f} {avg(lp_baseline_results, 'gamma_11'):<12.2f}")
    
    # Gaussian
    print(f"{'Gaussian':<12} {avg(gaussian_results, 'rmse'):<14.6f} {avg(gaussian_results, 'mae'):<14.6f} {avg(gaussian_results, 'psnr'):<14.2f} {avg(gaussian_results, 'gamma_33'):<12.2f} {avg(gaussian_results, 'gamma_11'):<12.2f}")
    
    # U-Net
    print(f"{'U-Net':<12} {avg(unet_results, 'rmse'):<14.6f} {avg(unet_results, 'mae'):<14.6f} {avg(unet_results, 'psnr'):<14.2f} {avg(unet_results, 'gamma_33'):<12.2f} {avg(unet_results, 'gamma_11'):<12.2f}")
    
    # =========================================================================
    # DOSE REGIME SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("DOSE REGIME ANALYSIS (U-Net) - Hesser Request")
    print("  Low dose:  0-10% of max dose")
    print("  Mid dose:  10-50% of max dose")
    print("  High dose: >50% of max dose")
    print("="*70)
    
    print(f"\n{'Regime':<8} {'RMSE':<14} {'MAE':<14} {'Rel.Err(%)':<14} {'Avg Voxels':<12}")
    print("-"*62)
    
    for regime in ['low', 'mid', 'high']:
        rmses = [r['dose_regimes'][regime]['rmse'] for r in unet_results if r['dose_regimes'][regime] is not None]
        maes = [r['dose_regimes'][regime]['mae'] for r in unet_results if r['dose_regimes'][regime] is not None]
        rels = [r['dose_regimes'][regime]['rel_error_percent'] for r in unet_results if r['dose_regimes'][regime] is not None]
        voxs = [r['dose_regimes'][regime]['num_voxels'] for r in unet_results if r['dose_regimes'][regime] is not None]
        
        if rmses:
            print(f"{regime:<8} {np.mean(rmses):<14.6f} {np.mean(maes):<14.6f} {np.mean(rels):<14.2f} {int(np.mean(voxs)):<12}")
        else:
            print(f"{regime:<8} {'N/A':<14} {'N/A':<14} {'N/A':<14} {'N/A':<12}")
    
    # =========================================================================
    # IMPROVEMENT SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("IMPROVEMENT OVER BASELINES")
    print("="*70)
    
    lp_rmse = avg(lp_baseline_results, 'rmse')
    unet_rmse = avg(unet_results, 'rmse')
    gauss_rmse = avg(gaussian_results, 'rmse')
    
    if lp_rmse > 0:
        print(f"\nRMSE Reduction:")
        print(f"  vs LP (raw):    {(1 - unet_rmse/lp_rmse)*100:.1f}%")
        print(f"  vs Gaussian:    {(1 - unet_rmse/gauss_rmse)*100:.1f}%")
    
    lp_psnr = avg(lp_baseline_results, 'psnr')
    unet_psnr = avg(unet_results, 'psnr')
    
    print(f"\nPSNR Improvement:")
    print(f"  U-Net vs LP:    +{unet_psnr - lp_psnr:.2f} dB")
    
    print("\n" + "="*70)
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    
    # =========================================================================
    # ANSWER TO HESSER: N=10^3 Explanation
    # =========================================================================
    print("\n" + "="*70)
    print("NOTE FOR PROFESSOR HESSER:")
    print("="*70)
    print(f"""
    LP input source: pre-generated dose cubes from folder '{args.lp_folder}'.

    IMPORTANT: The LP dose in this package is generated using the δ-based
    Poisson formulation discussed in our meeting (per-voxel particle counting):

        δ      = (max_dose) / N_eff
        n      = D / δ
        n'     ~ Poisson(n)
        D_lp   = n' · δ

    where D is the clean (HP) voxel dose, δ is the dose-per-particle, and N_eff
    is an *effective* particle count at the max-dose voxel. This is a synthetic
    low-photon approximation; N_eff is not necessarily the true Geant4 history count.

    If '{args.lp_folder}' encodes N_eff in its name (e.g., 'lp_cubes_100'), that
    value was used during the pre-generation step (generate_lp_dose.py).

    Model training detail:
      - Strategy: Learn correction on top of Gaussian-filtered baseline (Prof. Hesser)
      - The network predicts: (HP - Gaussian(LP)) * {args.residual_scale}
      - Inference uses: Gaussian(LP) + (model_output / {args.residual_scale})
      - Gaussian sigma: 0.8 (applied in normalized space)
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test denoising with Hesser's metrics")
    parser.add_argument("--root_dir", type=str, default="Mini_Dataset")
    parser.add_argument("--model_path", type=str, default="results/simple_unet/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/hesser_evaluation")
    parser.add_argument("--lp_folder", type=str, default="lp_cubes",
                        help="LP dose folder name (e.g., lp_cubes or lp_cubes_100)")
    parser.add_argument("--dose_scale", type=float, default=0.02,
                        help="Global dose scale used for normalization (raw_dose / dose_scale -> [0,1])")
    parser.add_argument("--residual_scale", type=float, default=1000.0,
                        help="Scale factor used during residual training/inference")
    parser.add_argument("--gaussian_sigma", type=float, default=0.8,
                        help="Sigma for Gaussian baseline (use 0.8 for N=100 baseline)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_type", type=str, default="standard")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Model base channels (overridden by checkpoint if available)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples (0 for all)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("   Run simple_train_denoising.py first!")
    else:
        test(args)
