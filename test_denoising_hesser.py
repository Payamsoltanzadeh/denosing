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
from utils.gamma_index import calculate_gamma_index_3d


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
def apply_gaussian_filter(volume, sigma=1.0):
    """
    Apply 3D Gaussian filter for baseline comparison.
    This is the "linear denoising filter" Hesser mentioned.
    
    Args:
        volume: 3D dose volume
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Smoothed volume
    """
    return gaussian_filter(volume, sigma=sigma)


def apply_bilateral_filter_approx(volume, sigma_spatial=1.0, sigma_intensity=None):
    """
    Approximate bilateral filter for 3D volumes.
    Bilateral filter is edge-preserving (non-linear).
    
    True 3D bilateral is slow, this is a fast approximation using:
    - Gaussian smoothing for spatial component
    - Gradient-based weighting for edge preservation
    
    Args:
        volume: 3D dose volume
        sigma_spatial: Spatial smoothing strength
        sigma_intensity: Intensity smoothing (auto-computed if None)
    
    Returns:
        Edge-preserving smoothed volume
    """
    if sigma_intensity is None:
        sigma_intensity = np.std(volume) * 0.5
    
    # Gaussian filtered version
    gaussian_filtered = gaussian_filter(volume, sigma=sigma_spatial)
    
    # Median filtered version (for edge preservation)
    median_filtered = median_filter(volume, size=3)
    
    # Blend based on local gradient (edge-aware)
    gradient = np.abs(volume - gaussian_filtered)
    alpha = np.exp(-gradient / (sigma_intensity + 1e-10))
    
    # Blend: high alpha → use Gaussian, low alpha → use Median (edge)
    result = alpha * gaussian_filtered + (1 - alpha) * median_filtered
    
    return result


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
    # =========================================================================
    
    print("\n📂 Loading datasets...")
    
    # Dataset for MODEL INPUT (normalized)
    dataset_norm = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,  # Model expects normalized input
        target_dim=64,
        num_photons=1e3,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
    )
    
    # Dataset for METRICS (raw, physical units)
    dataset_raw = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=False,  # RAW physical dose for metrics!
        target_dim=64,
        num_photons=1e3,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
    )
    
    print(f"   Loaded {len(dataset_norm)} samples")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\n🏗️ Loading model from: {args.model_path}")
    model = get_simple_denoiser(
        model_type=args.model_type,
        base_channels=args.base_channels
    ).to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("   ✅ Model loaded successfully")
    else:
        print(f"   ❌ Model not found! Using random weights.")
    
    model.eval()
    
    # =========================================================================
    # Results Storage
    # =========================================================================
    unet_results = []
    gaussian_results = []
    bilateral_results = []
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
            # =================================================================
            inputs_norm = torch.cat([ct_norm.unsqueeze(0), lp_norm.unsqueeze(0)], dim=1).to(device)
            pred_norm = model(inputs_norm)
            pred_norm_np = pred_norm[0, 0].cpu().numpy()
            
            # =================================================================
            # UN-NORMALIZE prediction to physical dose
            # 
            # Problem: Model output is in normalized space [-2, +3] roughly
            # Solution: Map back to physical dose range
            # 
            # Method: Linear mapping from normalized to raw range
            # =================================================================
            hp_norm_np = hp_norm.squeeze().numpy()
            
            # Get normalization stats from raw data
            hp_mean = hp_raw.mean()
            hp_std = hp_raw.std() + 1e-10
            
            # Un-normalize: x_raw = x_norm * std + mean
            pred_raw = (pred_norm_np * hp_std) + hp_mean
            pred_raw = np.maximum(pred_raw, 0)  # Dose must be >= 0
            
            # =================================================================
            # Compute U-Net Metrics (on PHYSICAL dose)
            # =================================================================
            unet_metrics = compute_all_metrics(hp_raw, pred_raw, lp_raw)
            unet_results.append(unet_metrics)
            
            # =================================================================
            # Classical Filter Baselines (on RAW LP)
            # =================================================================
            # Gaussian filter (sigma=1.0, typical for MC denoising)
            lp_gaussian = apply_gaussian_filter(lp_raw, sigma=args.gaussian_sigma)
            gaussian_metrics = compute_all_metrics(hp_raw, lp_gaussian, lp_raw)
            gaussian_results.append(gaussian_metrics)
            
            # Bilateral filter (edge-preserving)
            lp_bilateral = apply_bilateral_filter_approx(lp_raw, sigma_spatial=1.0)
            bilateral_metrics = compute_all_metrics(hp_raw, lp_bilateral, lp_raw)
            bilateral_results.append(bilateral_metrics)
            
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
            print(f"  {'Bilateral':<12} {bilateral_metrics['rmse']:<12.6f} {bilateral_metrics['psnr']:<12.2f} {bilateral_metrics['gamma_11']:<12.2f}")
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
    
    # Bilateral
    print(f"{'Bilateral':<12} {avg(bilateral_results, 'rmse'):<14.6f} {avg(bilateral_results, 'mae'):<14.6f} {avg(bilateral_results, 'psnr'):<14.2f} {avg(bilateral_results, 'gamma_33'):<12.2f} {avg(bilateral_results, 'gamma_11'):<12.2f}")
    
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
    bilat_rmse = avg(bilateral_results, 'rmse')
    
    if lp_rmse > 0:
        print(f"\nRMSE Reduction:")
        print(f"  vs LP (raw):    {(1 - unet_rmse/lp_rmse)*100:.1f}%")
        print(f"  vs Gaussian:    {(1 - unet_rmse/gauss_rmse)*100:.1f}%")
        print(f"  vs Bilateral:   {(1 - unet_rmse/bilat_rmse)*100:.1f}%")
    
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

    IMPORTANT: In this codebase the LP dose is generated by injecting voxel-wise
    Poisson noise into the HP dose (synthetic low-photon approximation):

        LP = Poisson(HP * N) / N

    Here, N is an effective scaling parameter controlling the noise level.
    It is NOT necessarily equal to a true Monte Carlo source-particle history count.
    The noise level scales approximately as sqrt(HP / N).

    If '{args.lp_folder}' encodes N in its name (e.g., 'lp_cubes_100'), then that
    N was used during the pre-generation step (generate_lp_dose.py).
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test denoising with Hesser's metrics")
    parser.add_argument("--root_dir", type=str, default="Mini_Dataset")
    parser.add_argument("--model_path", type=str, default="results/simple_unet/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="results/hesser_evaluation")
    parser.add_argument("--lp_folder", type=str, default="lp_cubes",
                        help="LP dose folder name (e.g., lp_cubes or lp_cubes_100)")
    parser.add_argument("--gaussian_sigma", type=float, default=0.8,
                        help="Sigma for Gaussian baseline (use 0.8 for N=100 baseline)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_type", type=str, default="standard")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples (0 for all)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("   Run simple_train_denoising.py first!")
    else:
        test(args)
