# ============================================================================
# utils/gamma_index.py
# ============================================================================
# Purpose: Calculate Gamma Index for Dose Distributions using pymedphys.
#          Standard metric in Medical Physics (e.g., 3%/3mm).
# ============================================================================

import numpy as np
import pymedphys

def calculate_gamma_index_3d(
    reference, 
    evaluated, 
    dta_mm=3.0, 
    dd_percent=3.0, 
    pixel_spacing_mm=(1.0, 1.0, 1.0), 
    threshold_percent=10.0
):
    """
    Calculate 3D Gamma Index using pymedphys (Gold Standard).
    
    Args:
        reference (np.ndarray): Reference dose (Ground Truth).
        evaluated (np.ndarray): Evaluated dose (Prediction).
        dta_mm (float): Distance-to-Agreement threshold in mm (default 3mm).
        dd_percent (float): Dose Difference threshold in % (default 3%).
        pixel_spacing_mm (tuple): Voxel size in mm (z, y, x).
        threshold_percent (float): Lower dose threshold to exclude background (default 10% of max).
        
    Returns:
        gamma_map (np.ndarray): 3D Gamma map.
        pass_rate (float): Percentage of points with Gamma <= 1.
    """
    
    # Construct axes based on pixel spacing
    # Assuming origin at (0,0,0) for relative comparison
    z_vals = np.arange(reference.shape[0]) * pixel_spacing_mm[0]
    y_vals = np.arange(reference.shape[1]) * pixel_spacing_mm[1]
    x_vals = np.arange(reference.shape[2]) * pixel_spacing_mm[2]
    
    axes = (z_vals, y_vals, x_vals)
    
    # Calculate Gamma
    # Note: pymedphys.gamma can be slow for large 3D volumes.
    # We use global gamma (local_gamma=False) which normalizes to max dose of reference.
    gamma = pymedphys.gamma(
        axes_reference=axes,
        dose_reference=reference,
        axes_evaluation=axes,
        dose_evaluation=evaluated,
        dose_percent_threshold=dd_percent,
        distance_mm_threshold=dta_mm,
        lower_percent_dose_cutoff=threshold_percent,
        interp_fraction=10, # 10x interpolation for accuracy
        max_gamma=2.0,      # Stop calculating if gamma > 2
        local_gamma=False,  # Global normalization (standard)
        quiet=True
    )
    
    # Calculate Pass Rate
    # Valid points are those where gamma was calculated (not NaN)
    valid_mask = ~np.isnan(gamma)
    if np.sum(valid_mask) == 0:
        return gamma, 0.0
        
    pass_rate = np.sum(gamma[valid_mask] <= 1.0) / np.sum(valid_mask) * 100.0
    
    return gamma, pass_rate

def calculate_gamma_index_simple(reference, evaluated, dd_percent=3.0, threshold_percent=10.0):
    """
    Simplified Gamma (Dose Difference only) for fast checking.
    Kept for backward compatibility or quick checks.
    """
    max_dose = np.max(reference)
    dd_abs = (dd_percent / 100.0) * max_dose
    threshold_val = (threshold_percent / 100.0) * max_dose
    
    mask = reference > threshold_val
    
    diff = np.abs(reference - evaluated)
    gamma = diff / dd_abs
    
    valid_gamma = gamma[mask]
    if valid_gamma.size == 0:
        return 0.0, 0.0
        
    pass_rate = np.sum(valid_gamma <= 1.0) / valid_gamma.size * 100.0
    
    return np.mean(valid_gamma), pass_rate
