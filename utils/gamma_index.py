# ============================================================================
# utils/gamma_index.py
# ============================================================================
# Purpose: Calculate Gamma Index for Dose Distributions.
#          Standard metric in Medical Physics (e.g., 3%/3mm).
# ============================================================================

import numpy as np

def calculate_gamma_index_3d(
    reference, 
    evaluated, 
    dta_mm=3.0, 
    dd_percent=3.0, 
    pixel_spacing_mm=(1.0, 1.0, 1.0), 
    threshold_percent=10.0
):
    """
    FAST Gamma Index calculation (Dose-Difference only approximation).
    
    The full 3D Gamma with DTA is extremely slow on CPU.
    This uses a dose-difference-only approximation which is:
    - Much faster (numpy vectorized)
    - Conservative (underestimates pass rate slightly)
    - Suitable for training/validation feedback
    
    For final publication results, use pymedphys.gamma separately.
    
    Args:
        reference (np.ndarray): Reference dose (Ground Truth).
        evaluated (np.ndarray): Evaluated dose (Prediction).
        dta_mm (float): Distance-to-Agreement threshold in mm (not used in fast mode).
        dd_percent (float): Dose Difference threshold in % (default 3%).
        pixel_spacing_mm (tuple): Voxel size in mm (not used in fast mode).
        threshold_percent (float): Lower dose threshold to exclude background.
        
    Returns:
        gamma_map (np.ndarray): Approximate Gamma map (DD-only).
        pass_rate (float): Percentage of points with Gamma <= 1.
    """
    max_dose = np.max(reference)
    if max_dose < 1e-10:
        return np.zeros_like(reference), 100.0
    
    # Dose difference threshold in absolute units
    dd_abs = (dd_percent / 100.0) * max_dose
    threshold_val = (threshold_percent / 100.0) * max_dose
    
    # Mask for valid dose region
    mask = reference > threshold_val
    
    if np.sum(mask) == 0:
        return np.zeros_like(reference), 100.0
    
    # Calculate dose difference (normalized to threshold)
    diff = np.abs(reference - evaluated)
    gamma = diff / (dd_abs + 1e-10)
    
    # Pass rate: fraction of masked voxels with gamma <= 1
    pass_rate = np.sum(gamma[mask] <= 1.0) / np.sum(mask) * 100.0
    
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
