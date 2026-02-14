# ============================================================================
# utils/gamma_index.py
# ============================================================================
# WARNING: This module implements a DOSE-DIFFERENCE-ONLY approximation.
#          It does NOT include Distance-To-Agreement (DTA).
#          The dta_mm and pixel_spacing_mm parameters are ACCEPTED but IGNORED.
#
#          DO NOT report results from this module as "Gamma X%/Xmm" in any
#          publication — that name implies DTA is included.
#
#          For proper Gamma (DTA+DD), use pymedphys.gamma() instead.
#          See: evaluate_gamma_proper.py, honest_evaluation.py
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
    FAST **Dose-Difference-only** approximation (NOT full Gamma).
    
    ⚠️  DTA IS NOT USED.  dta_mm and pixel_spacing_mm are accepted for
    API compatibility but silently ignored.  Results should be reported
    as "DD-only X%" and NEVER as "Gamma X%/Xmm".
    
    For publication-quality Gamma with DTA, use pymedphys.gamma().
    
    Args:
        reference (np.ndarray): Reference dose (Ground Truth).
        evaluated (np.ndarray): Evaluated dose (Prediction).
        dta_mm (float): **IGNORED** — kept for API compatibility.
        dd_percent (float): Dose Difference threshold in % (default 3%).
        pixel_spacing_mm (tuple): **IGNORED** — kept for API compatibility.
        threshold_percent (float): Lower dose threshold to exclude background.
        
    Returns:
        gamma_map (np.ndarray): DD-only approximate Gamma map.
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
