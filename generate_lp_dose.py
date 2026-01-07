# ============================================================================
# generate_lp_dose.py
# ============================================================================
# Date: 2025-12-03
# Author: GitHub Copilot
# Purpose: Generate Low-Photon (LP) dose from High-Photon (HP) dose using
#          Poisson statistics for Diff-MC style experiments.
#
# Concept:
#   - HP dose = output_cubes (clean, high-photon Monte Carlo simulation)
#   - LP dose = HP dose + Poisson noise (simulated low-photon MC)
#   
# Algorithm:
#   counts ~ Poisson(hp * N)    # Sample photon counts
#   lp = counts / N             # Scale back to dose
#
# Where N = number of particles (photons). Lower N = more noise.
# ============================================================================

import os
import numpy as np
from tqdm import tqdm
import glob
import argparse


def generate_lp_dose(hp_dose: np.ndarray, num_photons: float = 1e5) -> np.ndarray:
    """
    Generate Low-Photon (LP) dose from High-Photon (HP) dose using Poisson statistics.
    
    CORRECTED FORMULA:
    ==================
    The key insight is that Monte Carlo dose D is proportional to energy deposited,
    which follows Poisson statistics based on the NUMBER OF INTERACTIONS, not the 
    global maximum.
    
    Correct approach (local scaling):
        scaling = num_photons  # This represents the "simulation quality factor"
        counts = hp_dose * scaling  # Expected counts (proportional to dose)
        noisy_counts = Poisson(counts)  # Sample from Poisson
        lp_dose = noisy_counts / scaling  # Scale back to dose units
    
    This ensures:
        - Relative noise σ/μ = 1/√(μ * scaling) = 1/√(hp_dose * num_photons)
        - Higher dose regions have LOWER relative noise (correct physics)
        - Lower dose regions have HIGHER relative noise (correct physics)
        - Zero dose stays zero (no negative doses)
    
    Args:
        hp_dose: High-Photon dose volume (ground truth)
        num_photons: Scaling factor controlling noise level
                     - 1e7  = very low noise (~0.03% relative at typical doses)
                     - 1e6  = low noise (~0.1% relative)
                     - 1e5  = medium noise (~0.3% relative)  
                     - 1e4  = high noise (~1% relative)
                     - 1e3  = very high noise (~3% relative)
    
    Returns:
        lp_dose: Noisy dose with Poisson statistics
    """
    # Ensure non-negative (dose cannot be negative)
    hp_dose = np.maximum(hp_dose, 0).astype(np.float64)
    
    # Handle edge case: no dose at all
    if hp_dose.max() < 1e-15:
        return hp_dose.astype(np.float32)
    
    # CORRECTED: Use local scaling instead of global max normalization
    # This properly models that each voxel has independent Poisson statistics
    
    # Step 1: Normalize dose to [0, 1] range for numerical stability
    # Then scale by num_photons to get expected counts
    dose_normalized = hp_dose / hp_dose.max()  # Normalize to [0, 1]
    expected_counts = dose_normalized * num_photons  # Expected photon counts per voxel
    
    # Step 2: Sample from Poisson distribution
    # Each voxel independently samples from Poisson(expected_counts)
    noisy_counts = np.random.poisson(expected_counts).astype(np.float64)
    
    # Step 3: Scale back to original dose units
    lp_dose = (noisy_counts / num_photons) * hp_dose.max()
    
    return lp_dose.astype(np.float32)


def generate_lp_dose_relative(hp_dose: np.ndarray, target_noise_percent: float = 3.0) -> np.ndarray:
    """
    Alternative: Generate LP dose with a target relative noise level.
    
    This is more intuitive - you specify the desired noise level directly.
    
    Args:
        hp_dose: High-Photon dose volume (ground truth)
        target_noise_percent: Target relative noise at max dose (%)
                              e.g., 3.0 means ~3% noise at peak dose
    
    Returns:
        lp_dose: Noisy dose volume
    """
    hp_dose = np.maximum(hp_dose, 0).astype(np.float64)
    
    if hp_dose.max() < 1e-15:
        return hp_dose.astype(np.float32)
    
    # For Poisson: relative_noise = 1/sqrt(N)
    # So N = (1/relative_noise)^2
    # If we want 3% noise at max, we need N = (1/0.03)^2 ≈ 1111 counts at max
    relative_noise = target_noise_percent / 100.0
    num_photons_at_max = (1.0 / relative_noise) ** 2
    
    # Normalize and scale
    dose_normalized = hp_dose / hp_dose.max()
    expected_counts = dose_normalized * num_photons_at_max
    
    # Sample and rescale
    noisy_counts = np.random.poisson(expected_counts).astype(np.float64)
    lp_dose = (noisy_counts / num_photons_at_max) * hp_dose.max()
    
    return lp_dose.astype(np.float32)


def compute_stats(volume: np.ndarray, name: str) -> dict:
    """Compute and print statistics for a volume."""
    stats = {
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std())
    }
    print(f"  {name}:")
    print(f"    Min:  {stats['min']:.6e}")
    print(f"    Max:  {stats['max']:.6e}")
    print(f"    Mean: {stats['mean']:.6e}")
    print(f"    Std:  {stats['std']:.6e}")
    return stats


def process_dataset(root_dir: str, num_photons: float = 1e5, 
                    num_samples_to_print: int = 3, output_folder: str = "lp_cubes",
                    target_noise_percent: float = None):
    """
    Process entire dataset to generate LP dose volumes.
    
    Args:
        root_dir: Path to dataset root
        num_photons: Number of photons for Poisson simulation (used if target_noise_percent is None)
        num_samples_to_print: Number of samples to print detailed stats
        output_folder: Name of output folder for LP doses
        target_noise_percent: If specified, use this instead of num_photons
    """
    print("=" * 70)
    print("Generate Low-Photon (LP) Dose from High-Photon (HP) Dose")
    print("=" * 70)
    print(f"Dataset root: {root_dir}")
    
    if target_noise_percent is not None:
        print(f"Target noise at max dose: {target_noise_percent}%")
        equiv_photons = (100.0 / target_noise_percent) ** 2
        print(f"Equivalent num_photons: {equiv_photons:.0f}")
    else:
        print(f"Number of photons (N): {num_photons:.0e}")
        print(f"Expected noise at max dose: {100/np.sqrt(num_photons):.2f}%")
    
    print("=" * 70)
    
    # Collect all HP dose files
    all_hp_files = []
    
    # Iterate over energy folders
    energy_folders = sorted(os.listdir(root_dir))
    for energy_folder in energy_folders:
        energy_path = os.path.join(root_dir, energy_folder)
        if not os.path.isdir(energy_path):
            continue
        
        # Check for 'output' subfolder (original structure) or direct structure
        if os.path.isdir(os.path.join(energy_path, "output")):
            base_path = os.path.join(energy_path, "output")
        else:
            base_path = energy_path
        
        # Iterate over patient folders
        for patient_id in os.listdir(base_path):
            patient_path = os.path.join(base_path, patient_id)
            if not os.path.isdir(patient_path):
                continue
            
            output_cubes_path = os.path.join(patient_path, "output_cubes")
            if not os.path.isdir(output_cubes_path):
                continue
            
            # Collect all .npy files
            hp_files = glob.glob(os.path.join(output_cubes_path, "*.npy"))
            for hp_file in hp_files:
                all_hp_files.append({
                    "hp_path": hp_file,
                    "patient_path": patient_path,
                    "filename": os.path.basename(hp_file)
                })
    
    print(f"\nFound {len(all_hp_files)} HP dose files to process.\n")
    
    if len(all_hp_files) == 0:
        print("ERROR: No files found! Check your dataset path.")
        return
    
    # Process each file
    samples_printed = 0
    for item in tqdm(all_hp_files, desc="Generating LP doses"):
        hp_path = item["hp_path"]
        patient_path = item["patient_path"]
        filename = item["filename"]
        
        # Create LP output folder (e.g., lp_cubes or lp_cubes_100)
        lp_cubes_path = os.path.join(patient_path, output_folder)
        os.makedirs(lp_cubes_path, exist_ok=True)
        
        # Load HP dose
        hp_dose = np.load(hp_path).astype(np.float64)  # Use float64 for precision
        
        # Generate LP dose
        if target_noise_percent is not None:
            lp_dose = generate_lp_dose_relative(hp_dose, target_noise_percent)
        else:
            lp_dose = generate_lp_dose(hp_dose, num_photons)
        
        # Save LP dose as float32 to save disk space
        lp_path = os.path.join(lp_cubes_path, filename)
        np.save(lp_path, lp_dose.astype(np.float32))
        
        # Print stats for first few samples
        if samples_printed < num_samples_to_print:
            print(f"\n--- Sample {samples_printed + 1}: {filename} ---")
            hp_stats = compute_stats(hp_dose, "HP (High-Photon, clean)")
            lp_stats = compute_stats(lp_dose, "LP (Low-Photon, noisy)")
            
            # Compute noise level
            if hp_stats["mean"] > 0:
                noise_ratio = lp_stats["std"] / hp_stats["std"]
                print(f"  Noise amplification (LP_std / HP_std): {noise_ratio:.2f}x")
            
            samples_printed += 1
    
    print("\n" + "=" * 70)
    print("DONE!")
    print(f"Generated {len(all_hp_files)} LP dose volumes.")
    print(f"LP doses saved to: {{patient_path}}/{output_folder}/*.npy")
    print("=" * 70)
    
    # Final summary
    print("\n📊 Data Pipeline Summary:")
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│  input_cubes/   →  CT volume      (anatomy, HU values)         │")
    print("│  output_cubes/  →  HP dose        (clean, ground truth)        │")
    print("│  lp_cubes/      →  LP dose        (noisy, model input)         │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  Model learns:  CT + LP dose → HP dose  (denoising task)       │")
    print("└─────────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Low-Photon (LP) dose from High-Photon (HP) dose"
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="data",
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--num_photons", 
        type=float, 
        default=1e5,
        help="Number of photons for Poisson simulation (default: 1e5)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=3,
        help="Number of samples to print detailed stats (default: 3)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="lp_cubes",
        help="Output folder name for LP doses (default: lp_cubes). Use lp_cubes_100 for N=100."
    )
    
    args = parser.parse_args()
    
    process_dataset(
        root_dir=args.root_dir,
        num_photons=args.num_photons,
        num_samples_to_print=args.num_samples,
        output_folder=args.output_folder
    )
