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
    
    This simulates what would happen if we ran Monte Carlo with fewer photons:
    - HP dose: 10^10 photons (ground truth, very smooth)
    - LP dose: 10^5 photons (noisy, but fast to compute)
    
    The relationship between variance and photon count:
    - Variance ∝ 1/N (more photons = less variance)
    - SNR ∝ √N (more photons = better signal-to-noise)
    
    Args:
        hp_dose: High-Photon dose volume (float32 or float64)
                 Shape: (D, H, W), values typically in range [1e-6, 1e-2]
        num_photons: Number of photons to simulate
                     - 1e10 = no noise (original HP)
                     - 1e8  = very low noise
                     - 1e6  = low noise
                     - 1e5  = medium noise (default)
                     - 1e4  = high noise
                     - 1e3  = very high noise
    
    Returns:
        lp_dose: Low-Photon dose volume (float32)
                 Same shape as hp_dose, but with Poisson noise
    """
    # Ensure non-negative (dose cannot be negative)
    hp_dose = np.maximum(hp_dose, 0)
    
    # Scale to photon counts (expected number of photons per voxel)
    # If hp_dose = 0.01 and N = 1e5, then expected_counts = 1000
    expected_counts = hp_dose * num_photons
    
    # Sample from Poisson distribution
    # This is the key step: Poisson(λ) has variance = λ
    # So variance of lp_dose = hp_dose / N (inverse relationship with photon count)
    actual_counts = np.random.poisson(expected_counts).astype(np.float32)
    
    # Scale back to dose range
    lp_dose = actual_counts / num_photons
    
    return lp_dose


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
                    num_samples_to_print: int = 3, output_folder: str = "lp_cubes"):
    """
    Process entire dataset to generate LP dose volumes.
    
    Folder structure (before):
        root_dir/
        └── {energy}/           # e.g., "46_53" for 46.53 keV
            └── {patient_id}/
                ├── input_cubes/    # CT volumes
                │   └── *.npy
                └── output_cubes/   # HP dose volumes
                    └── *.npy
    
    Folder structure (after):
        root_dir/
        └── {energy}/
            └── {patient_id}/
                ├── input_cubes/    # CT volumes (unchanged)
                │   └── *.npy
                ├── output_cubes/   # HP dose volumes (unchanged)
                │   └── *.npy
                └── {output_folder}/ # NEW: LP dose volumes (e.g., lp_cubes or lp_cubes_100)
                    └── *.npy
    
    Args:
        root_dir: Path to dataset root
        num_photons: Number of photons for Poisson simulation
        num_samples_to_print: Number of samples to print detailed stats
        output_folder: Name of output folder for LP doses (default: lp_cubes)
    """
    print("=" * 70)
    print("Generate Low-Photon (LP) Dose from High-Photon (HP) Dose")
    print("=" * 70)
    print(f"Dataset root: {root_dir}")
    print(f"Number of photons (N): {num_photons:.0e}")
    print(f"Expected noise level: σ ∝ 1/√N = {1/np.sqrt(num_photons):.2e}")
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
        
        # Generate LP dose using Poisson statistics
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
