#!/usr/bin/env python3
"""
Pre-flight Check - Verify everything is ready
"""
import os
import sys

print("="*70)
print("🔍 PRE-FLIGHT CHECK - Hesser MC Denoising")
print("="*70)

models_dir = r"C:\Users\irpay\OneDrive\Desktop\HESSER\MC_Denoising_Training\models"
os.chdir(models_dir)

checks_passed = 0
checks_total = 0

# Check 1: Python files exist
print("\n📝 Check 1: Python scripts")
checks_total += 1
required_files = [
    "generate_lp_dose.py",
    "simple_train_denoising.py", 
    "test_denoising_hesser.py",
    "simple_unet_denoiser.py"
]
missing = []
for f in required_files:
    if os.path.exists(f):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} NOT FOUND")
        missing.append(f)

if not missing:
    checks_passed += 1
    print("  ✅ All Python scripts present")
else:
    print(f"  ❌ Missing: {', '.join(missing)}")

# Check 2: Dataset exists
print("\n📊 Check 2: Dataset")
checks_total += 1
data_path = "46_53-32_cube/output"
if os.path.exists(data_path):
    # Count patient folders
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print(f"  ✅ Dataset found: {len(folders)} patient folders")
    
    # Check first folder structure
    if folders:
        first_folder = os.path.join(data_path, folders[0])
        input_path = os.path.join(first_folder, "input_cubes")
        output_path = os.path.join(first_folder, "output_cubes")
        
        if os.path.exists(input_path):
            ct_files = len([f for f in os.listdir(input_path) if f.endswith('.npy')])
            print(f"  ✅ input_cubes/ found: {ct_files} CT files")
        else:
            print(f"  ❌ input_cubes/ not found in {folders[0]}")
        
        if os.path.exists(output_path):
            hp_files = len([f for f in os.listdir(output_path) if f.endswith('.npy')])
            print(f"  ✅ output_cubes/ found: {hp_files} HP dose files")
        else:
            print(f"  ❌ output_cubes/ not found in {folders[0]}")
        
        lp_path = os.path.join(first_folder, "lp_cubes")
        if os.path.exists(lp_path):
            lp_files = len([f for f in os.listdir(lp_path) if f.endswith('.npy')])
            print(f"  ⚠️  lp_cubes/ already exists: {lp_files} LP files (will be overwritten)")
        else:
            print(f"  ℹ️  lp_cubes/ not found (will be created)")
    
    checks_passed += 1
else:
    print(f"  ❌ Dataset not found at: {data_path}")

# Check 3: Dependencies
print("\n📦 Check 3: Python packages")
checks_total += 1
try:
    import torch
    print(f"  ✅ torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠️  CUDA not available (will use CPU - slower)")
except ImportError:
    print(f"  ❌ torch not installed")

try:
    import numpy
    print(f"  ✅ numpy {numpy.__version__}")
except ImportError:
    print(f"  ❌ numpy not installed")

try:
    import tqdm
    print(f"  ✅ tqdm")
except ImportError:
    print(f"  ❌ tqdm not installed")

try:
    import matplotlib
    print(f"  ✅ matplotlib")
except ImportError:
    print(f"  ❌ matplotlib not installed")

try:
    import scipy
    print(f"  ✅ scipy")
except ImportError:
    print(f"  ❌ scipy not installed")

checks_passed += 1

# Check 4: Dataset module
print("\n📚 Check 4: Custom modules")
checks_total += 1
sys.path.insert(0, os.path.dirname(models_dir))
try:
    from dataset.pl_dose_dataset import ConditionalDoseDataset
    print(f"  ✅ dataset.pl_dose_dataset")
except ImportError as e:
    print(f"  ❌ dataset.pl_dose_dataset: {e}")

try:
    from utils.gamma_index import calculate_gamma_index_3d
    print(f"  ✅ utils.gamma_index")
except ImportError as e:
    print(f"  ❌ utils.gamma_index: {e}")

checks_passed += 1

# Summary
print("\n" + "="*70)
print(f"📊 SUMMARY: {checks_passed}/{checks_total} checks passed")
print("="*70)

if checks_passed == checks_total:
    print("\n✅ All checks passed! You're ready to start.")
    print("\n🚀 Next step: Generate LP dose")
    print('   python start_now.py')
    print('   OR')
    print('   python generate_lp_dose.py --root_dir "46_53-32_cube/output" --num_photons 1000 --output_folder "lp_cubes"')
else:
    print("\n⚠️  Some checks failed. Please fix the issues above.")
    print("\nCommon fixes:")
    print("  - Install PyTorch: pip install torch torchvision torchaudio")
    print("  - Install dependencies: pip install -r requirements.txt")
    print("  - Check dataset path")

print("\n" + "="*70)
