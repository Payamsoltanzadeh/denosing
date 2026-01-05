#!/usr/bin/env python3
"""
EXECUTE NOW - RunPod Direct Execution
======================================
This will immediately start the LP dose generation process.
Run this with: python EXECUTE_NOW.py
"""

import os
import sys

# Find and change to models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("\n" + "="*80)
print("🚀 STARTING LP DOSE GENERATION NOW")
print("="*80)
print(f"📂 Working directory: {os.getcwd()}")
print("="*80 + "\n")

# Execute the LP generation script directly
try:
    print("⚙️  Loading generate_lp_now.py...\n")
    with open("generate_lp_now.py", "r") as f:
        code = f.read()
    
    print("▶️  Executing...\n")
    exec(code)
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    
except FileNotFoundError:
    print("❌ Error: generate_lp_now.py not found in current directory")
    print(f"   Current dir: {os.getcwd()}")
    print("\n💡 Solution:")
    print("   1. Find your models directory:")
    print("      find /workspace -name 'generate_lp_now.py' 2>/dev/null")
    print("   2. cd to that directory")
    print("   3. Run: python EXECUTE_NOW.py")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n📋 NEXT STEP: Training")
print("-"*80)
print("Run this command:")
print()
print("python simple_train_denoising.py \\")
print('  --root_dir "46_53-32_cube/output" \\')
print('  --lp_folder "lp_cubes" \\')
print("  --epochs 50 \\")
print("  --batch_size 4 \\")
print("  --lr 1e-4 \\")
print("  --device gpu \\")
print('  --save_dir "checkpoints"')
print()
print("⏱️  Time: ~3-5 hours on GPU")
print("="*80 + "\n")
