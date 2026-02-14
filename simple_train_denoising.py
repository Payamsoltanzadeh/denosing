# ============================================================================
# simple_train_denoising.py
# ============================================================================
# Date: 2025-12-05 (Updated: 2026-01-29)
# Purpose: Pure PyTorch training script (No Lightning dependencies).
#          Now with Hesser Loss (Weighted MSE + L4) for better peak preservation.
# ============================================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import Model and Dataset
from models.simple_unet_denoiser import get_simple_denoiser
from dataset.pl_dose_dataset import ConditionalDoseDataset
from utils.torch_gaussian import get_gaussian_layer
from utils.losses import HesserLoss, SobelGradient3DLoss  # NEW: Custom loss for peak preservation

def train(args):
    # ------------------------------------------------------------------------
    # 1. Setup Device and Folders
    # ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    print(f"🚀 Training on: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # 2. Prepare Data
    # ------------------------------------------------------------------------
    print(f"📂 Loading dataset from: {args.root_dir}")
    
    # ========================================================================
    # FOLDER-LEVEL SPLIT (Fixed: 2026-02-03)
    # Per Marcus Buchwald: Each folder is an independent simulation
    # Each folder: 10 patients × 200 patches = 2,000 patches
    # Total: 10 folders × 2,000 = 20,000 patches = 100 patients
    # 
    # Split by FOLDER to prevent data leakage:
    #   - Train: 5 folders (50% = 10,000 patches = 50 patients)
    #   - Val:   2 folders (20% = 4,000 patches = 20 patients)
    #   - Test:  3 folders (30% = 6,000 patches = 30 patients)
    # ========================================================================
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
    
    TRAIN_FOLDERS = ALL_FOLDERS[:7]   # First 7 folders (70%)
    VAL_FOLDERS = ALL_FOLDERS[7:8]    # Next 1 folder (10%)
    # TEST_FOLDERS = ALL_FOLDERS[8:]  # Last 2 folders (20%) - used in test script
    
    print(f"   Folder-level split (NO DATA LEAKAGE):")
    print(f"   - Train: {len(TRAIN_FOLDERS)} folders ({len(TRAIN_FOLDERS)*2000} patches, {len(TRAIN_FOLDERS)*10} patients)")
    print(f"   - Val:   {len(VAL_FOLDERS)} folders ({len(VAL_FOLDERS)*2000} patches, {len(VAL_FOLDERS)*10} patients)")
    print(f"   - Test:  2 folders (4000 patches, 20 patients) - held out")
    
    # Training Dataset (5 folders)
    train_set = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,
        target_dim=32,  # Match actual data size (32x32x32)
        delta=None,  # Auto-calculate using Hesser formula
        target_uncertainty=0.10,  # 10% uncertainty at max dose
        add_noise=False,  # Use pre-generated LP
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
        dose_scale=args.dose_scale,
        folder_names=TRAIN_FOLDERS,  # NEW: Folder-level split
        augment=args.augment,  # NEW: 3D random flips (2026-02-14)
    )
    
    # Validation Dataset (2 folders)
    val_set = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,
        target_dim=32,
        delta=None,
        target_uncertainty=0.10,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
        dose_scale=args.dose_scale,
        folder_names=VAL_FOLDERS,  # NEW: Folder-level split
    )
    
    # Limit samples if requested (for faster experimentation)
    if args.max_samples is not None and args.max_samples < len(train_set):
        print(f"   ⚡ Limiting training to {args.max_samples} samples (from {len(train_set)})")
        train_set, _ = torch.utils.data.random_split(
            train_set, 
            [args.max_samples, len(train_set) - args.max_samples]
        )
    
    print(f"   Train samples: {len(train_set)}")
    print(f"   Val samples:   {len(val_set)}")

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,  # Parallel data loading for speed
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # ------------------------------------------------------------------------
    # 3. Setup Model, Optimizer, Loss
    # ------------------------------------------------------------------------
    print(f"🏗️ Creating model: {args.model_type} (base_channels={args.base_channels})")
    model = get_simple_denoiser(
        model_type=args.model_type,
        base_channels=args.base_channels
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # NEW: Use Hesser Loss instead of plain MSE
    if args.loss_type == "mse":
        criterion = nn.MSELoss()
        print("📉 Loss: Standard MSE")
    elif args.loss_type == "hesser":
        criterion = HesserLoss(
            l4_weight=args.l4_weight,
            dose_weight_alpha=args.dose_weight_alpha,
            residual_scale=args.residual_scale,
        )
        print(f"📉 Loss: Hesser (Weighted MSE + L4)")
        print(f"   L4 weight: {args.l4_weight}")
        print(f"   Dose weight alpha: {args.dose_weight_alpha}")
        print(f"   L4 residual_scale: {args.residual_scale} (L4 computed in dose space)")
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Gradient Loss (Sobel) - Optional
    if args.gradient_weight > 0:
        print(f"   Gradient Loss (Sobel) enabled with weight: {args.gradient_weight}")
        grad_criterion = SobelGradient3DLoss(device=device)
    else:
        grad_criterion = None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # ------------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------------
    print("🔥 Starting training...")
    best_val_loss = float('inf')

    # Create Gaussian blur layer for residual baseline
    print(f"   Gaussian blur sigma: {args.gaussian_sigma}")
    gaussian_blur = get_gaussian_layer(channels=1, sigma=args.gaussian_sigma, device=device)
    gaussian_blur.eval()  # Always in eval mode
    
    scaling_factor = float(args.residual_scale)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            # Unpack batch
            hp_target, condition = batch
            
            ct = condition['ct'].to(device)
            lp = condition['lp_dose'].to(device)
            hp_target = hp_target.to(device)
            
            # Concatenate inputs: [B, 2, D, H, W]
            inputs = torch.cat([ct, lp], dim=1)
            
            # Forward - RESIDUAL LEARNING with SIGNAL AMPLIFICATION
            # New strategy: predict correction on top of Gaussian(LP)
            # Target = (HP - Gaussian(LP)) * scaling_factor
            
            optimizer.zero_grad()
            pred_residual = model(inputs)
            
            # Apply Gaussian blur to LP to get baseline
            with torch.no_grad():
                lp_gaussian = gaussian_blur(lp)
            
            # Target is the residual from Gaussian baseline, scaled up
            target_residual = (hp_target - lp_gaussian) * scaling_factor
            
            # Loss on the residual directly
            # Pass lp_gaussian as dose_weight so WMSE weights by actual dose,
            # not by residual magnitude (Bug A fix: 2026-02-14)
            loss = criterion(pred_residual, target_residual, dose_weight=lp_gaussian)

            # Add Gradient Loss if enabled
            if grad_criterion is not None:
                g_loss = grad_criterion(pred_residual, target_residual)
                loss = loss + args.gradient_weight * g_loss
            
            # Backward
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Step scheduler
        scheduler.step()

        # --------------------------------------------------------------------
        # 5. Validation Loop
        # --------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        # Must match training
        
        with torch.no_grad():
            for batch in val_loader:
                hp_target, condition = batch
                ct = condition['ct'].to(device)
                lp = condition['lp_dose'].to(device)
                hp_target = hp_target.to(device)
                
                inputs = torch.cat([ct, lp], dim=1)
                pred_residual = model(inputs)
                
                # Apply Gaussian blur to LP (same as training)
                lp_gaussian = gaussian_blur(lp)
                
                # Check loss in the scaled space (same as training)
                target_residual = (hp_target - lp_gaussian) * scaling_factor
                loss = criterion(pred_residual, target_residual, dose_weight=lp_gaussian)
                
                if grad_criterion is not None:
                    g_loss = grad_criterion(pred_residual, target_residual)
                    loss = loss + args.gradient_weight * g_loss

                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   Results: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --------------------------------------------------------------------
        # 6. Save Checkpoint
        # --------------------------------------------------------------------
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'args': vars(args),
            }, save_path)
            print(f"   ✅ Saved best model to {save_path}")
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': avg_val_loss,
        }, os.path.join(args.save_dir, "last_model.pth"))

    print("🏁 Training finished!")
    print(f"   Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/simple_unet")
    parser.add_argument("--lp_folder", type=str, default="lp_cubes",
                        help="LP dose folder name (e.g., lp_cubes or lp_cubes_100)")
    parser.add_argument("--dose_scale", type=float, default=0.02,
                        help="Global dose scale used for normalization (raw_dose / dose_scale -> [0,1])")
    parser.add_argument("--residual_scale", type=float, default=1000.0,
                        help="Scale factor for residual target: (HP-LP)*residual_scale")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--model_type", type=str, default="standard")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to use (for faster experimentation)")
    
    # NEW: Loss function arguments
    parser.add_argument("--loss_type", type=str, default="hesser",
                        choices=["mse", "hesser"],
                        help="Loss type: 'mse' or 'hesser' (weighted MSE + L4)")
    parser.add_argument("--l4_weight", type=float, default=0.1,
                        help="Weight for L4 loss term (default: 0.1)")
    parser.add_argument("--dose_weight_alpha", type=float, default=5.0,
                        help="Alpha for dose-dependent weighting (default: 5.0)")
    parser.add_argument("--gradient_weight", type=float, default=0.0,
                        help="Weight for Sobel gradient loss (default: 0.0)")
    parser.add_argument("--gaussian_sigma", type=float, default=0.8,
                        help="Sigma for Gaussian blur baseline (default: 0.8)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable 3D random flips for training data augmentation (default: True)")
    parser.add_argument("--no_augment", dest="augment", action="store_false",
                        help="Disable data augmentation")
    
    args = parser.parse_args()
    
    train(args)
