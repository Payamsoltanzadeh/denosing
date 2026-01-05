# ============================================================================
# simple_train_denoising.py
# ============================================================================
# Date: 2025-12-05
# Purpose: Pure PyTorch training script (No Lightning dependencies).
#          Robust and simple training loop for U-Net Denoiser.
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
    
    # Training Dataset
    train_dataset = ConditionalDoseDataset(
        root_dir=args.root_dir,
        normalize=True,
        target_dim=64,  # Resize to 64x64x64 to fit in memory
        num_photons=1e3,
        add_noise=False,
        use_pregenerated_lp=True,
        lp_folder=args.lp_folder,
    )
    
    # Split into Train/Val (Simple split)
    # Using a small subset for validation
    val_size = max(1, int(len(train_dataset) * 0.1))
    train_size = len(train_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"   Train samples: {len(train_set)}")
    print(f"   Val samples:   {len(val_set)}")

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, # Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )

    # ------------------------------------------------------------------------
    # 3. Setup Model, Optimizer, Loss
    # ------------------------------------------------------------------------
    print(f"🏗️ Creating model: {args.model_type}")
    model = get_simple_denoiser(
        model_type=args.model_type,
        base_channels=args.base_channels
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # ------------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------------
    print("🔥 Starting training...")
    best_val_loss = float('inf')

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
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss
            loss = criterion(outputs, hp_target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        # --------------------------------------------------------------------
        # 5. Validation Loop
        # --------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hp_target, condition = batch
                ct = condition['ct'].to(device)
                lp = condition['lp_dose'].to(device)
                hp_target = hp_target.to(device)
                
                inputs = torch.cat([ct, lp], dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, hp_target)
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
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ Saved best model to {save_path}")
        
        # Save last model
        torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))

    print("🏁 Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/simple_unet")
    parser.add_argument("--lp_folder", type=str, default="lp_cubes",
                        help="LP dose folder name (e.g., lp_cubes or lp_cubes_100)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="gpu") # 'gpu' or 'cpu'
    parser.add_argument("--model_type", type=str, default="standard")
    parser.add_argument("--base_channels", type=int, default=32)
    
    args = parser.parse_args()
    
    train(args)
