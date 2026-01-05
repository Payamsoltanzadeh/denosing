# MC Dose Denoising - Training Package

**Clean training package for Monte Carlo dose denoising with corrected Poisson noise formula (per Prof. Hesser, Dec 15, 2025)**

---

## 📦 Package Structure

```
MC_Denoising_Training/
├── simple_train_denoising.py    # Main training script
├── test_denoising_hesser.py     # Evaluation with Hesser's metrics
├── generate_lp_dose.py          # Pre-generate Low-Photon dose (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── dataset/
│   ├── __init__.py
│   └── pl_dose_dataset.py       # Dataset loader + Poisson formula
├── models/
│   ├── __init__.py
│   └── simple_unet_denoiser.py  # 3D U-Net architecture
└── utils/
    ├── __init__.py
    └── gamma_index.py           # Gamma Index metric (clinical)
```

---

## 🎯 What This Package Does

**Goal:** Train a 3D U-Net to denoise Low-Photon (LP) Monte Carlo dose distributions to High-Photon (HP) quality.

**Input Data Required:**
- `input_cubes/` → CT volumes (anatomy)
- `output_cubes/` → HP dose (clean, from Geant4 simulation)

**LP Generation:** Low-Photon (noisy) dose is generated on-the-fly using Prof. Hesser's Poisson formula:
```
N = D / δ           (calculate number of particles)
N' ~ Poisson(N)     (sample from Poisson distribution)
D' = N' × δ         (convert back to dose)
```

Where δ (delta) is auto-calculated to achieve target uncertainty (default: 10%).

---

## ⚙️ Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Training

```bash
python simple_train_denoising.py \
  --root_dir "path/to/46_53-32_cube/46_53" \
  --model_type standard \
  --epochs 50 \
  --batch_size 4 \
  --device gpu \
  --save_dir results/unet_training
```

**Arguments:**
- `--root_dir`: Path to data folder (with `output/<patient_id>/input_cubes` and `output_cubes`)
- `--model_type`: `standard` (direct HP prediction) or `residual` (predict correction)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (reduce if GPU memory is limited)
- `--device`: `gpu` or `cpu`
- `--target_uncertainty`: Target noise level (default: 0.10 = 10%)

### 2. Evaluation

```bash
python test_denoising_hesser.py \
  --root_dir "path/to/46_53-32_cube/46_53" \
  --model_path results/unet_training/best_model.pth \
  --model_type standard \
  --num_samples 50 \
  --output_dir results/evaluation
```

**Metrics Computed:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- PSNR (Peak Signal-to-Noise Ratio)
- Gamma Index (3%/3mm, 2%/2mm, 1%/1mm)
- Dose regime analysis (Low/Mid/High dose regions)

### 3. Pre-generate LP Data (Optional)

For reproducibility or faster training:

```bash
python generate_lp_dose.py \
  --root_dir "path/to/46_53-32_cube/46_53" \
  --target_uncertainty 0.10 \
  --output_folder lp_cubes_10pct
```

Then update training script to use `--use_pregenerated_lp --lp_folder lp_cubes_10pct`

---

## 📊 Expected Results

**With 32³ cubes (20,000 samples) and 10% noise:**
- Training time: ~2-4 hours (GPU) or ~8-12 hours (CPU)
- Expected RMSE: ~0.00003-0.00005 (physical units)
- Expected Gamma 3%/3mm: >95%
- Expected PSNR: >35 dB

---

## 🔧 Key Implementation Details

### Poisson Noise (Prof. Hesser's Formula)

Located in `dataset/pl_dose_dataset.py`:

```python
def add_poisson_noise(clean_dose, delta=None, target_uncertainty=0.10):
    if delta is None:
        max_dose = clean_dose.max()
        target_particles = (1.0 / target_uncertainty) ** 2
        delta = max_dose / target_particles
    
    N_particles = clean_dose / delta
    N_noisy = np.random.poisson(N_particles)
    noisy_dose = N_noisy * delta
    
    return noisy_dose
```

**Physical Interpretation:**
- For max_dose = 0.0027 Gy and target_uncertainty = 0.10:
  - δ ≈ 2.7e-5 Gy (27 µGy)
  - High dose: ~100 particles → 10% uncertainty ✓
  - Mid dose: ~40 particles → 16% uncertainty
  - Low dose: ~10 particles → 32% uncertainty

### Model Architecture

3D U-Net with:
- Encoder: 2 levels (32 → 64 channels)
- Bottleneck: 128 channels
- Decoder: 2 levels (64 → 32 channels)
- Skip connections between encoder and decoder
- Total parameters: ~1.4M

---

## 📝 Citation

If using this code, please cite:
- Prof. Hesser's Poisson noise formulation (Dec 15, 2025 meeting)
- Marcus's Geant4 simulation data (Heidelberg University)

---

## 📧 Contact

For questions about:
- **Code/Training**: Payam (this package)
- **Data/Simulation**: Marcus Buchwalder (marcus.buchwalder@...)
- **Physics/Methodology**: Prof. Hesser

---

## ✅ Verification

Test that all imports work:

```bash
cd MC_Denoising_Training
python -c "from dataset.pl_dose_dataset import add_poisson_noise; from models.simple_unet_denoiser import UNet3D; from utils.gamma_index import gamma_index; print('✅ All imports OK!')"
```

---

**Package Version:** 1.0 (January 5, 2026)  
**Corrected Poisson formula per Prof. Hesser's Dec 15, 2025 feedback**
