# SAR to EO Image Translation using CycleGAN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Team Members
- **Taksh Pal** (24/B03/048)
- **Ritij Raj** (24/B03/015)

## Project Overview

This project implements a **Cycle-Consistent Generative Adversarial Network (CycleGAN)** for translating **Synthetic Aperture Radar (SAR)** satellite images to **Electro-Optical (EO)** images. The goal is to generate visually realistic EO images from SAR data, which is often more available in cloudy or night-time conditions where optical sensors fail.

## Key Features

- ✅ **CycleGAN architecture**: Unpaired image-to-image translation without requiring one-to-one image mapping
- ✅ **Multi-channel support**: 
  - RGB output (Red, Green, Blue channels)
  - RGBNIR output (Red, Green, Blue, Near-Infrared channels)  
  - SWNIR output (Short Wave Near-Infrared channels)
- ✅ **True-color visualization**: Uses standard RGB bands (B4, B3, B2) to replicate natural color images
- ✅ **PSNR & SSIM Evaluation**: Quantitative metrics for image quality comparison
- ✅ **Custom dataloader**: Handles 2-channel SAR and multi-channel EO imagery
- ✅ **Training on Kaggle GPUs**: Easily reproducible on Kaggle platform

## Project Structure

```
SAR-to-EO-Translation/
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── .gitignore
├── LICENSE
├── SAR_TOEO(RGB)/
│   └── summerschool.ipynb
├── SAR_TO_EO(RGBNIR)/
│   └── summer-school-task-3.ipynb
├── SAR_TO_EO(SWNIR)/
│   └── school2.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cyclegan.py
│   │   ├── generator.py
│   │   └── discriminator.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── loss.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── data/
│   ├── processed_sar/
│   ├── processed_eo/
│   └── raw/
├── models/
│   └── checkpoints/
└── results/
    ├── generated_images/
    └── evaluation_metrics/
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8+ GB RAM

### Software Dependencies
See `requirements.txt` for complete list of dependencies.

## Installation

### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/SAR-to-EO-Translation.git
cd SAR-to-EO-Translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Clone the repository
git clone https://github.com/yourusername/SAR-to-EO-Translation.git
cd SAR-to-EO-Translation

# Create conda environment
conda env create -f environment.yml
conda activate sar-to-eo
```

### Option 3: Direct installation
```bash
pip install -e .
```

## Usage

### 1. Data Preprocessing
```python
from src.data.preprocessing import preprocess_data

# Preprocess SAR and EO images
preprocess_data(
    sar_folder="data/raw/s1",
    eo_folder="data/raw/s2", 
    output_sar="data/processed_sar",
    output_eo="data/processed_eo",
    target_size=(256, 256)
)
```

### 2. Training the Model
```python
from src.training.trainer import CycleGANTrainer

trainer = CycleGANTrainer(
    sar_path="data/processed_sar",
    eo_path="data/processed_eo",
    batch_size=16,
    num_epochs=200
)

trainer.train()
```

### 3. Evaluation
```python
from src.evaluation.metrics import evaluate_model

# Calculate PSNR and SSIM metrics
psnr, ssim = evaluate_model(
    model_path="models/checkpoints/cyclegan_latest.pth",
    test_data_path="data/test"
)
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
```

### 4. Running Jupyter Notebooks

The project includes three main notebooks for different output configurations:

- **RGB Output**: `SAR_TOEO(RGB)/summerschool.ipynb`
- **RGBNIR Output**: `SAR_TO_EO(RGBNIR)/summer-school-task-3.ipynb`
- **SWNIR Output**: `SAR_TO_EO(SWNIR)/school2.ipynb`

```bash
jupyter notebook
```

## Model Architecture

The CycleGAN consists of:

- **Two Generators**: 
  - G_SAR→EO: Translates SAR images to EO images
  - G_EO→SAR: Translates EO images to SAR images

- **Two Discriminators**:
  - D_SAR: Distinguishes real vs fake SAR images
  - D_EO: Distinguishes real vs fake EO images

### Loss Functions
- **Adversarial Loss**: Standard GAN loss for realistic image generation
- **Cycle Consistency Loss**: Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
- **Identity Loss**: Preserves color composition when possible

## Data Format

### Input (SAR Images)
- **Format**: GeoTIFF (.tif)
- **Channels**: 2 (VV and VH polarizations)
- **Resolution**: Variable (resized to 256x256 for training)

### Output (EO Images) 
- **Format**: GeoTIFF (.tif)
- **Channels**: 
  - RGB: 3 channels (Red, Green, Blue)
  - RGBNIR: 4 channels (Red, Green, Blue, Near-Infrared)
  - SWNIR: Multiple channels including Short Wave Near-Infrared
- **Resolution**: Variable (resized to 256x256 for training)

## Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)
Measures image fidelity and reconstruction quality.
- **Range**: 0 to ∞ (higher is better)
- **Typical Good Values**: > 20 dB

### SSIM (Structural Similarity Index)
Captures perceptual similarity between images.
- **Range**: -1 to 1 (higher is better)
- **Typical Good Values**: > 0.8

## Results

The model generates realistic EO images from SAR inputs with:
- Improved visual quality compared to raw SAR images
- Preservation of geographical features and structures
- Effective translation across different spectral configurations

## Training Details

- **Dataset Size**: 200 image pairs (can be extended)
- **Image Size**: 256x256 pixels
- **Batch Size**: 16
- **Epochs**: 200
- **Optimizer**: Adam (lr=0.0002, β1=0.5, β2=0.999)
- **Hardware**: CUDA-compatible GPU recommended

## Limitations

- Requires substantial computational resources for training
- Performance depends on quality and diversity of training data
- Generated images may not perfectly preserve fine-grained details
- Domain gap between SAR and EO modalities can affect translation quality

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Original CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- PyTorch CycleGAN implementation: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Sentinel-1 and Sentinel-2 satellite data from ESA

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sar-to-eo-cyclegan,
  title={SAR to EO Image Translation using CycleGAN},
  author={Taksh Pal and Ritij Raj},
  year={2025},
  url={https://github.com/yourusername/SAR-to-EO-Translation}
}
```

## Contact

- **Taksh Pal**: [email@example.com]
- **Ritij Raj**: [email@example.com]

For questions and support, please open an issue on GitHub.
