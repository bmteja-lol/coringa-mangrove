# Comparative Evaluation of ML Algorithms for Coringa Mangrove Mapping

Comparing **Random Forest**, **XGBoost**, and **DeepLabV3+** for mangrove forest classification using satellite imagery and spectral indices.

## Study Area

**Coringa Wildlife Sanctuary (CWS)** — Andhra Pradesh, India.

- Location: 16°42'–17°0'N, 82°13'–82°23'E
- Second largest mangrove formation on India's east coast
- Situated within the estuarine forest area of the Godavari River in Kakinada Bay
- Dominant species: *Avicennia marina*, *Rhizophora apiculata*, *Sonneratia apetala*, *Excoecaria agallocha*

## Background

Mangrove forests are salt-tolerant woody plants that thrive in intertidal habitats. They provide critical ecosystem services including coastal protection, carbon sequestration (3–5× greater than tropical upland forests), and breeding grounds for terrestrial and aquatic species.

Traditional field-based surveys are impractical for monitoring expansive mangrove areas. Satellite remote sensing combined with machine learning enables accurate, scalable mangrove mapping. This project extends the methodology from Sowjanya & Prasad (2025) by adding XGBoost and DeepLabV3+ to the original RF/SVM comparison.

## Methodology

### Satellite Data

| Sensor | Dataset | Resolution | Period |
|---|---|---|---|
| Sentinel-2 MSI | `COPERNICUS/S2_SR_HARMONIZED` | 10m (VIS/NIR), 20m (SWIR) | Sep–Dec 2022 |
| Landsat-8 OLI | `LANDSAT/LC08/C02/T1_L2` | 30m | Sep–Dec 2017 |
| ESA WorldCover | `ESA/WorldCover/v200/2021` | 10m | 2021 (labels) |

All imagery accessed and processed via Google Earth Engine (GEE). Cloud masking and median compositing applied.

### Spectral Indices

Five spectral indices derived from satellite bands:

| Index | Formula | Role |
|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation density, chlorophyll absorption |
| NDMI | (NIR − SWIR1) / (NIR + SWIR1) | Vegetation water content, moisture |
| MNDWI | (Green − SWIR1) / (Green + SWIR1) | Tidal inundation detection |
| GCVI | NIR / Green − 1 | Green leaf biomass |
| SR | NIR / Red | Simple ratio, vegetation density |

### Customized Thresholds (CT)

Pixel value ranges specific to Coringa mangroves, determined by inspecting spectral index maps via GEE inspector tool:

| Index | CT Range | Reference |
|---|---|---|
| NDVI | ≥ 0.4 | Kuenzer et al. (2011); Wang et al. (2019) |
| NDMI | −0.34 to −0.14 | Zhang et al. (2017); Shi et al. (2016) |
| MNDWI | −0.35 to −0.14 | Hickey & Radford (2022); Xu (2006) |
| GCVI | ≥ 3.7 | Giri et al. (2011) |
| SR | 6.5–11 | Hickey & Radford (2022) |

### Models

| Model | Type | Key Params | Source |
|---|---|---|---|
| Random Forest | Pixel-level ensemble | 100 trees, 5 features/split | `sklearn` |
| XGBoost | Pixel-level gradient boosting | 200 trees, depth 6, lr 0.1 | `xgboost` |
| DeepLabV3+ | Patch-level deep segmentation | EfficientNet-B0 encoder, focal loss | `segmentation_models_pytorch` |

**RF and XGBoost** operate on flattened pixel data (11 spectral channels per pixel).

**DeepLabV3+** operates on spatial patches (128×128 for S2, 64×64 for L8), learning spatial context.

### Training

- Labels: ESA WorldCover 2021 (class 95 = mangroves)
- Patch extraction: 1000 random patches per sensor
- Train/Val/Test split: 70/15/15
- DL loss: Focal Loss (α=0.75, γ=2.0) — handles class imbalance
- DL optimizer: Adam, lr=1e-4, 15 epochs

## Results Summary

From the reference paper (RF + SVM comparison):

| Method | Training Acc | Testing Acc |
|---|---|---|
| RF + Standard Threshold | 100% | 41–100% |
| RF + Customized Threshold | 100% | 85–100% **(best)** |
| SVM + Standard Threshold | 93–100% | 77–100% |
| SVM + Customized Threshold | 96–100% | 63–100% |

This project adds XGBoost and DeepLabV3+ for a more comprehensive comparison.

## Project Structure

```
├── configs/constants.py        # All parameters in one place
├── src/
│   ├── data/
│   │   ├── loader.py           # GEE image loading (S2, L8, WorldCover)
│   │   ├── indices.py          # Spectral index computation
│   │   └── patches.py          # Patch extraction + normalization
│   ├── models/
│   │   ├── random_forest.py    # RF training + evaluation
│   │   ├── xgboost_model.py    # XGBoost training + evaluation
│   │   └── deeplabv3.py        # DeepLabV3+ model + training loop
│   ├── evaluate.py             # IoU, accuracy, confusion matrix, plots
│   └── train.py                # Pipeline orchestrator
├── scripts/run_all.py          # CLI entry point
├── notebooks/                  # Jupyter notebooks
└── results/figures/            # Output plots
```

## Setup (VS Code)

### Prerequisites

- Python 3.9+
- Google Earth Engine account ([signup](https://earthengine.google.com/signup/))

### Installation

```bash
# Clone the repo
git clone https://github.com/bmteja-lol/coringa-mangrove.git
cd coringa-mangrove

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install earthengine-api geemap numpy pandas scikit-learn xgboost torch torchvision segmentation-models-pytorch matplotlib seaborn jupyter

# Authenticate with Earth Engine (one-time)
earthengine authenticate
```

### VS Code Extensions (recommended)

- Python (Microsoft)
- Jupyter
- Pylance

## Usage

### Notebooks

```bash
# Launch Jupyter from project root
jupyter notebook notebooks/
```

Open any notebook and run all cells.

### CLI

```bash
# Full comparison (RF + XGBoost + DeepLabV3+) on Sentinel-2
python scripts/run_all.py --sensor s2 --method all

# Just RF on Landsat-8
python scripts/run_all.py --sensor l8 --method rf

# Just DeepLabV3+ on Sentinel-2
python scripts/run_all.py --sensor s2 --method dl
```

## Key Findings

1. **Threshold selection matters more than model choice** — RF with customized thresholds outperforms SVM with any threshold
2. **RF is robust** — consistent 85–100% testing accuracy across all years when using CT
3. **SVM struggles with spectral similarity** — misclassifies creeks/rivers as mangrove
4. **Deep learning captures spatial context** — but requires more data and compute

## References

- Sowjanya D S and Prasad P R C (2025) Comparative evaluation of machine learning algorithms for Coringa Mangroves mapping with satellite imagery and spectral indices. *J. Earth Syst. Sci.* 134:1
- Hickey S M and Radford B (2022) Turning the tide on mapping marginal mangroves with multi-dimensional space–time remote sensing. *Remote Sens.* 14(14) 3365
- Kuenzer C et al. (2011) Remote sensing of mangrove ecosystems: A review. *Remote Sens.* 3(5) 878–928
- Tran T V et al. (2022) A review of spectral indices for mangrove remote sensing. *Remote Sens.* 14(19) 4868
