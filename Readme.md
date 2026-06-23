# Hybrid DIP + ViT Underwater Segmentation with Collision Alert

Multi-class underwater scene understanding pipeline for autonomous underwater
vehicles (AUVs) and ROVs. Combines classical Digital Image Processing (DIP)
with a Vision Transformer (ViT) backbone for 8-class semantic segmentation,
plus a real-time collision risk assessment system.

## Key Features

### 1. Multi-Class Segmentation (8 SUIM Categories)

| Class | Code | Colour | Collision Risk? |
|-------|------|--------|-----------------|
| Background/Water | BW | Black | No |
| Human Divers | HD | Blue | No |
| Aquatic Plants/Seagrass | PF | Green | Yes |
| Wrecks/Ruins | WR | Cyan | Yes |
| Robots/Instruments | RO | Yellow | No |
| Reefs/Invertebrates | RI | Red | Yes |
| Fish/Vertebrates | FV | Magenta | No |
| Seafloor/Rocks | SR | White | Yes |

### 2. Adaptive DIP Preprocessing (Quality-Aware)

The system automatically analyses each image's degradation level and selects
the optimal preprocessing chain:

- **Colour Cast Index** → selects Gray World, Max-RGB, or Histogram Stretch
- **Noise Level** → selects NL-Means, Bilateral, or Median filter
- **Contrast Ratio** → selects CLAHE, Gamma Correction, or Histogram EQ

### 3. Learnable Enhancement Stem + ViT Encoder

- Lightweight residual CNN block that learns image-space corrections
- Vision Transformer encoder with multi-scale FPN decoder
- End-to-end trainable with combined CE + Dice loss

### 4. Collision Alert System

Zone-based collision risk assessment for underwater drones:

- **Far zone** (top 1/3): distant objects (low weight)
- **Approaching zone** (middle 1/3): objects getting closer
- **Near zone** (bottom 1/3): imminent collision risk (high weight)
- **Centre detection**: objects directly in the drone's path

Risk levels: 🟢 SAFE → 🟡 CAUTION → 🟠 WARNING → 🔴 DANGER

## Repository Structure

```
PROJECT/
├── suim_utils.py              # Shared utilities (SUIM labels, model, DIP, collision)
├── train_multiclass.py        # Training script (8-class segmentation)
├── predict.py                 # Inference + coverage + collision alert
├── cache_dataset.py           # Offline preprocessing cache
├── data/
│   └── dataset/
│       ├── images/            # Input underwater images (*.jpg)
│       └── masks/             # SUIM RGB-encoded masks (*.bmp)
├── TEST/
│   ├── images/
│   └── masks/
├── outputs/
│   ├── best_model_multiclass.pth
│   ├── training_history_multiclass.csv
│   └── predictions/
├── Readme.md
├── requirements.txt
└── research-papers/
```

## Quick Start

### Train

```bash
python train_multiclass.py --epochs 15 --batch 4 --lr 1e-4
```

### Predict

```bash
# Basic segmentation
python predict.py --image path/to/underwater.jpg

# With specific model
python predict.py --image path/to/image.jpg --model outputs/best_model_multiclass.pth
```

### Output Files

For each prediction, the system generates:

| File | Description |
|------|-------------|
| `*_segmentation.png` | 8-class colourised segmentation mask |
| `*_overlay.png` | Segmentation overlay on original image |
| `*_collision_alert.png` | Zone-based collision risk visualisation |
| `*_coverage.png` | Per-class object coverage bar chart |
| `*_full_analysis.jpg` | 6-panel comprehensive comparison |
| `*_report.json` | Complete JSON report with all metrics |

## Technical Details

### Adaptive Preprocessing Decision Logic

| Quality Metric | Threshold | Selected Method |
|----------------|-----------|-----------------|
| Colour cast > 1.5 | Strong B-G dominance | Max-RGB correction |
| Colour cast > 1.2 | Moderate cast | Gray World |
| Colour cast ≤ 1.2 | Mild/none | Histogram Stretch |
| Noise > 20 | Heavy noise | NL-Means denoising |
| Noise > 10 | Moderate | Bilateral filter |
| Noise ≤ 10 | Light | Median filter |
| Contrast < 0.3 | Very flat | CLAHE |
| Contrast < 0.5 | Low | Gamma correction |
| Contrast ≥ 0.5 | Adequate | Histogram EQ |

### Collision Risk Scoring

```
Risk = Σ (zone_weight × class_coverage) + centre_bonus

Zone weights: Far=0.10, Mid=0.30, Near=0.60
Centre bonus: 0.4 × centre_coverage

SAFE     : score < 15
CAUTION  : 15 ≤ score < 35
WARNING  : 35 ≤ score < 60
DANGER   : score ≥ 60
```

## Team Members

| Name | Roll Number |
|------|-------------|
| K V Jaya Harsha | CS23B1034 |
| G Sai Charan Reddy | CS23B1023 |
| Y Santhosh | AD23B1060 |
| Darshan Gowda DS | AD23B1015 |

## License

For academic use only.
