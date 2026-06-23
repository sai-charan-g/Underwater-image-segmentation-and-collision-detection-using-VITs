# ============================================================
# suim_utils.py — SUIM Dataset Utilities + Adaptive Preprocessing
# Shared across train_multiclass.py and predict.py
# ============================================================

import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ─────────────────────────────────────────────────────────────
# SUIM 8-CLASS LABEL SYSTEM (RGB colour encoding)
# ─────────────────────────────────────────────────────────────
# The SUIM dataset encodes classes as RGB colours in the mask.
# Due to BMP/JPEG artifacts the colours are not always exact,
# so we use nearest-colour matching with a tolerance.

NUM_CLASSES = 8
IMG_SIZE    = 224

SUIM_CLASSES = {
    0: {"name": "Background/Water",       "short": "BW", "rgb": (0,   0,   0  )},
    1: {"name": "Human Divers",           "short": "HD", "rgb": (0,   0,   255)},
    2: {"name": "Aquatic Plants/Seagrass", "short": "PF", "rgb": (0,   255, 0  )},
    3: {"name": "Wrecks/Ruins",           "short": "WR", "rgb": (0,   255, 255)},
    4: {"name": "Robots/Instruments",     "short": "RO", "rgb": (255, 255, 0  )},
    5: {"name": "Reefs/Invertebrates",    "short": "RI", "rgb": (255, 0,   0  )},
    6: {"name": "Fish/Vertebrates",       "short": "FV", "rgb": (255, 0,   255)},
    7: {"name": "Seafloor/Rocks",         "short": "SR", "rgb": (255, 255, 255)},
}

# Pre-build a lookup array  (8 x 3) for vectorised distance
_CLASS_RGB = np.array([SUIM_CLASSES[i]["rgb"] for i in range(NUM_CLASSES)],
                      dtype=np.float32)

# Colours for overlay visualisation (BGR for OpenCV)
CLASS_COLORS_BGR = [
    (0,   0,   0  ),   # BW — black
    (255, 0,   0  ),   # HD — blue
    (0,   255, 0  ),   # PF — green
    (255, 255, 0  ),   # WR — cyan
    (0,   255, 255),   # RO — yellow  (BGR)
    (0,   0,   255),   # RI — red
    (255, 0,   255),   # FV — magenta
    (255, 255, 255),   # SR — white
]

# Collision-relevant classes: objects that a drone can collide with
COLLISION_CLASSES = {
    2: "Aquatic Plants",    # PF
    3: "Wrecks/Ruins",      # WR
    5: "Reefs/Invertebrates",# RI
    7: "Seafloor/Rocks",    # SR
}

CLASS_NAMES = [SUIM_CLASSES[i]["name"] for i in range(NUM_CLASSES)]
CLASS_SHORTS = [SUIM_CLASSES[i]["short"] for i in range(NUM_CLASSES)]


def rgb_mask_to_class_index(mask_rgb):
    """
    Convert an RGB-encoded SUIM mask (H, W, 3) to a class-index map (H, W).
    Uses nearest-colour matching to handle compression artifacts.
    """
    H, W, _ = mask_rgb.shape
    flat = mask_rgb.reshape(-1, 3).astype(np.float32)  # (N, 3)

    # Compute squared L2 distance to each class colour
    # flat: (N, 3),  _CLASS_RGB: (8, 3)  →  dists: (N, 8)
    dists = np.sum((flat[:, None, :] - _CLASS_RGB[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).reshape(H, W).astype(np.uint8)
    return labels


def class_index_to_color(label_map):
    """Convert class-index map (H, W) to colourised BGR image (H, W, 3)."""
    H, W = label_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        mask = (label_map == cls_id)
        out[mask] = CLASS_COLORS_BGR[cls_id]
    return out


# ─────────────────────────────────────────────────────────────
# OPTION C: UNDERWATER IMAGE QUALITY ASSESSMENT + ADAPTIVE DIP
# ─────────────────────────────────────────────────────────────

def assess_underwater_quality(img_bgr):
    """
    Compute degradation-specific quality metrics for an underwater image.

    Returns dict with:
      - color_cast: ratio of (B+G)/(2R) — higher = stronger blue-green cast
      - blur_score: Laplacian variance — lower = blurrier
      - contrast:   std/mean of grayscale — lower = flatter
      - noise:      mean absolute Laplacian — higher = noisier
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    color_cast = (b.mean() + g.mean()) / (2 * r.mean() + 1e-8)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast   = gray.std() / (gray.mean() + 1e-8)
    noise      = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).mean()

    return {
        "color_cast": float(color_cast),
        "blur_score": float(blur_score),
        "contrast":   float(contrast),
        "noise":      float(noise),
    }


# --- DIP preprocessing functions ---

def gray_world(img):
    """Gray World colour correction."""
    img = img.astype(np.float32)
    avg = img.mean()
    for c in range(3):
        ch_avg = img[:, :, c].mean()
        img[:, :, c] = img[:, :, c] * (avg / (ch_avg + 1e-8))
    return np.clip(img, 0, 255).astype(np.uint8)


def max_rgb(img):
    """White-Patch / Max-RGB colour correction."""
    img = img.astype(np.float32)
    for c in range(3):
        pmax = np.percentile(img[:, :, c], 98)
        img[:, :, c] = img[:, :, c] / (pmax + 1e-8) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def linear_hist_stretch(img):
    """Linear histogram stretching (2nd/98th percentile)."""
    from skimage import exposure
    out = img.copy()
    for c in range(3):
        p2, p98 = np.percentile(out[:, :, c], (2, 98))
        out[:, :, c] = exposure.rescale_intensity(
            out[:, :, c], in_range=(p2, p98)
        )
    return out


def median_denoise(img, k=5):
    return cv2.medianBlur(img, k)

def bilateral_denoise(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def nl_means_denoise(img, h=10):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def global_hist_eq(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def clahe_enhance(img, clipLimit=2.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def gamma_correction(img, gamma=1.2):
    inv = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def preprocess_pipeline(img_bgr, cfg):
    """Apply DIP preprocessing chain specified by *cfg* dict."""
    out = img_bgr.copy()

    # Colour correction
    cc = cfg.get("color", "grayworld")
    if cc == "grayworld":
        out = gray_world(out)
    elif cc == "max_rgb":
        out = max_rgb(out)
    elif cc == "stretch":
        out = linear_hist_stretch(out)

    # Denoising
    dn = cfg.get("denoise", "nl_means")
    if dn == "median":
        out = median_denoise(out, k=5)
    elif dn == "bilateral":
        out = bilateral_denoise(out)
    elif dn == "nl_means":
        out = nl_means_denoise(out, h=10)

    # Contrast enhancement
    ct = cfg.get("contrast", "clahe")
    if ct == "hist":
        out = global_hist_eq(out)
    elif ct == "clahe":
        out = clahe_enhance(out)
    elif ct == "gamma":
        out = gamma_correction(out, gamma=1.2)

    return out


def preprocess_pipeline_steps(img_bgr, cfg):
    """Apply DIP preprocessing and return named intermediate images."""
    steps = [("Original", img_bgr.copy())]
    out = img_bgr.copy()

    # Colour correction
    cc = cfg.get("color", "grayworld")
    if cc == "grayworld":
        out = gray_world(out)
    elif cc == "max_rgb":
        out = max_rgb(out)
    elif cc == "stretch":
        out = linear_hist_stretch(out)
    steps.append((f"After color correction ({cc})", out.copy()))

    # Denoising
    dn = cfg.get("denoise", "nl_means")
    if dn == "median":
        out = median_denoise(out, k=5)
    elif dn == "bilateral":
        out = bilateral_denoise(out)
    elif dn == "nl_means":
        out = nl_means_denoise(out, h=10)
    steps.append((f"After denoising ({dn})", out.copy()))

    # Contrast enhancement
    ct = cfg.get("contrast", "clahe")
    if ct == "hist":
        out = global_hist_eq(out)
    elif ct == "clahe":
        out = clahe_enhance(out)
    elif ct == "gamma":
        out = gamma_correction(out, gamma=1.2)
    steps.append((f"After contrast enhancement ({ct})", out.copy()))

    return steps


def adaptive_preprocess(img_bgr):
    """
    Automatically select the best DIP config based on image quality metrics.
    Returns (preprocessed_image, chosen_config, quality_metrics).
    """
    q = assess_underwater_quality(img_bgr)

    cfg = {}

    # Adaptive colour correction
    if q["color_cast"] > 1.5:
        cfg["color"] = "max_rgb"       # strong blue-green cast
    elif q["color_cast"] > 1.2:
        cfg["color"] = "grayworld"     # moderate cast
    else:
        cfg["color"] = "stretch"       # minor cast

    # Adaptive denoising
    if q["noise"] > 20:
        cfg["denoise"] = "nl_means"    # heavy noise
    elif q["noise"] > 10:
        cfg["denoise"] = "bilateral"   # moderate noise
    else:
        cfg["denoise"] = "median"      # light noise

    # Adaptive contrast
    if q["contrast"] < 0.3:
        cfg["contrast"] = "clahe"      # very flat → local enhancement
    elif q["contrast"] < 0.5:
        cfg["contrast"] = "gamma"      # low contrast → brightening
    else:
        cfg["contrast"] = "hist"       # already reasonable

    return preprocess_pipeline(img_bgr, cfg), cfg, q


# ─────────────────────────────────────────────────────────────
# MULTI-CLASS ViT SEGMENTATION MODEL
# ─────────────────────────────────────────────────────────────

class LearnableEnhancementStem(nn.Module):
    """Lightweight residual CNN that learns image-space corrections."""
    def __init__(self, channels=3, hidden=24, strength=0.1):
        super().__init__()
        self.strength = strength
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        return x + self.strength * self.net(x)


class ViTMultiClassSegmentation(nn.Module):
    """
    Multi-class semantic segmentation model:
      LearnableEnhancementStem → ViT encoder → FPN-style decoder → 8-class output
    """
    def __init__(self, vit_name="vit_small_patch16_224", pretrained=True,
                 num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.enhancement = LearnableEnhancementStem()

        self.vit = timm.create_model(
            vit_name, pretrained=pretrained,
            num_classes=0, global_pool=""
        )
        self.patch_size = getattr(self.vit, "patch_size", 16)
        self.embed_dim  = self.vit.embed_dim
        self.n_patches  = img_size // self.patch_size

        # Register hooks for mid-level and final features
        self._feats = {}
        n_blocks = len(self.vit.blocks)
        mid = n_blocks // 2
        self.vit.blocks[mid - 1].register_forward_hook(self._hook("mid"))
        self.vit.blocks[-1].register_forward_hook(self._hook("final"))

        # Lateral connections (FPN-style)
        self.lat_mid   = nn.Conv2d(self.embed_dim, 256, 1)
        self.lat_final = nn.Conv2d(self.embed_dim, 256, 1)

        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, num_classes, 1),
        )

    def _hook(self, name):
        def fn(mod, inp, out):
            self._feats[name] = out
        return fn

    def _to_map(self, tokens):
        B, N, C = tokens.shape
        if N == self.n_patches ** 2 + 1:
            tokens = tokens[:, 1:]          # drop CLS token
        H = W = int(tokens.shape[1] ** 0.5)
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x):
        B, _, H, W = x.shape
        self._feats = {}
        x = self.enhancement(x)
        self.vit.forward_features(x)

        f_mid   = self._to_map(self._feats["mid"])
        f_final = self._to_map(self._feats["final"])

        p = self.lat_final(f_final) + F.interpolate(
            self.lat_mid(f_mid), size=f_final.shape[-2:], mode="nearest"
        )
        out = self.decoder(p)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────
# COLLISION ALERT SYSTEM
# ─────────────────────────────────────────────────────────────

def compute_coverage(pred_labels, num_classes=NUM_CLASSES):
    """
    Compute per-class pixel coverage as a percentage.
    pred_labels: (H, W) numpy array with class indices 0..7
    Returns dict {class_id: coverage_pct}
    """
    total = pred_labels.size
    coverage = {}
    for cls_id in range(num_classes):
        count = (pred_labels == cls_id).sum()
        coverage[cls_id] = float(count) / total * 100.0
    return coverage


def compute_zone_coverage(pred_labels, num_classes=NUM_CLASSES):
    """
    Divide the image into 3 horizontal zones and compute per-class coverage.
      - Top zone    (0-33%):    far away objects
      - Middle zone (33-66%):   approaching objects
      - Bottom zone (66-100%):  imminent collision risk

    Returns dict {zone_name: {class_id: coverage_pct}}
    """
    H, W = pred_labels.shape
    zones = {
        "far":      pred_labels[:H // 3, :],
        "mid":      pred_labels[H // 3: 2 * H // 3, :],
        "near":     pred_labels[2 * H // 3:, :],
    }
    result = {}
    for zone_name, zone_pixels in zones.items():
        total = zone_pixels.size
        zone_cov = {}
        for cls_id in range(num_classes):
            count = (zone_pixels == cls_id).sum()
            zone_cov[cls_id] = float(count) / total * 100.0
        result[zone_name] = zone_cov
    return result


def compute_center_coverage(pred_labels, num_classes=NUM_CLASSES, margin=0.2):
    """
    Compute coverage in the central region of the frame.
    The drone's path is usually straight ahead → centre is the risk area.
    """
    H, W = pred_labels.shape
    y1, y2 = int(H * margin), int(H * (1 - margin))
    x1, x2 = int(W * margin), int(W * (1 - margin))
    centre = pred_labels[y1:y2, x1:x2]
    total = centre.size
    coverage = {}
    for cls_id in range(num_classes):
        count = (centre == cls_id).sum()
        coverage[cls_id] = float(count) / total * 100.0
    return coverage


def collision_alert(pred_labels):
    """
    Assess collision risk for an underwater drone based on segmentation output.

    Risk scoring:
      - High weight on objects in bottom zone (near) and centre
      - Only solid objects (reefs, rocks, wrecks, plants) trigger alerts
      - Water/background is safe

    Returns dict with:
      - overall_risk_score:  0–100 (0 = safe, 100 = imminent collision)
      - risk_level:          "SAFE" / "CAUTION" / "WARNING" / "DANGER"
      - per_class_risk:      risk contribution from each obstacle class
      - zone_breakdown:      coverage in far/mid/near zones
      - dominant_obstacle:   name of the class posing most risk
      - recommended_action:  human-readable guidance string
    """
    # Full-frame coverage
    full_cov = compute_coverage(pred_labels)

    # Zone-based coverage
    zone_cov = compute_zone_coverage(pred_labels)

    # Centre coverage
    centre_cov = compute_center_coverage(pred_labels)

    # Zone weight factors (near + centre = highest risk)
    zone_weights = {"far": 0.10, "mid": 0.30, "near": 0.60}

    per_class_risk = {}
    for cls_id in COLLISION_CLASSES:
        # Weighted zone score
        zone_score = sum(
            zone_cov[z].get(cls_id, 0) * w
            for z, w in zone_weights.items()
        )
        # Centre bonus (objects directly ahead are extra dangerous)
        centre_bonus = centre_cov.get(cls_id, 0) * 0.4
        per_class_risk[cls_id] = min(zone_score + centre_bonus, 100.0)

    # Overall risk = sum of individual class risks, capped at 100
    overall_risk = min(sum(per_class_risk.values()), 100.0)

    # Determine risk level
    if overall_risk < 15:
        risk_level = "SAFE"
        action = "All clear — continue current trajectory."
    elif overall_risk < 35:
        risk_level = "CAUTION"
        action = "Obstacles detected at distance — monitor trajectory."
    elif overall_risk < 60:
        risk_level = "WARNING"
        action = "Obstacles approaching — consider course adjustment."
    else:
        risk_level = "DANGER"
        action = "IMMINENT COLLISION RISK — execute evasive manoeuvre!"

    # Find dominant obstacle
    dominant_cls = max(per_class_risk, key=per_class_risk.get) if per_class_risk else None
    dominant_name = COLLISION_CLASSES.get(dominant_cls, "None") if dominant_cls else "None"

    return {
        "overall_risk_score": round(overall_risk, 1),
        "risk_level":         risk_level,
        "per_class_risk":     {COLLISION_CLASSES[k]: round(v, 1) for k, v in per_class_risk.items()},
        "per_class_coverage": {CLASS_NAMES[k]: round(v, 1) for k, v in full_cov.items()},
        "zone_breakdown":     {z: {CLASS_NAMES[k]: round(v, 1) for k, v in zc.items()} for z, zc in zone_cov.items()},
        "dominant_obstacle":  dominant_name,
        "recommended_action": action,
    }
