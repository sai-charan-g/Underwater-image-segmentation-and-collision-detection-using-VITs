# ============================================================
# predict.py — Multi-Class Underwater Segmentation + Collision Alert
#
# Features:
#   ✓ 8-class SUIM semantic segmentation
#   ✓ Adaptive DIP preprocessing (quality-aware)
#   ✓ Per-class object coverage analysis
#   ✓ Zone-based collision risk assessment
#   ✓ Colourised segmentation overlays
#   ✓ Detailed JSON report with all metrics
#
# Usage:
#   python predict.py --image path/to/your/image.jpg
#   python predict.py --image path/to/your/image.jpg --alert
#   python predict.py --image path/to/your/image.jpg --model outputs/best_model_multiclass.pth
# ============================================================

import os
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import autocast

from suim_utils import (
    NUM_CLASSES, IMG_SIZE, CLASS_NAMES, CLASS_SHORTS,
    SUIM_CLASSES, CLASS_COLORS_BGR, COLLISION_CLASSES,
    class_index_to_color, adaptive_preprocess, assess_underwater_quality,
    preprocess_pipeline_steps,
    ViTMultiClassSegmentation,
    compute_coverage, compute_zone_coverage, compute_center_coverage,
    collision_alert,
)

# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model_multiclass.pth")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "predictions")
VIT_NAME   = "vit_small_patch16_224"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# COLOUR LEGEND BAR
# ─────────────────────────────────────────────────────────────
def draw_legend(width=800, bar_h=30):
    """Create a colour legend bar showing all 8 SUIM classes."""
    legend = np.zeros((bar_h, width, 3), dtype=np.uint8)
    n = NUM_CLASSES
    seg_w = width // n
    for i in range(n):
        x1 = i * seg_w
        x2 = (i + 1) * seg_w if i < n - 1 else width
        legend[:, x1:x2] = CLASS_COLORS_BGR[i]
        # Put class short name
        cv2.putText(legend, CLASS_SHORTS[i],
                    (x1 + 4, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255) if i != 7 else (0, 0, 0),
                    1, cv2.LINE_AA)
    return legend


# ─────────────────────────────────────────────────────────────
# COLLISION RISK OVERLAY
# ─────────────────────────────────────────────────────────────
def draw_collision_overlay(image_bgr, pred_labels, alert_info):
    """
    Draw collision risk zones and alert level on the image.
    """
    overlay = image_bgr.copy()
    H, W = overlay.shape[:2]

    # Draw zone boundaries
    zone_y = [0, H // 3, 2 * H // 3, H]
    zone_names = ["FAR", "APPROACHING", "NEAR"]
    zone_colors = [(0, 180, 0), (0, 200, 255), (0, 0, 255)]  # green, orange, red

    for i in range(3):
        y1, y2 = zone_y[i], zone_y[i + 1]
        # Semi-transparent zone highlight
        zone_overlay = overlay.copy()
        cv2.rectangle(zone_overlay, (0, y1), (W, y2), zone_colors[i], -1)
        overlay = cv2.addWeighted(overlay, 0.85, zone_overlay, 0.15, 0)
        # Zone label
        cv2.putText(overlay, zone_names[i],
                    (5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, zone_colors[i], 1, cv2.LINE_AA)

    # Risk level badge
    risk = alert_info["risk_level"]
    score = alert_info["overall_risk_score"]

    badge_colors = {
        "SAFE":    (0, 200, 0),
        "CAUTION": (0, 200, 255),
        "WARNING": (0, 140, 255),
        "DANGER":  (0, 0, 255),
    }
    badge_color = badge_colors.get(risk, (255, 255, 255))

    # Badge background
    badge_w = 250
    badge_h = 55
    cv2.rectangle(overlay, (W - badge_w - 10, 5),
                  (W - 10, 5 + badge_h), badge_color, -1)
    cv2.rectangle(overlay, (W - badge_w - 10, 5),
                  (W - 10, 5 + badge_h), (255, 255, 255), 2)

    # Badge text
    cv2.putText(overlay, f"{risk}",
                (W - badge_w, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Risk Score: {score:.0f}/100",
                (W - badge_w, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Dominant obstacle label
    dom = alert_info["dominant_obstacle"]
    cv2.putText(overlay, f"Dominant: {dom}",
                (W - badge_w, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, badge_color, 1, cv2.LINE_AA)

    return overlay


# ─────────────────────────────────────────────────────────────
# COVERAGE BAR CHART
# ─────────────────────────────────────────────────────────────
def draw_coverage_chart(coverage_dict, width=400, height=250):
    """Draw a horizontal bar chart of per-class coverage."""
    chart = np.ones((height, width, 3), dtype=np.uint8) * 30  # dark bg

    bar_h = 20
    margin = 5
    max_bar_w = width - 120

    y = 10
    cv2.putText(chart, "Object Coverage (%)", (10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 25

    for cls_id in range(NUM_CLASSES):
        name = CLASS_SHORTS[cls_id]
        pct = coverage_dict.get(CLASS_NAMES[cls_id], 0.0)
        bar_w = int(pct / 100.0 * max_bar_w)
        bar_w = max(bar_w, 1)

        color = CLASS_COLORS_BGR[cls_id]
        if cls_id == 0:  # background — grey bar
            color = (80, 80, 80)

        cv2.putText(chart, f"{name}", (5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.rectangle(chart, (40, y), (40 + bar_w, y + bar_h), color, -1)
        cv2.putText(chart, f"{pct:.1f}%", (45 + bar_w, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        y += bar_h + margin

    return chart


# ─────────────────────────────────────────────────────────────
# MAIN PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(image_path, model_path=MODEL_PATH, save_dir=OUT_DIR,
            show_alert=True, show_quality=True):
    """
    Run 8-class segmentation + adaptive preprocessing + collision alert
    on a single underwater image.
    """
    # ── Validate inputs ──
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train first with: python train_multiclass.py"
        )

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ── Load image ──
    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Could not read image: {image_path}")
    H, W = orig.shape[:2]
    print(f"  Image size    : {W}×{H} px")

    # ── Image quality assessment ──
    quality = assess_underwater_quality(orig)
    if show_quality:
        print(f"  Quality metrics:")
        print(f"    Color cast  : {quality['color_cast']:.2f}"
              f"  ({'strong' if quality['color_cast'] > 1.5 else 'moderate' if quality['color_cast'] > 1.2 else 'mild'})")
        print(f"    Blur score  : {quality['blur_score']:.1f}")
        print(f"    Contrast    : {quality['contrast']:.3f}")
        print(f"    Noise level : {quality['noise']:.1f}")

    # ── Adaptive DIP preprocessing ──
    print("  Applying adaptive DIP preprocessing ...")
    preprocessed, chosen_cfg, _ = adaptive_preprocess(orig)
    dip_steps = preprocess_pipeline_steps(orig, chosen_cfg)
    print(f"    Config: color={chosen_cfg['color']}, "
          f"denoise={chosen_cfg['denoise']}, "
          f"contrast={chosen_cfg['contrast']}")

    # ── Prepare tensor ──
    rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    tensor = norm(T.ToTensor()(resized.copy())).unsqueeze(0).to(DEVICE)

    # ── Load model ──
    print(f"  Loading model : {os.path.basename(model_path)}")
    model = ViTMultiClassSegmentation(
        VIT_NAME, pretrained=False, num_classes=NUM_CLASSES, img_size=IMG_SIZE
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  Device        : {DEVICE.upper()}")

    # ── Inference ──
    print("  Running inference ...")
    with torch.no_grad():
        with autocast(enabled=(DEVICE == "cuda")):
            logits = model(tensor)
        pred_small = logits.argmax(dim=1).cpu().numpy()[0]  # (H_small, W_small)

    # ── Upscale to original size ──
    pred_full = cv2.resize(pred_small.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)

    # ──────────────────────────────────────────────
    # OBJECT COVERAGE ANALYSIS
    # ──────────────────────────────────────────────
    coverage = compute_coverage(pred_full)
    coverage_named = {CLASS_NAMES[k]: v for k, v in coverage.items()}

    print(f"\n  {'─'*45}")
    print(f"  OBJECT COVERAGE ANALYSIS")
    print(f"  {'─'*45}")
    for cls_id in range(NUM_CLASSES):
        pct = coverage[cls_id]
        if pct > 0.1:  # Only show classes with >0.1% coverage
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            print(f"    {CLASS_SHORTS[cls_id]:>3s} {CLASS_NAMES[cls_id]:<25s}: "
                  f"{pct:5.1f}%  {bar}")

    # ──────────────────────────────────────────────
    # COLLISION ALERT SYSTEM
    # ──────────────────────────────────────────────
    alert_info = collision_alert(pred_full)

    if show_alert:
        risk_icons = {
            "SAFE":    "🟢",
            "CAUTION": "🟡",
            "WARNING": "🟠",
            "DANGER":  "🔴",
        }
        icon = risk_icons.get(alert_info["risk_level"], "⚪")

        print(f"\n  {'─'*45}")
        print(f"  {icon} COLLISION ALERT: {alert_info['risk_level']}")
        print(f"  {'─'*45}")
        print(f"    Risk Score      : {alert_info['overall_risk_score']:.0f} / 100")
        print(f"    Dominant Threat  : {alert_info['dominant_obstacle']}")
        print(f"    Recommended      : {alert_info['recommended_action']}")

        if alert_info["per_class_risk"]:
            print(f"\n    Per-obstacle risk:")
            for name, risk_val in sorted(alert_info["per_class_risk"].items(),
                                         key=lambda x: -x[1]):
                if risk_val > 0.5:
                    print(f"      {name:<25s}: {risk_val:5.1f}")

    # ──────────────────────────────────────────────
    # SAVE OUTPUTS
    # ──────────────────────────────────────────────

    # 1. DIP stage images
    stage_files = {}
    stage_names = [
        "00_original",
        "01_after_color_correction",
        "02_after_denoising",
        "03_after_contrast_enhancement",
    ]
    for file_tag, (stage_label, stage_img) in zip(stage_names, dip_steps):
        stage_path = os.path.join(save_dir, f"{base}_{file_tag}.png")
        cv2.imwrite(stage_path, stage_img)
        stage_files[stage_label] = os.path.basename(stage_path)

    # 2. Colourised segmentation mask
    color_mask = class_index_to_color(pred_full)
    mask_path = os.path.join(save_dir, f"{base}_segmentation.png")
    cv2.imwrite(mask_path, color_mask)

    # 3. Segmentation overlay on original
    overlay = cv2.addWeighted(orig, 0.55, color_mask, 0.45, 0)
    overlay_path = os.path.join(save_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # 4. Collision alert overlay
    alert_overlay = draw_collision_overlay(orig, pred_full, alert_info)
    alert_path = os.path.join(save_dir, f"{base}_collision_alert.png")
    cv2.imwrite(alert_path, alert_overlay)

    # 5. Coverage bar chart
    coverage_chart = draw_coverage_chart(coverage_named)
    chart_path = os.path.join(save_dir, f"{base}_coverage.png")
    cv2.imwrite(chart_path, coverage_chart)

    # 6. Comprehensive comparison panel
    panel_w = 400
    panel_h = 300

    def resize_panel(img):
        return cv2.resize(img, (panel_w, panel_h))

    def add_label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (panel_w, 28), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    row1 = np.hstack([
        add_label(resize_panel(orig), "Original"),
        add_label(resize_panel(preprocessed), f"Adaptive DIP ({chosen_cfg['color']})"),
    ])
    row2 = np.hstack([
        add_label(resize_panel(color_mask), "8-Class Segmentation"),
        add_label(resize_panel(overlay), "Overlay"),
    ])
    row3 = np.hstack([
        add_label(resize_panel(alert_overlay), f"Collision: {alert_info['risk_level']}"),
        add_label(resize_panel(
            cv2.resize(coverage_chart, (panel_w, panel_h))
        ), "Coverage Analysis"),
    ])

    # Legend bar
    legend = draw_legend(width=panel_w * 2, bar_h=30)

    comparison = np.vstack([row1, row2, row3, legend])
    comparison_path = os.path.join(save_dir, f"{base}_full_analysis.jpg")
    cv2.imwrite(comparison_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 7. JSON report
    report = {
        "image": os.path.basename(image_path),
        "image_size": {"width": W, "height": H},
        "quality_metrics": quality,
        "preprocessing_config": chosen_cfg,
        "preprocessing_stage_files": stage_files,
        "object_coverage": coverage_named,
        "collision_alert": {
            "risk_level": alert_info["risk_level"],
            "risk_score": alert_info["overall_risk_score"],
            "dominant_obstacle": alert_info["dominant_obstacle"],
            "recommended_action": alert_info["recommended_action"],
            "per_obstacle_risk": alert_info["per_class_risk"],
        },
    }
    report_path = os.path.join(save_dir, f"{base}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Summary ──
    print(f"\n  ✅ Done! Results saved to: {save_dir}")
    print(f"\n  Output files:")
    print(f"    → {base}_00_original.png         (stage 0: input image)")
    print(f"    → {base}_01_after_color_correction.png")
    print(f"    → {base}_02_after_denoising.png")
    print(f"    → {base}_03_after_contrast_enhancement.png")
    print(f"    → {base}_segmentation.png     (8-class colourised mask)")
    print(f"    → {base}_overlay.png          (segmentation on original)")
    print(f"    → {base}_collision_alert.png  (collision zone analysis)")
    print(f"    → {base}_coverage.png         (object coverage chart)")
    print(f"    → {base}_full_analysis.jpg    (comprehensive comparison)")
    print(f"    → {base}_report.json          (full JSON report)")

    return report


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="8-Class Underwater Segmentation + Collision Alert"
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to input underwater image (jpg/png)"
    )
    parser.add_argument(
        "--model", "-m", default=MODEL_PATH,
        help=f"Path to trained model weights (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--outdir", "-o", default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})"
    )
    parser.add_argument(
        "--no-alert", action="store_true",
        help="Disable collision alert output"
    )
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Disable quality metrics display"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🌊 UNDERWATER SEGMENTATION + COLLISION ALERT SYSTEM")
    print("=" * 60)
    print(f"  Input image  : {args.image}")
    print(f"  Model        : {args.model}")
    print(f"  Device       : {DEVICE.upper()}")
    print("-" * 60)

    predict(
        image_path=args.image,
        model_path=args.model,
        save_dir=args.outdir,
        show_alert=not args.no_alert,
        show_quality=not args.no_quality,
    )
