# ============================================================
# generate_results.py — Generate presentation-ready visuals
# Works WITHOUT a trained model (uses ground truth masks)
# Run: python generate_results.py
# ============================================================
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, cv2, random, json
import numpy as np
from glob import glob

from suim_utils import (
    NUM_CLASSES, CLASS_NAMES, CLASS_SHORTS, CLASS_COLORS_BGR,
    COLLISION_CLASSES, rgb_mask_to_class_index, class_index_to_color,
    adaptive_preprocess, assess_underwater_quality,
    preprocess_pipeline, compute_coverage, collision_alert,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "presentation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(42)

# ── Find pairs ──
def find_pairs(img_dir, mask_dir):
    imgs, masks = {}, {}
    for e in (".jpg", ".jpeg", ".png"):
        for p in glob(os.path.join(img_dir, f"*{e}")):
            imgs[os.path.splitext(os.path.basename(p))[0]] = p
    for e in (".bmp", ".png"):
        for p in glob(os.path.join(mask_dir, f"*{e}")):
            masks[os.path.splitext(os.path.basename(p))[0]] = p
    paired = sorted(set(imgs.keys()) & set(masks.keys()))
    return [(imgs[k], masks[k]) for k in paired]

pairs = find_pairs(os.path.join(DATA_ROOT, "images"),
                   os.path.join(DATA_ROOT, "masks"))
print(f"Found {len(pairs)} pairs")

# ── Helper: add text label to image ──
def add_label(img, text, bg_color=(0,0,0), font_scale=0.55):
    img = img.copy()
    h = 28
    cv2.rectangle(img, (0, 0), (img.shape[1], h), bg_color, -1)
    cv2.putText(img, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
    return img

def add_bottom_label(img, text, bg_color=(40,40,40)):
    img = img.copy()
    h = img.shape[0]
    cv2.rectangle(img, (0, h-24), (img.shape[1], h), bg_color, -1)
    cv2.putText(img, text, (6, h-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1, cv2.LINE_AA)
    return img

def draw_legend(width=800, bar_h=35):
    legend = np.zeros((bar_h, width, 3), dtype=np.uint8)
    n = NUM_CLASSES
    seg_w = width // n
    for i in range(n):
        x1 = i * seg_w
        x2 = (i + 1) * seg_w if i < n - 1 else width
        legend[:, x1:x2] = CLASS_COLORS_BGR[i]
        txt_color = (255,255,255) if i != 7 else (0,0,0)
        cv2.putText(legend, f"{CLASS_SHORTS[i]}: {CLASS_NAMES[i]}",
                    (x1 + 3, bar_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, txt_color, 1, cv2.LINE_AA)
    return legend

SZ = 320  # panel size

# ──────────────────────────────────────────────
# RESULT 1: Before/After DIP Preprocessing
# Pick 4 diverse images
# ──────────────────────────────────────────────
print("\n[1/5] Generating DIP Before/After comparisons...")

# Find images with diverse quality
quality_data = []
for img_path, _ in random.sample(pairs, min(200, len(pairs))):
    img = cv2.imread(img_path)
    if img is None: continue
    q = assess_underwater_quality(img)
    quality_data.append((img_path, q))

# Pick: 1 strong cast, 1 mild, 1 high noise, 1 low contrast
quality_data.sort(key=lambda x: x[1]["color_cast"], reverse=True)
picks = [quality_data[0][0]]  # strongest cast
quality_data.sort(key=lambda x: x[1]["color_cast"])
picks.append(quality_data[0][0])  # mildest cast
quality_data.sort(key=lambda x: x[1]["noise"], reverse=True)
picks.append(quality_data[0][0])  # noisiest
quality_data.sort(key=lambda x: x[1]["contrast"])
picks.append(quality_data[0][0])  # lowest contrast

rows = []
for p in picks:
    img = cv2.imread(p)
    q = assess_underwater_quality(img)
    pre, cfg, _ = adaptive_preprocess(img)
    
    orig_panel = add_label(cv2.resize(img, (SZ, SZ)), "Original")
    orig_panel = add_bottom_label(orig_panel,
        f"Cast={q['color_cast']:.1f} Noise={q['noise']:.0f} Contrast={q['contrast']:.2f}")
    
    pre_panel = add_label(cv2.resize(pre, (SZ, SZ)),
        f"Adaptive: {cfg['color']}+{cfg['denoise']}+{cfg['contrast']}")
    
    rows.append(np.hstack([orig_panel, pre_panel]))

dip_result = np.vstack(rows)
path = os.path.join(RESULTS_DIR, "01_dip_before_after.jpg")
cv2.imwrite(path, dip_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# RESULT 2: 8-Class Segmentation Examples
# ──────────────────────────────────────────────
print("[2/5] Generating 8-class segmentation visualisations...")

# Pick images with diverse classes
seg_picks = random.sample(pairs, min(6, len(pairs)))
rows = []
for img_path, mask_path in seg_picks:
    img = cv2.imread(img_path)
    mask_bgr = cv2.imread(mask_path)
    if img is None or mask_bgr is None: continue
    
    if img.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    labels = rgb_mask_to_class_index(mask_rgb)
    color_mask = class_index_to_color(labels)
    enhanced, cfg, _ = adaptive_preprocess(img)
    
    overlay = cv2.addWeighted(enhanced, 0.5, color_mask, 0.5, 0)
    
    coverage = compute_coverage(labels)
    classes_found = [CLASS_SHORTS[c] for c in range(NUM_CLASSES) if coverage[c] > 1.0]
    
    orig_p = add_label(cv2.resize(img, (SZ, SZ)), "Original")
    enhanced_p = add_label(cv2.resize(enhanced, (SZ, SZ)),
        f"Enhanced ({cfg['color']}+{cfg['denoise']}+{cfg['contrast']})")
    mask_p = add_label(cv2.resize(color_mask, (SZ, SZ)), "8-Class Segmentation")
    over_p = add_label(cv2.resize(overlay, (SZ, SZ)),
        f"Enhanced + Segmentation: {', '.join(classes_found)}")
    
    rows.append(np.hstack([orig_p, enhanced_p, mask_p, over_p]))

seg_result = np.vstack(rows[:4])  # max 4 rows
legend = draw_legend(width=SZ*4, bar_h=40)
seg_result = np.vstack([seg_result, legend])
path = os.path.join(RESULTS_DIR, "02_multiclass_segmentation.jpg")
cv2.imwrite(path, seg_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# RESULT 3: Collision Alert Demo
# ──────────────────────────────────────────────
print("[3/5] Generating collision alert demonstrations...")

# Find images with different risk levels
alert_data = []
for img_path, mask_path in pairs:
    mask_bgr = cv2.imread(mask_path)
    img = cv2.imread(img_path)
    if mask_bgr is None or img is None: continue
    if img.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    labels = rgb_mask_to_class_index(mask_rgb)
    alert = collision_alert(labels)
    alert_data.append((img_path, mask_path, alert))

# Group by risk level and pick one of each
by_level = {}
for ip, mp, a in alert_data:
    level = a["risk_level"]
    if level not in by_level:
        by_level[level] = (ip, mp, a)

rows = []
for level in ["SAFE", "CAUTION", "WARNING", "DANGER"]:
    if level not in by_level:
        continue
    ip, mp, alert = by_level[level]
    img = cv2.imread(ip)
    mask_bgr = cv2.imread(mp)
    if img.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    labels = rgb_mask_to_class_index(mask_rgb)
    color_mask = class_index_to_color(labels)
    
    # Draw zone overlay
    H, W = img.shape[:2]
    zone_overlay = img.copy()
    zone_colors = [(0,180,0), (0,200,255), (0,0,255)]
    zone_names = ["FAR", "APPROACHING", "NEAR"]
    for i in range(3):
        y1, y2 = i * H // 3, (i + 1) * H // 3
        tmp = zone_overlay.copy()
        cv2.rectangle(tmp, (0, y1), (W, y2), zone_colors[i], -1)
        zone_overlay = cv2.addWeighted(zone_overlay, 0.82, tmp, 0.18, 0)
        cv2.putText(zone_overlay, zone_names[i], (5, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_colors[i], 1, cv2.LINE_AA)
    
    # Risk badge
    badge_colors = {"SAFE": (0,200,0), "CAUTION": (0,200,255),
                    "WARNING": (0,140,255), "DANGER": (0,0,255)}
    bc = badge_colors[level]
    score = alert["overall_risk_score"]
    cv2.rectangle(zone_overlay, (W-200, 5), (W-5, 55), bc, -1)
    cv2.rectangle(zone_overlay, (W-200, 5), (W-5, 55), (255,255,255), 2)
    cv2.putText(zone_overlay, level, (W-195, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(zone_overlay, f"Score: {score:.0f}/100", (W-195, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    
    seg_p = add_label(cv2.resize(color_mask, (SZ, SZ)), "Segmentation")
    zone_p = add_label(cv2.resize(zone_overlay, (SZ, SZ)),
        f"{level} - {alert['dominant_obstacle']}")
    
    rows.append(np.hstack([seg_p, zone_p]))

if rows:
    alert_result = np.vstack(rows)
    path = os.path.join(RESULTS_DIR, "03_collision_alerts.jpg")
    cv2.imwrite(path, alert_result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# RESULT 4: Coverage Analysis Chart
# ──────────────────────────────────────────────
print("[4/5] Generating coverage analysis...")

# Aggregate coverage across whole dataset (sample 100)
sample_pairs = random.sample(pairs, min(100, len(pairs)))
all_coverage = np.zeros(NUM_CLASSES)
count = 0
for ip, mp in sample_pairs:
    mask_bgr = cv2.imread(mp)
    img = cv2.imread(ip)
    if mask_bgr is None or img is None: continue
    if img.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    labels = rgb_mask_to_class_index(mask_rgb)
    cov = compute_coverage(labels)
    for c in range(NUM_CLASSES):
        all_coverage[c] += cov[c]
    count += 1

avg_coverage = all_coverage / count

# Draw chart
chart_w, chart_h = 700, 400
chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 30

cv2.putText(chart, "Average Object Coverage Across Dataset (%)", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

bar_h = 30
y = 60
max_bar = chart_w - 200
for c in range(NUM_CLASSES):
    pct = avg_coverage[c]
    bar_w = max(int(pct / 100 * max_bar), 2)
    color = CLASS_COLORS_BGR[c] if c > 0 else (80, 80, 80)
    
    cv2.putText(chart, f"{CLASS_SHORTS[c]}", (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.rectangle(chart, (60, y), (60 + bar_w, y + bar_h), color, -1)
    cv2.putText(chart, f"{pct:.1f}%", (70 + bar_w, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    cv2.putText(chart, CLASS_NAMES[c], (60 + bar_w + 60, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
    y += bar_h + 8

path = os.path.join(RESULTS_DIR, "04_coverage_chart.jpg")
cv2.imwrite(path, chart, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# RESULT 5: Full Pipeline Demo (single image)
# ──────────────────────────────────────────────
print("[5/5] Generating full pipeline demo...")

# Pick an image with multiple classes
best_ip, best_mp = None, None
best_classes = 0
for ip, mp in random.sample(pairs, min(200, len(pairs))):
    mask_bgr = cv2.imread(mp)
    img = cv2.imread(ip)
    if mask_bgr is None or img is None: continue
    if img.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    labels = rgb_mask_to_class_index(mask_rgb)
    n_classes = len(np.unique(labels))
    if n_classes > best_classes:
        best_classes = n_classes
        best_ip, best_mp = ip, mp

img = cv2.imread(best_ip)
mask_bgr = cv2.imread(best_mp)
if img.shape[:2] != mask_bgr.shape[:2]:
    mask_bgr = cv2.resize(mask_bgr, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

q = assess_underwater_quality(img)
pre, cfg, _ = adaptive_preprocess(img)
mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
labels = rgb_mask_to_class_index(mask_rgb)
color_mask = class_index_to_color(labels)
overlay = cv2.addWeighted(pre, 0.5, color_mask, 0.5, 0)
alert = collision_alert(labels)

P = 350
row1 = np.hstack([
    add_label(cv2.resize(img, (P, P)), "1. Original Input"),
    add_label(cv2.resize(pre, (P, P)),
        f"2. Adaptive DIP ({cfg['color']})"),
])
row2 = np.hstack([
    add_label(cv2.resize(color_mask, (P, P)), "3. 8-Class Segmentation"),
    add_label(cv2.resize(overlay, (P, P)),
        f"4. Enhanced Overlay + Collision: {alert['risk_level']} ({alert['overall_risk_score']:.0f}/100)"),
])
legend = draw_legend(width=P*2, bar_h=40)

# Quality + alert info bar
info_bar = np.ones((80, P*2, 3), dtype=np.uint8) * 30
info_lines = [
    f"Quality: cast={q['color_cast']:.1f}  noise={q['noise']:.0f}  contrast={q['contrast']:.2f}",
    f"Config: {cfg['color']} + {cfg['denoise']} + {cfg['contrast']}",
    f"Alert: {alert['risk_level']} | Threat: {alert['dominant_obstacle']} | Action: {alert['recommended_action'][:60]}",
]
for i, line in enumerate(info_lines):
    cv2.putText(info_bar, line, (10, 20 + i*25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

full_demo = np.vstack([row1, row2, legend, info_bar])
path = os.path.join(RESULTS_DIR, "05_full_pipeline_demo.jpg")
cv2.imwrite(path, full_demo, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"  Saved: {path}")

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ALL RESULTS SAVED TO: {RESULTS_DIR}")
print(f"{'='*55}")
print(f"\n  Files generated:")
print(f"    01_dip_before_after.jpg      - DIP preprocessing comparison")
print(f"    02_multiclass_segmentation.jpg - 8-class segmentation examples")
print(f"    03_collision_alerts.jpg       - SAFE/CAUTION/WARNING/DANGER examples")
print(f"    04_coverage_chart.jpg         - Dataset coverage bar chart")
print(f"    05_full_pipeline_demo.jpg     - Complete pipeline in one image")
print(f"\n  Use these directly in your presentation slides!")
print(f"{'='*55}")
