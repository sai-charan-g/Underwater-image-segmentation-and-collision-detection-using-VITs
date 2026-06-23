import os
import cv2
import numpy as np
import streamlit as st
from glob import glob
from PIL import Image

# Must be first Streamlit call
st.set_page_config(
    page_title="Underwater Segmentation + Collision Alert",
    page_icon="🌊",
    layout="wide",
)

from suim_utils import (
    NUM_CLASSES, IMG_SIZE, CLASS_NAMES, CLASS_SHORTS, SUIM_CLASSES,
    COLLISION_CLASSES, rgb_mask_to_class_index, class_index_to_color,
    adaptive_preprocess, assess_underwater_quality, preprocess_pipeline,
    preprocess_pipeline_steps,
    compute_coverage, compute_zone_coverage, collision_alert,
    ViTMultiClassSegmentation,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "dataset")
TEST_ROOT = os.path.join(BASE_DIR, "TEST")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model_multiclass.pth")

# ── Check if trained model exists ──
HAS_MODEL = os.path.exists(MODEL_PATH)

def download_model_weights(url):
    """Download model weights from a direct link."""
    import urllib.request
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("Downloading model weights from URL... This might take a moment."):
            urllib.request.urlretrieve(url, MODEL_PATH)
        st.success("Download complete!")
        return True
    except Exception as e:
        st.error(f"Failed to download weights: {e}")
        return False

# ── Colour palette for display (RGB for streamlit) ──
CLASS_COLORS_RGB = [SUIM_CLASSES[i]["rgb"] for i in range(NUM_CLASSES)]


@st.cache_resource
def load_model():
    """Load the trained multi-class segmentation model."""
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMultiClassSegmentation(
        "vit_small_patch16_224", pretrained=False,
        num_classes=NUM_CLASSES, img_size=IMG_SIZE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, DEVICE


def run_model_inference(img_bgr, model, device, input_bgr=None, cfg=None, q=None):
    """Run the ViT model on a preprocessed image."""
    import torch
    import torchvision.transforms as T
    from torch.cuda.amp import autocast

    if input_bgr is None:
        pre, cfg, q = adaptive_preprocess(img_bgr)
    else:
        pre = input_bgr
    rgb = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    tensor = norm(T.ToTensor()(resized.copy())).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(enabled=(device == "cuda")):
            logits = model(tensor)
        pred = logits.argmax(dim=1).cpu().numpy()[0]

    H, W = img_bgr.shape[:2]
    pred_full = cv2.resize(pred.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST)
    return pred_full, pre, cfg, q


def get_gt_labels(img_bgr, mask_path):
    """Get ground truth labels from a mask file."""
    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None:
        return None
    if img_bgr.shape[:2] != mask_bgr.shape[:2]:
        mask_bgr = cv2.resize(mask_bgr, (img_bgr.shape[1], img_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    return rgb_mask_to_class_index(mask_rgb)


def find_gt_mask(img_name):
    """Try to find the ground truth mask for a given image."""
    base = os.path.splitext(img_name)[0]
    for root in [os.path.join(DATA_ROOT, "masks"), os.path.join(TEST_ROOT, "masks")]:
        for ext in [".bmp", ".png"]:
            path = os.path.join(root, base + ext)
            if os.path.exists(path):
                return path
    return None


def labels_to_rgb(labels):
    """Convert label map to RGB image for Streamlit display."""
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        mask = (labels == cls_id)
        out[mask] = CLASS_COLORS_RGB[cls_id]
    return out


# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #0083b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00d2ff;
    }
    .risk-safe { color: #00e676; font-weight: bold; font-size: 1.5rem; }
    .risk-caution { color: #ffea00; font-weight: bold; font-size: 1.5rem; }
    .risk-warning { color: #ff9100; font-weight: bold; font-size: 1.5rem; }
    .risk-danger { color: #ff1744; font-weight: bold; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌊 Underwater Scene Analysis + Collision Alert</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hybrid DIP + Vision Transformer  |  8-Class SUIM Segmentation  |  Drone Safety System</p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────
# SIDEBAR — Input Selection
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Input")

    input_mode = st.radio("Choose input method:",
                          ["Select from dataset", "Upload image"])

    img_bgr = None
    img_name = None
    gt_mask_path = None
    segmentation_mode = "Model prediction" if HAS_MODEL else "Ground truth"

    if input_mode == "Select from dataset":
        # List available images from dataset and test
        dataset_images = sorted(glob(os.path.join(DATA_ROOT, "images", "*.jpg")))
        test_images = sorted(glob(os.path.join(TEST_ROOT, "images", "*.jpg")))

        source = st.selectbox("Source:", ["Training Set", "Test Set"])
        img_list = dataset_images if source == "Training Set" else test_images
        img_names = [os.path.basename(p) for p in img_list]

        if img_names:
            selected = st.selectbox("Select image:", img_names)
            idx = img_names.index(selected)
            img_bgr = cv2.imread(img_list[idx])
            img_name = selected
            gt_mask_path = find_gt_mask(selected)
        else:
            st.warning("No images found!")

    else:
        uploaded = st.file_uploader("Upload underwater image",
                                     type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_name = uploaded.name

    st.divider()
    st.header("⚙️ Settings")
    use_adaptive = st.checkbox("Adaptive DIP preprocessing", value=True)
    show_zones = st.checkbox("Show collision zones", value=True)

    if img_bgr is not None:
        source_options = []
        if HAS_MODEL:
            source_options.append("Model prediction")
        if gt_mask_path:
            source_options.append("Ground truth")
        if HAS_MODEL and gt_mask_path:
            source_options.append("Model + ground truth compare")

        if len(source_options) > 1:
            segmentation_mode = st.selectbox("Segmentation source:", source_options)
        elif source_options:
            segmentation_mode = source_options[0]

    if HAS_MODEL:
        st.success("✅ Trained model found")
    else:
        st.warning("⚠️ Model weights not found.")
        
        # 1. File Uploader for Model Weights
        uploaded_weights = st.file_uploader(
            "Upload model weights (.pth)", 
            type=["pth"], 
            key="model_uploader",
            help="Upload the trained 'best_model_multiclass.pth' file (up to 200MB)."
        )
        if uploaded_weights is not None:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_weights.getbuffer())
            st.success("Model weights uploaded successfully! Reloading...")
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        st.markdown("**OR**")
        
        # 2. Text Input for Direct URL Download
        weights_url = st.text_input(
            "Paste direct download URL:", 
            placeholder="https://example.com/best_model_multiclass.pth",
            key="model_url_input"
        )
        if st.button("Download Weights", key="download_weights_btn"):
            if weights_url:
                if download_model_weights(weights_url):
                    st.success("Weights downloaded successfully! Reloading...")
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
            else:
                st.warning("Please enter a URL first.")

# ─────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────
if img_bgr is not None:
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 1: Quality Assessment ──
    st.subheader("📊 Step 1 — Image Quality Assessment")
    quality = assess_underwater_quality(img_bgr)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cast_label = "Strong 🔴" if quality["color_cast"] > 1.5 else "Moderate 🟡" if quality["color_cast"] > 1.2 else "Mild 🟢"
        st.metric("Color Cast", f"{quality['color_cast']:.1f}", cast_label)
    with c2:
        st.metric("Blur Score", f"{quality['blur_score']:.0f}")
    with c3:
        st.metric("Contrast", f"{quality['contrast']:.2f}")
    with c4:
        noise_label = "High" if quality["noise"] > 20 else "Medium" if quality["noise"] > 10 else "Low"
        st.metric("Noise Level", f"{quality['noise']:.1f}", noise_label)

    # ── Step 2: Adaptive DIP Preprocessing ──
    st.subheader("🔧 Step 2 — Adaptive DIP Preprocessing")

    if use_adaptive:
        preprocessed, cfg, _ = adaptive_preprocess(img_bgr)
    else:
        cfg = {"color": "grayworld", "denoise": "nl_means", "contrast": "clahe"}
        preprocessed = preprocess_pipeline(img_bgr, cfg)
    dip_steps = preprocess_pipeline_steps(img_bgr, cfg)

    step_cols = st.columns(len(dip_steps))
    for col, (caption, step_img) in zip(step_cols, dip_steps):
        with col:
            step_rgb = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
            st.image(step_rgb, caption=caption, use_container_width=True)

    st.info(f"**Selected config:** Color = `{cfg['color']}` → Denoise = `{cfg['denoise']}` → Contrast = `{cfg['contrast']}`")

    # ── Step 3: Segmentation ──
    st.subheader("🧠 Step 3 — 8-Class Semantic Segmentation")

    labels = None
    gt_labels = None

    if segmentation_mode in ("Model prediction", "Model + ground truth compare") and HAS_MODEL:
        model, device = load_model()
        labels, _, _, _ = run_model_inference(
            img_bgr, model, device, input_bgr=preprocessed, cfg=cfg, q=quality
        )
        seg_source = "Model Prediction"

    if gt_mask_path and segmentation_mode in ("Ground truth", "Model + ground truth compare"):
        gt_labels = get_gt_labels(img_bgr, gt_mask_path)

    if segmentation_mode == "Ground truth" and gt_labels is not None:
        labels = gt_labels
        seg_source = "Ground Truth"

    if labels is None:
        st.warning("No trained model and no ground truth mask available for this image.")

    if labels is not None:
        color_mask_rgb = labels_to_rgb(labels)
        overlay_rgb = cv2.addWeighted(img_rgb, 0.5, color_mask_rgb, 0.5, 0)

        if segmentation_mode == "Model + ground truth compare" and gt_labels is not None:
            gt_color_rgb = labels_to_rgb(gt_labels)
            gt_overlay_rgb = cv2.addWeighted(img_rgb, 0.5, gt_color_rgb, 0.5, 0)
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with col2:
            st.image(color_mask_rgb, caption=f"Segmentation ({seg_source})", use_container_width=True)
        with col3:
            st.image(overlay_rgb, caption="Overlay", use_container_width=True)
        if segmentation_mode == "Model + ground truth compare" and gt_labels is not None:
            with col4:
                st.image(gt_overlay_rgb, caption="Ground Truth Overlay", use_container_width=True)

        # Legend
        legend_html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin:10px 0;">'
        for i in range(NUM_CLASSES):
            r, g, b = CLASS_COLORS_RGB[i]
            text_color = "white" if i != 7 else "black"
            legend_html += (
                f'<span style="background:rgb({r},{g},{b}); color:{text_color}; '
                f'padding:4px 10px; border-radius:4px; font-size:0.8rem;">'
                f'{CLASS_SHORTS[i]}: {CLASS_NAMES[i]}</span>'
            )
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

        # ── Step 4: Coverage Analysis ──
        st.subheader("📈 Step 4 — Object Coverage Analysis")

        coverage = compute_coverage(labels)

        cols = st.columns(4)
        for i, cls_id in enumerate(range(NUM_CLASSES)):
            pct = coverage[cls_id]
            if pct > 0.1:
                with cols[i % 4]:
                    r, g, b = CLASS_COLORS_RGB[cls_id]
                    st.markdown(
                        f'<div style="background:rgba({r},{g},{b},0.2); border-left:4px solid rgb({r},{g},{b}); '
                        f'padding:8px; border-radius:6px; margin:4px 0;">'
                        f'<b>{CLASS_SHORTS[cls_id]}</b> {CLASS_NAMES[cls_id]}<br>'
                        f'<span style="font-size:1.3rem; font-weight:bold;">{pct:.1f}%</span></div>',
                        unsafe_allow_html=True
                    )

        # ── Step 5: Collision Alert ──
        st.subheader("⚠️ Step 5 — Collision Risk Assessment")

        alert = collision_alert(labels)
        risk = alert["risk_level"]
        score = alert["overall_risk_score"]

        risk_styles = {
            "SAFE":    ("risk-safe",    "🟢", "#00e676"),
            "CAUTION": ("risk-caution", "🟡", "#ffea00"),
            "WARNING": ("risk-warning", "🟠", "#ff9100"),
            "DANGER":  ("risk-danger",  "🔴", "#ff1744"),
        }
        css_class, icon, color = risk_styles.get(risk, ("", "⚪", "#fff"))

        # Risk display
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(
                f'<div style="text-align:center; padding:20px; background:#1a1a2e; border-radius:12px; '
                f'border:3px solid {color};">'
                f'<div style="font-size:3rem;">{icon}</div>'
                f'<div class="{css_class}">{risk}</div>'
                f'<div style="font-size:2rem; font-weight:bold; color:{color};">{score:.0f}/100</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with rc2:
            st.markdown(f"**Dominant Threat:** {alert['dominant_obstacle']}")
            st.markdown(f"**Recommended Action:** {alert['recommended_action']}")

            if alert["per_class_risk"]:
                st.markdown("**Per-obstacle risk breakdown:**")
                for name, val in sorted(alert["per_class_risk"].items(), key=lambda x: -x[1]):
                    if val > 0.5:
                        bar_pct = min(val, 100)
                        st.progress(bar_pct / 100, text=f"{name}: {val:.1f}")

        # Zone visualization
        if show_zones:
            st.markdown("**Zone Coverage Analysis:**")
            zone_cov = compute_zone_coverage(labels)

            zcols = st.columns(3)
            zone_labels = [("far", "🟢 Far Zone", "Top 1/3"),
                           ("mid", "🟡 Approaching", "Middle 1/3"),
                           ("near", "🔴 Near Zone", "Bottom 1/3")]
            for i, (zkey, zlabel, zdesc) in enumerate(zone_labels):
                with zcols[i]:
                    st.markdown(f"**{zlabel}** ({zdesc})")
                    for cls_id in COLLISION_CLASSES:
                        pct = zone_cov[zkey].get(cls_id, 0)
                        if pct > 0.5:
                            st.markdown(f"- {CLASS_NAMES[cls_id]}: **{pct:.1f}%**")

        # JSON report
        with st.expander("📋 Full JSON Report"):
            report = {
                "image": img_name,
                "quality": quality,
                "preprocessing": cfg,
                "coverage": {CLASS_NAMES[k]: round(v, 1) for k, v in coverage.items()},
                "collision_alert": {
                    "risk_level": risk,
                    "risk_score": score,
                    "dominant_obstacle": alert["dominant_obstacle"],
                    "action": alert["recommended_action"],
                    "per_obstacle_risk": alert["per_class_risk"],
                },
            }
            st.json(report)

else:
    # Landing page
    st.markdown("""
    ### How to use:
    1. **Select an image** from the dataset or upload your own using the sidebar
    2. The system will automatically:
       - Assess image quality (colour cast, noise, contrast)
       - Apply adaptive DIP preprocessing
       - Segment into 8 underwater object classes
       - Analyse object coverage
       - Compute collision risk for drone navigation

    ### Pipeline Architecture:
    ```
    Input Image → Quality Assessment → Adaptive DIP → ViT Segmentation → Coverage + Collision Alert
    ```
    """)
