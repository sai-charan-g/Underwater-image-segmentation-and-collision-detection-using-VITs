# ============================================================
# train_multiclass.py — 8-Class SUIM Underwater Segmentation
# Features:
#   ✓ Multi-class segmentation (8 SUIM categories)
#   ✓ Adaptive DIP preprocessing (quality-aware)
#   ✓ Learnable Enhancement Stem + ViT encoder
#   ✓ Mixed-precision training (AMP)
#   ✓ Cosine LR scheduler + early stopping
#   ✓ Data augmentation (flip, rotate, colour jitter)
#   ✓ Per-class IoU (mIoU) evaluation
#
# Usage:
#   python train_multiclass.py
#   python train_multiclass.py --epochs 20 --batch 8
# ============================================================

import os
import sys
import math
import time
import random
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

from sklearn.model_selection import train_test_split

from suim_utils import (
    NUM_CLASSES, IMG_SIZE, CLASS_NAMES, CLASS_SHORTS,
    SUIM_CLASSES, rgb_mask_to_class_index, class_index_to_color,
    adaptive_preprocess, assess_underwater_quality,
    ViTMultiClassSegmentation,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(BASE_DIR, "data", "dataset")
IMAGE_DIR  = os.path.join(DATA_ROOT, "images")
MASK_DIR   = os.path.join(DATA_ROOT, "masks")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
VIT_NAME   = "vit_small_patch16_224"
SEED       = 42

# ─────────────────────────────────────────────────────────────
# UTILITY: find image-mask pairs
# ─────────────────────────────────────────────────────────────
def list_pairs(image_dir, mask_dir,
               img_exts=(".jpg", ".jpeg", ".png"),
               mask_exts=(".bmp", ".png")):
    images = []
    for e in img_exts:
        images += glob(os.path.join(image_dir, f"*{e}"))
    images = sorted(images)

    masks_map = {}
    for e in mask_exts:
        for p in glob(os.path.join(mask_dir, f"*{e}")):
            masks_map[os.path.splitext(os.path.basename(p))[0]] = p

    pairs = []
    for ip in images:
        key = os.path.splitext(os.path.basename(ip))[0]
        if key in masks_map:
            pairs.append((ip, masks_map[key]))
    return pairs


# ─────────────────────────────────────────────────────────────
# DATASET with augmentation + adaptive preprocessing
# ─────────────────────────────────────────────────────────────
class SUIMMultiClassDataset(Dataset):
    """
    Loads SUIM image-mask pairs.
    Masks are RGB-encoded → converted to class indices (0–7).
    Optionally applies data augmentation and adaptive DIP preprocessing.
    """
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False, use_adaptive_dip=True):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.use_adaptive_dip = use_adaptive_dip
        self.norm = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ── Load ──
        img_bgr = cv2.imread(img_path)
        mask_bgr = cv2.imread(mask_path)  # RGB mask (3-channel)

        if img_bgr is None or mask_bgr is None:
            # Fallback: return zeros
            dummy_img = torch.zeros(3, self.img_size, self.img_size)
            dummy_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
            return dummy_img, dummy_mask

        # ── Handle image-mask size mismatch (9 pairs have this) ──
        if img_bgr.shape[:2] != mask_bgr.shape[:2]:
            mask_bgr = cv2.resize(mask_bgr,
                                  (img_bgr.shape[1], img_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # ── Adaptive DIP preprocessing ──
        if self.use_adaptive_dip:
            img_bgr, _, _ = adaptive_preprocess(img_bgr)

        # ── Convert mask to RGB then to class indices ──
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        label_map = rgb_mask_to_class_index(mask_rgb)  # (H, W) with values 0-7

        # ── Resize ──
        img_resized = cv2.resize(img_bgr, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label_map, (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_NEAREST)

        # ── Augmentation ──
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_resized = np.flip(img_resized, axis=1).copy()
                label_resized = np.flip(label_resized, axis=1).copy()

            # Random vertical flip
            if random.random() > 0.5:
                img_resized = np.flip(img_resized, axis=0).copy()
                label_resized = np.flip(label_resized, axis=0).copy()

            # Random 90° rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img_resized = np.rot90(img_resized, k).copy()
                label_resized = np.rot90(label_resized, k).copy()

            # Random brightness/contrast jitter (image only)
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta  = random.randint(-15, 15)    # brightness
            img_resized = np.clip(
                img_resized.astype(np.float32) * alpha + beta, 0, 255
            ).astype(np.uint8)

        # ── To tensor ──
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(
            img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        img_tensor = self.norm(img_tensor)
        label_tensor = torch.from_numpy(label_resized.astype(np.int64))

        return img_tensor, label_tensor


# ─────────────────────────────────────────────────────────────
# LOSS: Cross-Entropy + Multi-class Dice
# ─────────────────────────────────────────────────────────────
class MultiClassSegLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for multi-class segmentation."""
    def __init__(self, num_classes=NUM_CLASSES, dice_weight=1.0, ce_weight=1.0,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        logits:  (B, C, H, W) raw scores
        targets: (B, H, W) class indices 0..C-1
        """
        # Cross-entropy
        ce = F.cross_entropy(logits, targets,
                             weight=self.class_weights,
                             ignore_index=255)

        # Multi-class Dice
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        one_hot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)  # sum over batch, H, W
        inter = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────────
# METRICS: per-class IoU and mIoU
# ─────────────────────────────────────────────────────────────
class SegmentationMetrics:
    """Accumulates per-class IoU, Dice, and pixel accuracy."""
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes),
                                  dtype=np.int64)

    def update(self, pred, target):
        """
        pred:   (H, W) numpy with class indices
        target: (H, W) numpy with class indices
        """
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def get_results(self):
        cm = self.confusion
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        pixel_acc = tp.sum() / (cm.sum() + 1e-8)

        # Only count classes that actually appear in the dataset
        present = (cm.sum(axis=1) > 0)

        return {
            "per_class_iou":  iou,
            "per_class_dice": dice,
            "mIoU":           float(iou[present].mean()) if present.any() else 0.0,
            "mDice":          float(dice[present].mean()) if present.any() else 0.0,
            "pixel_acc":      float(pixel_acc),
            "present_classes": present,
        }


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(args.outdir, exist_ok=True)

    # ── Dataset ──
    pairs = list_pairs(IMAGE_DIR, MASK_DIR)
    if len(pairs) == 0:
        raise RuntimeError(f"No pairs found in {DATA_ROOT}")

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.12, random_state=SEED
    )
    print(f"Dataset: {len(pairs)} total  |  {len(train_pairs)} train  |  {len(val_pairs)} val")

    train_ds = SUIMMultiClassDataset(train_pairs, augment=True, use_adaptive_dip=True)
    val_ds   = SUIMMultiClassDataset(val_pairs,   augment=False, use_adaptive_dip=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ──
    model = ViTMultiClassSegmentation(
        vit_name=VIT_NAME, pretrained=True,
        num_classes=NUM_CLASSES, img_size=IMG_SIZE,
    ).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {VIT_NAME}  |  {param_count:,} parameters  |  {NUM_CLASSES} classes")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Loss with class weights (fixes 44x class imbalance) ──
    # Inverse-frequency weights from dataset audit:
    #   BW=32.4%, HD=2.3%, PF=2.2%, WR=6.7%, RO=7.2%,
    #   RI=0.8%, FV=34.3%, SR=14.2%
    class_freq = torch.tensor(
        [0.324, 0.023, 0.022, 0.067, 0.072, 0.008, 0.343, 0.142],
        dtype=torch.float32,
    )
    class_weights = (1.0 / (class_freq + 1e-6))
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalise
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    criterion = MultiClassSegLoss(
        num_classes=NUM_CLASSES, class_weights=class_weights.to(DEVICE)
    ).to(DEVICE)

    # ── Mixed precision ──
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # ── Training ──
    best_miou = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'='*65}")
    print(f"  TRAINING: 8-Class Underwater Segmentation")
    print(f"  Device: {DEVICE.upper()}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        metrics = SegmentationMetrics(NUM_CLASSES)

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                with autocast(enabled=(DEVICE == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                gts   = labels.cpu().numpy()

                for i in range(preds.shape[0]):
                    metrics.update(preds[i], gts[i])

        val_loss /= len(val_ds)
        results = metrics.get_results()
        elapsed = time.time() - t0

        # ── Log ──
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.1f}s)  lr={lr_now:.2e}")
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"  mIoU: {results['mIoU']:.4f}  |  mDice: {results['mDice']:.4f}  "
              f"|  Pixel Acc: {results['pixel_acc']:.4f}")

        # Per-class IoU
        print("  Per-class IoU:")
        for cls_id in range(NUM_CLASSES):
            if results["present_classes"][cls_id]:
                print(f"    {CLASS_SHORTS[cls_id]:>3s} ({CLASS_NAMES[cls_id]:<25s}): "
                      f"{results['per_class_iou'][cls_id]:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mIoU": results["mIoU"],
            "mDice": results["mDice"],
            "pixel_acc": results["pixel_acc"],
            "lr": lr_now,
        })

        # ── Save best ──
        if results["mIoU"] > best_miou:
            best_miou = results["mIoU"]
            patience_counter = 0
            save_path = os.path.join(args.outdir, "best_model_multiclass.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping — no improvement for {args.patience} epochs.")
                break

    # ── Save history ──
    pd.DataFrame(history).to_csv(
        os.path.join(args.outdir, "training_history_multiclass.csv"), index=False
    )

    print(f"\n{'='*65}")
    print(f"  ✓ Training complete!  Best mIoU = {best_miou:.4f}")
    print(f"  ✓ Model saved to: {os.path.join(args.outdir, 'best_model_multiclass.pth')}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 8-class SUIM underwater segmentation model"
    )
    parser.add_argument("--epochs",   type=int,   default=15)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--workers",  type=int,   default=2)
    parser.add_argument("--patience", type=int,   default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--outdir",   default=OUT_DIR)
    args = parser.parse_args()
    train(args)
