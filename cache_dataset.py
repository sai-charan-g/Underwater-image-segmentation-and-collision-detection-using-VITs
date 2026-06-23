import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def red_channel_compensation(img_bgr):
    img = img_bgr.astype(np.float32)
    b_mean, r_mean = img[:, :, 0].mean(), img[:, :, 2].mean()
    img[:, :, 2] = np.clip(img[:, :, 2] + (b_mean - r_mean) * 0.8, 0, 255)
    return img.astype(np.uint8)

def gray_world(img):
    img = img.astype(np.float32)
    avg = img.mean()
    for c in range(3):
        ch_avg = img[:, :, c].mean()
        img[:, :, c] = np.clip(img[:, :, c] * (avg / (ch_avg + 1e-8)), 0, 255)
    return img.astype(np.uint8)

def nl_means_denoise(img, h=7):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

def adaptive_gamma(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_b = gray.mean() / 255.0
    gamma = np.clip(np.log(0.5) / np.log(mean_b + 1e-8), 0.4, 2.5)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img_bgr, table)

def clahe_enhance(img, clipLimit=3.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_pipeline(img_bgr):
    out = red_channel_compensation(img_bgr)
    out = gray_world(out)
    out = nl_means_denoise(out, h=7)
    out = adaptive_gamma(out)
    out = clahe_enhance(out, clipLimit=3.0)
    return out

def process_dir(img_dir, out_img_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    images = glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))
    for img_path in tqdm(images, desc=f"Processing {os.path.basename(img_dir)}"):
        name = os.path.basename(img_path)
        out_path = os.path.join(out_img_dir, name)
        if os.path.exists(out_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        pre = preprocess_pipeline(img)
        # resize to 224x224 to save space and time!
        pre = cv2.resize(pre, (224, 224))
        cv2.imwrite(out_path, pre)

def process_masks(mask_dir, out_mask_dir):
    os.makedirs(out_mask_dir, exist_ok=True)
    masks = glob(os.path.join(mask_dir, "*.png")) + glob(os.path.join(mask_dir, "*.bmp"))
    for mask_path in tqdm(masks, desc=f"Processing masks"):
        name = os.path.basename(mask_path)
        # Save masks as png consistently
        name = os.path.splitext(name)[0] + ".png"
        out_path = os.path.join(out_mask_dir, name)
        if os.path.exists(out_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # resize to 224x224 using nearest neighbor
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_path, mask)

if __name__ == "__main__":
    DATA_ROOT = r"C:\Users\Darshan\Desktop\mini project 2\data\dataset"
    OUT_ROOT = r"C:\Users\Darshan\Desktop\mini project 2\data\preprocessed"
    
    print("Caching training set...")
    process_dir(os.path.join(DATA_ROOT, "images"), os.path.join(OUT_ROOT, "images"))
    process_masks(os.path.join(DATA_ROOT, "masks"), os.path.join(OUT_ROOT, "masks"))
    
    TEST_ROOT = r"C:\Users\Darshan\Desktop\mini project 2\TEST"
    TEST_OUT = r"C:\Users\Darshan\Desktop\mini project 2\TEST_preprocessed"
    
    print("Caching test set...")
    process_dir(os.path.join(TEST_ROOT, "images"), os.path.join(TEST_OUT, "images"))
    process_masks(os.path.join(TEST_ROOT, "masks"), os.path.join(TEST_OUT, "masks"))
    
    print("Done!")
