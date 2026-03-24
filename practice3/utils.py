import os
import csv
import time
import random

import numpy as np
import torch


def set_seed(seed: int = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    return device


def setup_folders(models_dir: str, results_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def get_latest_checkpoint(models_dir: str):
    if not os.path.exists(models_dir):
        return None
    ckpts = [f for f in os.listdir(models_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not ckpts:
        return None
    epochs = [int(f.split("_")[2].split(".")[0]) for f in ckpts]
    e = max(epochs)
    return os.path.join(models_dir, f"checkpoint_epoch_{e}.pth")


def find_best_weights(models_dir: str):
    if not os.path.exists(models_dir):
        return None
    preferred = ["best.pth", "best.pt"]
    for n in preferred:
        p = os.path.join(models_dir, n)
        if os.path.exists(p):
            return p
    candidates = []
    for f in os.listdir(models_dir):
        lower = f.lower()
        if lower.startswith("best") and (lower.endswith(".pth") or lower.endswith(".pt")):
            candidates.append(os.path.join(models_dir, f))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    return get_latest_checkpoint(models_dir)


def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, models_dir: str):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": float(val_loss),
    }
    path = os.path.join(models_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("val_loss", float("inf")))


def dice_iou_from_probs(prob01: np.ndarray, target01: np.ndarray, thr: float = 0.5, eps: float = 1e-7):
    pred = (prob01 >= thr).astype(np.uint8)
    tgt = (target01 >= 0.5).astype(np.uint8)
    inter = (pred & tgt).sum()
    union = (pred | tgt).sum()
    dice = (2 * inter + eps) / (pred.sum() + tgt.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def pixel_metrics_from_probs(prob01: np.ndarray, target01: np.ndarray, thr: float = 0.5):
    pred = (prob01 >= thr).astype(np.uint8)
    tgt = (target01 >= 0.5).astype(np.uint8)
    tp = int((pred & tgt).sum())
    tn = int(((1 - pred) & (1 - tgt)).sum())
    fp = int((pred & (1 - tgt)).sum())
    fn = int(((1 - pred) & tgt).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, pixel_acc=float(acc), pixel_precision=float(precision), pixel_recall=float(recall))


def apply_roi(prob01: np.ndarray, target01: np.ndarray, roi01: np.ndarray):
    m = (roi01 >= 0.5).astype(np.float32)
    return prob01 * m, target01 * m


def overlay(image01: np.ndarray, mask01: np.ndarray, color=(0, 1, 0), alpha=0.45):
    img = np.clip(image01, 0, 1)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img[..., :3]
    out = img_rgb.copy()
    m = mask01 >= 0.5
    out[m] = out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out


def match_overlay(image01: np.ndarray, gt01: np.ndarray, pred01: np.ndarray, alpha=0.55):
    img = np.clip(image01, 0, 1)
    if img.ndim == 2:
        base = np.stack([img, img, img], axis=-1)
    else:
        base = img[..., :3].copy()
    gt_b = (gt01 >= 0.5)
    pr_b = (pred01 >= 0.5)
    tp = gt_b & pr_b
    fp = (~gt_b) & pr_b
    fn = gt_b & (~pr_b)
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.4, 1.0], dtype=np.float32)
    out = base.copy()
    for mask, col in ((tp, green), (fp, red), (fn, blue)):
        if mask.any():
            out[mask] = out[mask] * (1 - alpha) + col * alpha
    return out


def save_metrics_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_viz_folder(out_dir: str, items: list[dict]):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for idx, it in enumerate(items):
        img = it["img"]
        gt = it["gt"]
        pred = it["pred"]
        out_png = os.path.join(out_dir, f"sample_{idx + 1:03d}.png")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(overlay(img, gt))
        axes[0].set_title("GT")
        axes[0].axis("off")
        axes[1].imshow(overlay(img, pred))
        axes[1].set_title("Pred")
        axes[1].axis("off")
        axes[2].imshow(match_overlay(img, gt, pred))
        axes[2].set_title("TP / FP / FN")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()

