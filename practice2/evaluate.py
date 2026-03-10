import os
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub

import utils
from model import UNet


_BASE_DIR = Path(__file__).resolve().parent


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _dice_iou_from_probs(probs: np.ndarray, target: np.ndarray, thr: float = 0.5, eps: float = 1e-7):
    pred = (probs >= thr).astype(np.uint8)
    tgt = (target >= 0.5).astype(np.uint8)
    inter = (pred & tgt).sum()
    union = (pred | tgt).sum()
    pred_sum = pred.sum()
    tgt_sum = tgt.sum()
    dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou), int(pred_sum), int(tgt_sum)


def _pixel_metrics_from_probs(probs: np.ndarray, target: np.ndarray, thr: float = 0.5):
    pred = (probs >= thr).astype(np.uint8)
    tgt = (target >= 0.5).astype(np.uint8)
    tp = int((pred & tgt).sum())
    tn = int(((1 - pred) & (1 - tgt)).sum())
    fp = int((pred & (1 - tgt)).sum())
    fn = int(((1 - pred) & tgt).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, pixel_acc=float(acc), pixel_precision=float(precision), pixel_recall=float(recall))


def _overlay(image01: np.ndarray, mask01: np.ndarray, color=(0, 1, 0), alpha=0.45):
    img = np.clip(image01, 0, 1)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img[..., :3]
    out = img_rgb.copy()
    m = mask01 >= 0.5
    out[m] = out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return out


def _export_pr_curves(y_true: np.ndarray, y_score: np.ndarray, out_png: Path):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1], 1)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pixel-level Precision-Recall")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def _export_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_png: Path):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    tpr = tp / np.maximum(tp[-1], 1)
    fpr = fp / np.maximum(fp[-1], 1)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pixel-level ROC")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def _load_split_df(dataset_root: str, split: str, val_split: float, seed: int):
    train_dir = os.path.join(dataset_root, "training_set")
    csv_path = os.path.join(dataset_root, "training_set_pixel_size_and_HC.csv")
    df = pd.read_csv(csv_path)
    if split == "train":
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(df, test_size=val_split, random_state=seed)
        return train_df.reset_index(drop=True), train_dir
    if split == "val":
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(df, test_size=val_split, random_state=seed)
        return val_df.reset_index(drop=True), train_dir
    if split == "test":
        test_dir = os.path.join(dataset_root, "test_set")
        test_csv = os.path.join(dataset_root, "test_set_pixel_size.csv")
        test_df = pd.read_csv(test_csv)
        test_df["head circumference (mm)"] = np.nan
        return test_df.reset_index(drop=True), test_dir
    raise ValueError(f"Unsupported split: {split}")


class _EvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, target_size=(256, 256)):
        self.df = df
        self.img_dir = img_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        pixel_size = float(row.get("pixel size(mm)", np.nan))
        hc_truth = float(row.get("head circumference (mm)", np.nan))

        img_path = os.path.join(self.img_dir, filename)
        base, ext = os.path.splitext(filename)
        mask_filename = f"{base}_Annotation{ext}"
        mask_path = os.path.join(self.img_dir, mask_filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image).unsqueeze(0)

        if os.path.exists(mask_path):
            mask_t = utils.process_mask(mask_path, target_size=self.target_size)
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
        else:
            mask_t = torch.zeros((1, *self.target_size), dtype=torch.float32)

        return image_t, mask_t, pixel_size, hc_truth, filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default=str(_BASE_DIR / "models"))
    parser.add_argument("--results_dir", default=str(_BASE_DIR / "results"))
    parser.add_argument("--weights", default=None, help="Optional path to weights; otherwise auto-detect best.* in models_dir")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--num_viz", type=int, default=24)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    dataset_root = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")
    df, img_dir = _load_split_df(dataset_root, args.split, args.val_split, args.seed)

    ds = _EvalDataset(df, img_dir, target_size=tuple(args.target_size))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(n_channels=1, n_classes=1).to(device)
    weights_path = args.weights or utils.find_best_weights(args.models_dir)
    if not weights_path:
        raise FileNotFoundError(f"No weights found in {args.models_dir}")
    meta = utils.load_model_weights(weights_path, model, device=device)
    model.eval()

    run_dir = _ensure_dir(Path(args.results_dir) / f"eval_{args.split}_{_timestamp()}")
    viz_dir = _ensure_dir(run_dir / "viz")
    masks_dir = _ensure_dir(run_dir / "pred_masks")

    rows = []
    agg = dict(
        weights=str(weights_path),
        weights_meta=meta,
        split=args.split,
        val_split=args.val_split,
        seed=args.seed,
        thr=args.thr,
        target_size=list(args.target_size),
        n_samples=len(ds),
        device=device,
    )

    dice_list = []
    iou_list = []
    mae_list = []
    pixel_tp = pixel_tn = pixel_fp = pixel_fn = 0

    y_true_all = []
    y_score_all = []

    with torch.no_grad():
        for images, masks, pixel_sizes, hc_truths, filenames in dl:
            images = images.to(device)
            logits = model(images).detach().cpu().numpy()
            probs = _sigmoid_np(logits)

            masks_np = masks.numpy()
            images_np = images.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                img01 = images_np[i, 0]
                prob01 = probs[i, 0]
                gt01 = masks_np[i, 0]
                dice, iou, pred_sum, tgt_sum = _dice_iou_from_probs(prob01, gt01, thr=args.thr)
                px = _pixel_metrics_from_probs(prob01, gt01, thr=args.thr)

                pixel_tp += px["tp"]
                pixel_tn += px["tn"]
                pixel_fp += px["fp"]
                pixel_fn += px["fn"]

                dice_list.append(dice)
                iou_list.append(iou)

                prob_orig = cv2.resize(prob01, (800, 540))
                pred_hc_pixels = utils.fit_ellipse_and_measure_hc(prob_orig)
                px_size = float(pixel_sizes[i])
                hc_pred_mm = float(pred_hc_pixels * px_size) if np.isfinite(px_size) else float("nan")
                hc_true_mm = float(hc_truths[i])
                abs_err = float(abs(hc_pred_mm - hc_true_mm)) if np.isfinite(hc_pred_mm) and np.isfinite(hc_true_mm) else float("nan")
                if np.isfinite(abs_err):
                    mae_list.append(abs_err)

                filename = str(filenames[i])
                rows.append(
                    dict(
                        filename=filename,
                        dice=dice,
                        iou=iou,
                        pred_pixels=pred_sum,
                        gt_pixels=tgt_sum,
                        pixel_size_mm=px_size,
                        hc_pred_pixels=float(pred_hc_pixels),
                        hc_pred_mm=hc_pred_mm,
                        hc_true_mm=hc_true_mm,
                        hc_abs_error_mm=abs_err,
                        tp=px["tp"],
                        tn=px["tn"],
                        fp=px["fp"],
                        fn=px["fn"],
                    )
                )

                if args.split != "test":
                    y_true_all.append((gt01.reshape(-1) >= 0.5).astype(np.uint8))
                    y_score_all.append(prob01.reshape(-1).astype(np.float32))

                if len(rows) <= args.num_viz:
                    pred_bin = (prob01 >= args.thr).astype(np.float32)
                    cv2.imwrite(str(masks_dir / f"{Path(filename).stem}_pred.png"), (pred_bin * 255).astype(np.uint8))
                    fig = plt.figure(figsize=(12, 4))
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax1.imshow(img01, cmap="gray")
                    ax1.set_title("Image")
                    ax1.axis("off")
                    ax2.imshow(_overlay(img01, gt01))
                    ax2.set_title("GT overlay")
                    ax2.axis("off")
                    ax3.imshow(_overlay(img01, (prob01 >= args.thr).astype(np.float32)))
                    ax3.set_title(f"Pred overlay (dice={dice:.3f})")
                    ax3.axis("off")
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"{Path(filename).stem}_viz.png", dpi=140)
                    plt.close()

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)
    if args.split == "test":
        pred_df[["filename", "pixel_size_mm", "hc_pred_pixels", "hc_pred_mm"]].to_csv(run_dir / "test_predictions.csv", index=False)

    agg["mean_dice"] = float(np.mean(dice_list)) if dice_list else float("nan")
    agg["mean_iou"] = float(np.mean(iou_list)) if iou_list else float("nan")
    agg["hc_mae_mm"] = float(np.mean(mae_list)) if mae_list else float("nan")

    denom = max(pixel_tp + pixel_tn + pixel_fp + pixel_fn, 1)
    agg["pixel_acc"] = float((pixel_tp + pixel_tn) / denom)
    agg["pixel_precision"] = float(pixel_tp / max(pixel_tp + pixel_fp, 1))
    agg["pixel_recall"] = float(pixel_tp / max(pixel_tp + pixel_fn, 1))

    if args.split != "test":
        y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.uint8)
        y_score = np.concatenate(y_score_all) if y_score_all else np.array([], dtype=np.float32)
        if y_true.size > 0:
            _export_pr_curves(y_true, y_score, run_dir / "pixel_pr_curve.png")
            _export_roc_curve(y_true, y_score, run_dir / "pixel_roc_curve.png")

    with open(run_dir / "summary.json", "w") as f:
        json.dump(agg, f, indent=2)

    print(f"Saved evaluation to: {run_dir}")
    print(json.dumps({k: agg[k] for k in ["weights", "split", "mean_dice", "mean_iou", "hc_mae_mm", "pixel_acc"]}, indent=2))


if __name__ == "__main__":
    main()

