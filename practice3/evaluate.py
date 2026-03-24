import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import dataset
import utils
from model import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["lung", "infection"], default="lung")
    p.add_argument("--split", choices=["Train", "Val", "Test"], default="Val")
    p.add_argument("--models_dir", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--results_dir", default="results_eval")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--target_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--num_viz", type=int, default=24)
    p.add_argument("--use_lung_roi", action="store_true")
    args = p.parse_args()

    device = utils.pick_device()
    print(f"Using device: {device}")

    if args.models_dir is None:
        args.models_dir = "models_lung" if args.task == "lung" else "models_infection"

    root = dataset.resolve_covidqu_root(None)
    ds = dataset.CovidQuSegmentationDataset(
        root=root,
        task=args.task,
        split=args.split,
        target_size=tuple(args.target_size),
        include_lung_roi_for_infection=args.use_lung_roi,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(n_channels=1, n_classes=1).to(device)
    weights = args.weights or utils.find_best_weights(args.models_dir)
    if not weights:
        raise FileNotFoundError(f"No weights found (models_dir={args.models_dir})")
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    run_dir = Path(args.results_dir) / f"eval_{args.task}_{args.split}_{utils.timestamp()}"
    viz_dir = run_dir / "viz"
    pred_dir = run_dir / "pred_masks"
    viz_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dice_list = []
    iou_list = []
    pixel_tp = pixel_tn = pixel_fp = pixel_fn = 0

    viz_items = []

    with torch.no_grad():
        for images, masks, lung_roi, meta in dl:
            images = images.to(device)
            masks = masks.to(device)
            lung_roi = lung_roi.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()
            lung_np = lung_roi.detach().cpu().numpy()

            for i in range(images_np.shape[0]):
                img01 = images_np[i, 0]
                prob01 = probs[i, 0]
                gt01 = masks_np[i, 0]
                if args.task == "infection" and args.use_lung_roi:
                    prob01, gt01 = utils.apply_roi(prob01, gt01, lung_np[i, 0])

                dice, iou = utils.dice_iou_from_probs(prob01, gt01, thr=args.thr)
                px = utils.pixel_metrics_from_probs(prob01, gt01, thr=args.thr)
                dice_list.append(dice)
                iou_list.append(iou)
                pixel_tp += px["tp"]
                pixel_tn += px["tn"]
                pixel_fp += px["fp"]
                pixel_fn += px["fn"]

                filename = meta["filename"][i]
                cls = meta["cls"][i]

                rows.append(
                    dict(
                        filename=str(filename),
                        cls=str(cls),
                        dice=dice,
                        iou=iou,
                        tp=px["tp"],
                        tn=px["tn"],
                        fp=px["fp"],
                        fn=px["fn"],
                    )
                )

                if len(viz_items) < args.num_viz:
                    pred = (prob01 >= args.thr).astype(np.float32)
                    viz_items.append(
                        dict(img=img01, gt=gt01, pred=pred, filename=str(filename))
                    )
                    out_mask = (pred * 255).astype(np.uint8)
                    _, png_bytes = cv2.imencode(".png", out_mask)
                    (pred_dir / f"{Path(filename).stem}_pred.png").write_bytes(png_bytes.tobytes())

    mean_dice = float(np.mean(dice_list)) if dice_list else float("nan")
    mean_iou = float(np.mean(iou_list)) if iou_list else float("nan")
    denom = max(pixel_tp + pixel_tn + pixel_fp + pixel_fn, 1)
    pixel_acc = float((pixel_tp + pixel_tn) / denom)
    pixel_precision = float(pixel_tp / max(pixel_tp + pixel_fp, 1))
    pixel_recall = float(pixel_tp / max(pixel_tp + pixel_fn, 1))

    summary = dict(
        task=args.task,
        split=args.split,
        weights=str(weights),
        thr=args.thr,
        target_size=list(args.target_size),
        n_samples=len(ds),
        mean_dice=mean_dice,
        mean_iou=mean_iou,
        pixel_acc=pixel_acc,
        pixel_precision=pixel_precision,
        pixel_recall=pixel_recall,
        use_lung_roi=bool(args.use_lung_roi),
    )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)
    if viz_items:
        utils.save_viz_folder(str(run_dir / "qualitative"), viz_items)

    print(json.dumps(summary, indent=2))
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()

