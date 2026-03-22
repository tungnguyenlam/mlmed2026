import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_metrics(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df


def _plot_metrics(df: pd.DataFrame, title: str, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_loss"], label="val", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["mean_dice"], label="Dice", marker="o", markersize=3)
    ax.plot(df["epoch"], df["mean_iou"], label="IoU", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Dice / IoU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["pixel_acc"], label="acc", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("pixel accuracy")
    ax.set_title("Pixel accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["pixel_precision"], label="precision", marker="o", markersize=3)
    ax.plot(df["epoch"], df["pixel_recall"], label="recall", marker="s", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Pixel precision / recall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _print_tail(df: pd.DataFrame, name: str):
    last = df.iloc[-1].to_dict()
    print(f"\n{name} (last epoch):")
    for k, v in last.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


def _maybe_print_eval_summary(path: Path):
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    print(f"\nEval summary ({path.name}):")
    for k in ("mean_dice", "mean_iou", "pixel_acc", "pixel_precision", "pixel_recall", "n_samples"):
        if k in data:
            print(f"  {k}: {data[k]}")


def main():
    p = argparse.ArgumentParser(description="Plot training metrics from metrics.csv and optional eval summary.json")
    p.add_argument(
        "--lung_dir",
        default=os.path.join(SCRIPT_DIR, "results_lung"),
        help="Folder with results_lung/metrics.csv",
    )
    p.add_argument(
        "--infection_dir",
        default=os.path.join(SCRIPT_DIR, "results_infection"),
        help="Folder with results_infection/metrics.csv",
    )
    p.add_argument(
        "--eval_run",
        default=None,
        help="Optional path to an evaluate.py output folder (summary.json)",
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(SCRIPT_DIR, "plots"),
        help="Where to save training_curves_*.png",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    for name, folder in (("lung", args.lung_dir), ("infection", args.infection_dir)):
        csv_path = Path(folder) / "metrics.csv"
        df = _load_metrics(csv_path)
        if df is None:
            print(f"No metrics at {csv_path} (train first or check path).")
            continue
        _print_tail(df, name)
        out_png = out_dir / f"training_curves_{name}.png"
        _plot_metrics(df, f"{name} segmentation — training log", out_png)
        print(f"Saved plot: {out_png}")

    if args.eval_run:
        eval_path = Path(args.eval_run)
        _maybe_print_eval_summary(eval_path / "summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
