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


def _plot_metrics(df: pd.DataFrame, train_steps_df: Optional[pd.DataFrame], title: str, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=12)

    ax = axes[0, 0]
    if train_steps_df is not None and "global_step" in train_steps_df and "train_loss" in train_steps_df:
        ax.plot(
            train_steps_df["global_step"],
            train_steps_df["train_loss"],
            label="train (step)",
            linewidth=1.0,
            alpha=0.8,
        )
        if "global_step_end" in df:
            val_x = df["global_step_end"]
            ax.set_xlabel("step")
        else:
            val_x = df["epoch"]
            ax.set_xlabel("step / epoch")
    else:
        ax.plot(df["epoch"], df["train_loss"], label="train (epoch avg)", marker="o", markersize=3)
        val_x = df["epoch"]
        ax.set_xlabel("epoch")
    ax.plot(val_x, df["val_loss"], label="val", marker="s", markersize=3)
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
    p = argparse.ArgumentParser()
    p.add_argument("--lung_dir", default=os.path.join(SCRIPT_DIR, "results_lung"))
    p.add_argument("--infection_dir", default=os.path.join(SCRIPT_DIR, "results_infection"))
    p.add_argument("--eval_run", default=None)
    p.add_argument("--out_dir", default=os.path.join(SCRIPT_DIR, "plots"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    for name, folder in (("lung", args.lung_dir), ("infection", args.infection_dir)):
        csv_path = Path(folder) / "metrics.csv"
        train_steps_path = Path(folder) / "train_steps.csv"
        df = _load_metrics(csv_path)
        train_steps_df = _load_metrics(train_steps_path)
        if df is None:
            print(f"skip {name}: no {csv_path}")
            continue
        _print_tail(df, name)
        out_png = out_dir / f"training_curves_{name}.png"
        _plot_metrics(df, train_steps_df, f"{name} train log", out_png)
        print(f"wrote {out_png}")

    if args.eval_run:
        eval_path = Path(args.eval_run)
        _maybe_print_eval_summary(eval_path / "summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
