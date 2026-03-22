import argparse
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import dataset
import utils
from model import UNet, BCEDiceLoss


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
TARGET_SIZE = (256, 256)
SEED = 3407
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_lung")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_lung")
THR = 0.5
WEIGHT_DECAY = 1e-4
CLIP_NORM = 1.0


def train(total_epochs: int = EPOCHS):
    utils.set_seed(SEED)
    utils.setup_folders(MODELS_DIR, RESULTS_DIR)

    device = utils.pick_device()
    print(f"Using device: {device}")

    train_loader, val_loader = dataset.get_dataloaders(
        task="lung",
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
        num_workers=0,
    )

    model = UNet(n_channels=1, n_classes=1).to(device)
    loss_fn = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=LR * 0.05)

    start_epoch = 0
    best_val_loss = float("inf")
    latest = utils.get_latest_checkpoint(MODELS_DIR)
    if latest:
        start_epoch, best_val_loss = utils.load_checkpoint(latest, model, optimizer, scheduler, device=device)
        print(f"Resumed from {latest} (epoch {start_epoch}, best val loss {best_val_loss:.4f})")

    if start_epoch >= total_epochs:
        print(f"Already trained {start_epoch} epochs. Pass --epochs > {start_epoch} to continue.")
        return

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
        for images, masks, _, _ in loop:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            train_loss += float(loss.item())
            loop.set_postfix(loss=float(loss.item()))

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        dice_list = []
        iou_list = []
        pixel_tp = pixel_tn = pixel_fp = pixel_fn = 0
        viz_items = []

        with torch.no_grad():
            for images, masks, _, meta in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += float(loss.item())
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                images_np = images.detach().cpu().numpy()

                for i in range(images_np.shape[0]):
                    img01 = images_np[i, 0]
                    prob01 = probs[i, 0]
                    gt01 = masks_np[i, 0]
                    dice, iou = utils.dice_iou_from_probs(prob01, gt01, thr=THR)
                    px = utils.pixel_metrics_from_probs(prob01, gt01, thr=THR)
                    dice_list.append(dice)
                    iou_list.append(iou)
                    pixel_tp += px["tp"]
                    pixel_tn += px["tn"]
                    pixel_fp += px["fp"]
                    pixel_fn += px["fn"]
                    if len(viz_items) < 12:
                        viz_items.append(
                            dict(
                                img=img01,
                                gt=gt01,
                                pred=(prob01 >= THR).astype(np.float32),
                                filename=str(meta["filename"][i]),
                            )
                        )

        avg_val_loss = val_loss / max(len(val_loader), 1)
        mean_dice = float(np.mean(dice_list)) if dice_list else float("nan")
        mean_iou = float(np.mean(iou_list)) if iou_list else float("nan")
        denom = max(pixel_tp + pixel_tn + pixel_fp + pixel_fn, 1)
        pixel_acc = float((pixel_tp + pixel_tn) / denom)
        pixel_precision = float(pixel_tp / max(pixel_tp + pixel_fp, 1))
        pixel_recall = float(pixel_tp / max(pixel_tp + pixel_fn, 1))

        is_best = avg_val_loss < best_val_loss
        print(
            f"epoch {epoch+1}/{total_epochs}, train loss: {avg_train_loss:.4f}, val loss: {avg_val_loss:.4f}, "
            f"dice: {mean_dice:.4f}, iou: {mean_iou:.4f}, pixel_acc: {pixel_acc:.4f}"
            + (" [best]" if is_best else "")
        )

        scheduler.step()

        utils.save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_val_loss, MODELS_DIR)
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best.pth"))
            print(f"  -> Saved best weights (val loss {best_val_loss:.4f})")

        utils.save_metrics_row(
            os.path.join(RESULTS_DIR, "metrics.csv"),
            dict(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                mean_dice=mean_dice,
                mean_iou=mean_iou,
                pixel_acc=pixel_acc,
                pixel_precision=pixel_precision,
                pixel_recall=pixel_recall,
            ),
        )
        if viz_items:
            utils.save_viz_folder(os.path.join(RESULTS_DIR, f"viz_epoch_{epoch+1}"), viz_items)

    print("Done training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total epochs to train (use > current to continue)")
    args = parser.parse_args()
    train(total_epochs=args.epochs)

