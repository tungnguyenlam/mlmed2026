import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2

import dataset
import utils
from model import UNet

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
TARGET_SIZE = (256, 256)
MODELS_DIR = "models"
RESULTS_DIR = "results"

def train():
    utils.setup_folders(MODELS_DIR, RESULTS_DIR)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = dataset.get_dataloaders(batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    loss_fn = utils.BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    start_epoch = 0
    best_val_loss = float('inf')
    
    latest_ckpt = utils.get_latest_checkpoint(MODELS_DIR)
    if latest_ckpt:
        start_epoch, best_val_loss = utils.load_checkpoint(latest_ckpt, model, optimizer, scheduler, DEVICE)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, masks, _, _ in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        mae_mm = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, masks, pixel_sizes, hc_truths in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                
                for i in range(images.size(0)):
                    prob_mask_np = probs[i].squeeze().cpu().numpy()
                    prob_mask_orig = cv2.resize(prob_mask_np, (800, 540))
                    
                    pred_hc_pixels = utils.fit_ellipse_and_measure_hc(prob_mask_orig)
                    
                    pred_hc_mm = pred_hc_pixels * pixel_sizes[i].item()
                    truth_mm = hc_truths[i].item()
                    
                    mae_mm += abs(pred_hc_mm - truth_mm)
                    total_samples += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_mae = mae_mm / total_samples if total_samples > 0 else 0
        
        print(f"epoch {epoch+1}, train loss: {avg_train_loss}, val loss: {avg_val_loss}, MAE: {avg_mae} mm")
        
        scheduler.step(avg_val_loss)

        utils.save_checkpoint(epoch + 1, model, optimizer, scheduler, avg_val_loss, MODELS_DIR)

        utils.save_visualizations(epoch + 1, model, val_loader, DEVICE, RESULTS_DIR)
        utils.save_metrics(epoch + 1, avg_train_loss, avg_val_loss, avg_mae, RESULTS_DIR)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_unet_hc18.pth"))
            print("saved best model")

    print("Done training")

if __name__ == "__main__":
    train()
