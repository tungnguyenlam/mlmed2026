
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt

def process_mask(mask_path, target_size=(256, 256)):
    mask = cv2.imread(mask_path, 0)
    
    if mask is None:
        raise ValueError("Could not load mask from path:", mask_path)

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_mask = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    else:
        filled_mask = mask

    mask = cv2.resize(filled_mask, target_size, interpolation=cv2.INTER_NEAREST)

    mask = (mask > 0).astype(np.float32)
    mask_tensor = torch.from_numpy(mask)

    return mask_tensor

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce = self.bce(inputs, targets)
 
        inputs_sigmoid = torch.sigmoid(inputs)

        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_score
        
        return bce + dice_loss

def fit_ellipse_and_measure_hc(mask_input):
    if torch.is_tensor(mask_input):
        mask_np = mask_input.squeeze().detach().cpu().numpy()
    else:
        mask_np = mask_input

    if mask_np.max() <= 1.0:
        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = mask_np.astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 5:
        return 0.0
    
    try:
        (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
        a = MA / 2
        b = ma / 2
        hc_pixels = np.pi * (3*(a+b) - np.sqrt((3*a + b) * (a + 3*b)))
        return hc_pixels
    except:
        return 0.0

def setup_folders(models_dir="models", results_dir="results"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

def get_latest_checkpoint(models_dir="models"):
    if not os.path.exists(models_dir):
        return None
    
    checkpoints = [f for f in os.listdir(models_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None

    epochs = [int(f.split("_")[2].split(".")[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    return os.path.join(models_dir, f"checkpoint_epoch_{latest_epoch}.pth")

def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, models_dir="models"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    path = os.path.join(models_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer, scheduler, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from checkpoint: {path} (Epoch {checkpoint['epoch']})")
    return start_epoch, checkpoint.get('val_loss', float('inf'))

def save_visualizations(epoch, model, val_loader, device, results_dir="results", num_samples=10):
    model.eval()
    
    all_images = []
    all_masks_pred = []
    all_masks_gt = []
    
    with torch.no_grad():
        for images, masks, _, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            for i in range(images.size(0)):
                all_images.append(images[i].cpu())
                all_masks_pred.append(probs[i].cpu())
                all_masks_gt.append(masks[i].cpu())
    
    num_samples = min(num_samples, len(all_images))
    indices = random.sample(range(len(all_images)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for row, idx in enumerate(indices):
        img = all_images[idx].squeeze().numpy()
        pred_mask = (all_masks_pred[idx].squeeze().numpy() > 0.5).astype(np.float32)
        gt_mask = all_masks_gt[idx].squeeze().numpy()
        
        img_rgb = np.stack([img, img, img], axis=-1)
        
        overlay_pred = img_rgb.copy()
        overlay_pred[pred_mask > 0.5] = overlay_pred[pred_mask > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
        
        overlay_gt = img_rgb.copy()
        overlay_gt[gt_mask > 0.5] = overlay_gt[gt_mask > 0.5] * 0.5 + np.array([0, 1, 0]) * 0.5
        
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_title("Original")
        axes[row][0].axis('off')
        
        axes[row][1].imshow(overlay_gt)
        axes[row][1].set_title("Ground Truth")
        axes[row][1].axis('off')
        
        axes[row][2].imshow(overlay_pred)
        axes[row][2].set_title("Prediction")
        axes[row][2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"visualization_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Visualization saved: {save_path}")

def save_metrics(epoch, train_loss, val_loss, mae, results_dir="results"):
    """Append metrics to a CSV file."""
    csv_path = os.path.join(results_dir, "metrics.csv")
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "mae_mm"])
        writer.writerow([epoch, train_loss, val_loss, mae])