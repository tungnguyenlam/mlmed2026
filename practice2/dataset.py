
import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import utils
import kagglehub

class HC18Dataset(Dataset):
    def __init__(self, data_frame, img_dir, transform=None, target_size=(256, 256)):
        self.df = data_frame
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        pixel_size = row['pixel size(mm)']
        hc_truth = row['head circumference (mm)']

        img_path = os.path.join(self.img_dir, filename)

        base, ext = os.path.splitext(filename)
        mask_filename = f"{base}_Annotation{ext}"
        mask_path = os.path.join(self.img_dir, mask_filename)

        if not os.path.exists(img_path):
             raise FileNotFoundError("Image not found:", img_path)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size)
        
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0) # turns to (1, H, W)

        if not os.path.exists(mask_path):
             mask = torch.zeros((1, *self.target_size), dtype=torch.float32)
        else:
            mask = utils.process_mask(mask_path, target_size=self.target_size)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask, float(pixel_size), float(hc_truth)

def get_dataloaders(batch_size=32, val_split=0.2, target_size=(256, 256), seed = 3407):

    path = kagglehub.dataset_download("thanhbnhphan/hc18-grand-challenge")
    
    train_dir = os.path.join(path, 'training_set')
    csv_path = os.path.join(path, 'training_set_pixel_size_and_HC.csv')

    full_df = pd.read_csv(csv_path)
    print(f"Len df: {len(full_df)}")
    
    train_df, val_df = train_test_split(full_df, test_size=val_split, random_state=seed)
    print("len training samples:", len(train_df))
    print("len validation samples:", len(val_df))

    train_dataset = HC18Dataset(train_df, train_dir, target_size=target_size)
    val_dataset = HC18Dataset(val_df, train_dir, target_size=target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

