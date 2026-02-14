import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as albu
from tqdm import tqdm

IMAGES_DIR = '../dataset/3- resized_with_bg/'
HEAD_MASKS_DIR = './mask_head/'
BALD_MASKS_DIR = './masks_bald/'
MODEL_SAVE_NAME = 'best_skin_model_2ch.pth'

ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0001
INPUT_SIZE = 512

class SkinSegmentationDataset(Dataset):
    def __init__(self, ids, images_dir, head_dir, bald_dir, augmentation=None, preprocessing=None):
        self.ids = ids
        self.images_dir = images_dir
        self.head_dir = head_dir
        self.bald_dir = bald_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        id = self.ids[i]
        
        # Read Image
        img_name = f"{id}.jpg"
        if not os.path.exists(os.path.join(self.images_dir, img_name)):
            img_name = f"{id}.png"
        img = cv2.imread(os.path.join(self.images_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_head = cv2.imread(os.path.join(self.head_dir, f"{id}.png"), 0)
        
        mask_bald = cv2.imread(os.path.join(self.bald_dir, f"{id}.png"), 0)
        
        if mask_head is None: mask_head = np.zeros(img.shape[:2], dtype=np.uint8)
        if mask_bald is None: mask_bald = np.zeros(img.shape[:2], dtype=np.uint8)

        # Normalize to 0.0 - 1.0
        mask_head = (mask_head > 127).astype('float32')
        mask_bald = (mask_bald > 127).astype('float32')
        
        mask = np.dstack([mask_head, mask_bald])

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask

    def __len__(self):
        return len(self.ids)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.7, border_mode=0),
        albu.RandomBrightnessContrast(p=0.5),
        albu.LongestMaxSize(max_size=INPUT_SIZE),
        albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, border_mode=0),
    ])

def get_validation_augmentation():
    return albu.Compose([
        albu.LongestMaxSize(max_size=INPUT_SIZE),
        albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, border_mode=0),
    ])

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
        albu.Lambda(mask=to_tensor),
    ])

def train():
    print(f"🚀 Initializing Dual-Folder Training...")
    
    # Get IDs present in ALL three folders
    img_ids = {os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR)}
    head_ids = {os.path.splitext(f)[0] for f in os.listdir(HEAD_MASKS_DIR)}
    bald_ids = {os.path.splitext(f)[0] for f in os.listdir(BALD_MASKS_DIR)}
    
    all_ids = list(img_ids.intersection(head_ids).intersection(bald_ids))
    all_ids.sort()
    print(f"✅ Found {len(all_ids)} valid samples.")
    
    split = int(0.8 * len(all_ids))
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=CLASSES, activation=None).to(DEVICE)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_ds = SkinSegmentationDataset(train_ids, IMAGES_DIR, HEAD_MASKS_DIR, BALD_MASKS_DIR, 
                                       augmentation=get_training_augmentation(), 
                                       preprocessing=get_preprocessing(preprocessing_fn))
    val_ds = SkinSegmentationDataset(val_ids, IMAGES_DIR, HEAD_MASKS_DIR, BALD_MASKS_DIR, 
                                     augmentation=get_validation_augmentation(), 
                                     preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Loss: BCE + Dice (Multilabel)
    criterion = lambda y_pred, y_true: smp.losses.DiceLoss(mode='multilabel')(y_pred, y_true) + nn.BCEWithLogitsLoss()(y_pred, y_true)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_iou = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                
                # Simple IoU Calc
                pr = (torch.sigmoid(outputs) > 0.5).float()
                inter = (pr * masks).sum()
                union = pr.sum() + masks.sum() - inter
                val_iou += ((inter + 1e-7) / (union + 1e-7)).item()

        avg_iou = val_iou / len(val_loader)
        print(f"   Val IoU: {avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            print(f"   ✨ Saved New Best Model")

if __name__ == '__main__':
    train()
