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

IMAGES_DIR = '../dataset/3- resized/'
MASKS_DIR = './masks/'
MODEL_SAVE_NAME = 'best_skin_model.pth'

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0001
INPUT_SIZE = 320

def find_files(directory):
    files_map = {}
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    if not os.path.exists(directory):
        return {}

    for f in os.listdir(directory):
        if f.lower().endswith(valid_extensions):
            name_id = os.path.splitext(f)[0]
            files_map[name_id] = f
    return files_map

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.bce(y_pred, y_true)

def calculate_iou(pr, gt, th=0.5):
    pr = (torch.sigmoid(pr) > th).float()
    intersection = (pr * gt).sum()
    union = pr.sum() + gt.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)

class SkinSegmentationDataset(Dataset):
    def __init__(self, ids, image_map, mask_map, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = ids
        self.image_map = image_map 
        self.mask_map = mask_map   
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        id_ = self.ids[i]
        
        img_filename = self.image_map[id_]
        mask_filename = self.mask_map[id_]
        
        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Binarize mask (ensure it's 0 or 1)
        mask = (mask > 0).astype('float32')
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.2),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.7, border_mode=0),
        albu.RandomBrightnessContrast(p=0.5),
        albu.HueSaturationValue(p=0.3),
        albu.GaussianBlur(p=0.2),
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
    print(f"🚀 Initializing training on: {DEVICE}")

    image_map = find_files(IMAGES_DIR)
    mask_map = find_files(MASKS_DIR)
    
    all_ids = list(set(image_map.keys()) & set(mask_map.keys()))
    all_ids.sort()
    
    if not all_ids:
        print("❌ No matching image-mask pairs found! Check your directories.")
        print(f"   Images found: {len(image_map)}")
        print(f"   Masks found: {len(mask_map)}")
        return
    
    print(f"✅ Found {len(all_ids)} valid pairs (Images & Masks matched by name).")
    np.random.shuffle(all_ids)

    # Split Data
    split = int(0.8 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=CLASSES, 
        activation=None
    ).to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = SkinSegmentationDataset(
        train_ids, image_map, mask_map, IMAGES_DIR, MASKS_DIR, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    val_dataset = SkinSegmentationDataset(
        val_ids, image_map, mask_map, IMAGES_DIR, MASKS_DIR, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        model.train()
        train_loss, train_iou = 0, 0
        pbar = tqdm(train_loader, desc="Training")
        
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Note: calculate_iou returns a tensor, so .item() is correct
            train_iou += calculate_iou(outputs, masks).item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # VALIDATION 
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks).item()

        # Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Result -> Train Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f} | Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")

        # Save Best Model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            print(f"✨ New Best Model Saved! (IoU: {best_iou:.4f})")

if __name__ == '__main__':
    train()
