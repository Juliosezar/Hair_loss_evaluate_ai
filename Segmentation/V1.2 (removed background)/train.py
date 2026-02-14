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

# ================= CONFIGURATION =================
IMAGES_DIR = '../dataset/3- resized/'
MASKS_DIR = './masks/'
MODEL_SAVE_NAME = 'best_skin_model.pth'

ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.0001
INPUT_SIZE = 512

# ================= 2. LOSS FUNCTION =================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.bce(y_pred, y_true)

# ================= 3. METRICS =================
def calculate_iou(pr, gt, th=0.5):
    # Convert logits to binary (0 or 1)
    pr = (torch.sigmoid(pr) > th).float()
    
    intersection = (pr * gt).sum()
    union = pr.sum() + gt.sum() - intersection
    
    # Smooth division to prevent NaN
    return (intersection + 1e-7) / (union + 1e-7)

# ================= 4. DATASET (FIXED) =================
class SkinSegmentationDataset(Dataset):
    def __init__(self, ids, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        id = self.ids[i]
        
        # --- FIX: Check which extension exists for this ID ---
        # We try .jpg first, then .png, then .jpeg
        img_name = f"{id}.jpg"
        if not os.path.exists(os.path.join(self.images_dir, img_name)):
            img_name = f"{id}.png"
        if not os.path.exists(os.path.join(self.images_dir, img_name)):
            img_name = f"{id}.jpeg"
            
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, f"{id}.png") # Masks are always png
        # ---------------------------------------------------
        
        # Read Image
        image = cv2.imread(img_path)
        if image is None:
            # If we still can't find it, print what we tried
            raise FileNotFoundError(f"Image not found: {img_path} (ID: {id})")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Mask
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Standardize mask (0 and 255 -> 0.0 and 1.0)
        mask = (mask > 127).astype('float32')
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

# ================= 5. AUGMENTATIONS =================
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_training_augmentation():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.7, border_mode=0),
        # Color robustness
        albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        albu.HueSaturationValue(p=0.5), 
        albu.RGBShift(p=0.5),
        # Resizing
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

# ================= 6. MAIN TRAINING LOOP =================
def train():
    print(f"🚀 Initializing training on: {DEVICE}")

    # 1. Setup Data
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(MASKS_DIR):
        print(f"❌ Error: Directories not found!")
        return

    img_ids = {os.path.splitext(f)[0] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    mask_ids = {os.path.splitext(f)[0] for f in os.listdir(MASKS_DIR) if f.lower().endswith('.png')}
    
    all_ids = list(img_ids.intersection(mask_ids))
    all_ids.sort()
    
    if not all_ids:
        print("❌ No matching image-mask pairs found!")
        return
    
    print(f"✅ Found {len(all_ids)} valid pairs.")
    np.random.shuffle(all_ids)

    # Split 80/20
    split = int(0.8 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    # 2. Setup Model & Preprocessing
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=CLASSES, 
        activation=None
    ).to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 3. Create Datasets
    train_dataset = SkinSegmentationDataset(
        train_ids, IMAGES_DIR, MASKS_DIR, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    val_dataset = SkinSegmentationDataset(
        val_ids, IMAGES_DIR, MASKS_DIR, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # --- SAFETY CHECK: Verify data is not empty before starting ---
    print("\n🔍 performing Safety Check on first batch...")
    sample_img, sample_mask = next(iter(train_loader))
    print(f"   Max value in mask tensor: {sample_mask.max().item()}")
    if sample_mask.max().item() == 0:
        print("❌ WARNING: The first batch of masks is completely black (all zeros)!")
        print("   Did you run generate_masks.py? The model will NOT learn if this persists.")
        # We don't stop, but we warn heavily.
    else:
        print("✅ Data looks good! Masks contain values > 0.")
    # -------------------------------------------------------------

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler to reduce LR if we get stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    best_iou = 0.0

    print("\n🏁 Starting Training Loop...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_iou = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks).item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- VALIDATION ---
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Update Scheduler (monitor Val IoU)
        scheduler.step(avg_val_iou)

        print(f"   Train Loss: {avg_train_loss:.4f}, IoU: {avg_train_iou:.4f}")
        print(f"   Val   Loss: {avg_val_loss:.4f},   IoU: {avg_val_iou:.4f}")

        # Save Best Model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            print(f"   ✨ Saved New Best Model (IoU: {best_iou:.4f})")

if __name__ == '__main__':
    train()
