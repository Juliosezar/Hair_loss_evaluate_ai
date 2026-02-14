import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
MODEL_PATH = 'best_skin_model.pth' 
IMAGE_PATH = '/home/sezar/x/photo_2026-02-13_21-09-50.jpg' # Change to your target image

ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE = 512

# ================= 1. PREPROCESSING =================
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_validation_augmentation():
    return albu.Compose([
        albu.LongestMaxSize(max_size=INPUT_SIZE),
        albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, border_mode=0, value=[0, 0, 0]),
    ])

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ])

# ================= 2. POST-PROCESSING =================
def post_process_mask(mask_prob, threshold=0.5):
    # Thresholding
    mask = (mask_prob > threshold).astype(np.uint8)
    
    # Noise removal
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Keep largest blob (assumes one main bald area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.zeros_like(mask)
        final_mask[labels == largest_label] = 1
        return final_mask
    return mask

# ================= 3. MAIN PREDICTION =================
def predict():
    print(f"🔄 Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    # Load Model
    model = smp.Unet(encoder_name=ENCODER, classes=CLASSES, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load and prepare image
    image_raw = cv2.imread(IMAGE_PATH)
    if image_raw is None:
        print(f"❌ Error: Image '{IMAGE_PATH}' not found!")
        return
    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    
    # Standardize size for model
    aug_fn = get_validation_augmentation()
    image_padded = aug_fn(image=image_rgb)['image']
    
    # Preprocess for Neural Net
    preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    pp_fn = get_preprocessing(preprocess_fn)
    tensor = torch.from_numpy(pp_fn(image=image_padded)['image']).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(tensor)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    final_mask = post_process_mask(prob_mask)

    # === DENSITY CALCULATION FOR BACKGROUND IMAGES ===
    h, w = image_padded.shape[:2]
    
    # Create a Circular ROI to estimate the head area and exclude background
    head_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(head_roi, (w // 2, h // 2), int(min(h, w) * 0.42), 255, -1)
    
    total_head_pixels = np.count_nonzero(head_roi)
    # Only count bald pixels that fall inside the Head ROI
    bald_pixels = np.count_nonzero(np.logical_and(final_mask, head_roi > 0))
    
    hair_pixels = max(0, total_head_pixels - bald_pixels)
    density = (hair_pixels / total_head_pixels) * 100

    print(f"\n================ PREDICTION ================")
    print(f"📊 Estimated Head Area: {total_head_pixels} px")
    print(f"📊 Detected Bald Area:  {bald_pixels} px")
    print(f"🔥 Calculated Density:  {density:.2f}%")
    print(f"============================================\n")

    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_padded)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prob_mask, cmap='hot')
    plt.title("AI Confidence Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    overlay = image_padded.copy()
    overlay[final_mask == 1] = [255, 0, 0] # Red for baldness
    result = cv2.addWeighted(image_padded, 0.7, overlay, 0.3, 0)
    plt.imshow(result)
    plt.title(f"Density: {density:.1f}%")
    plt.axis('off')

    plt.savefig('latest_prediction.png')
    print("✅ Result saved to 'latest_prediction.png'")
    plt.show()

if __name__ == "__main__":
    predict()
