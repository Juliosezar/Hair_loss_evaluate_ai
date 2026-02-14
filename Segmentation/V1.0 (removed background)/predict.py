import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# Path to your trained model
MODEL_PATH = 'best_skin_model.pth' 

# Path to the image you want to test
IMAGE_PATH = '/home/sezar/x/x/photo_2026-02-13_21-09-54.jpg' 

# Settings (Must match train.py)
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE = 512

# ================= 1. PREPROCESSING HELPERS =================
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_validation_augmentation():
    """Resizes and pads the image exactly like we did during training."""
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
    """Cleans up the raw prediction."""
    # 1. Thresholding (Probability -> Binary)
    mask = (mask_prob > threshold).astype(np.uint8)
    
    # 2. Remove small noise (Morphological Opening)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Keep only the largest blob (Assumption: Baldness is one connected area)
    #    (Optional: Comment this out if you have alopecia areata with many spots)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = np.zeros_like(mask)
        final_mask[labels == largest_label] = 1
        return final_mask
    
    return mask

# ================= 3. MAIN PREDICTION FUNCTION =================
def predict():
    print(f"🔄 Loading model from {MODEL_PATH}...")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    model = smp.Unet(
        encoder_name=ENCODER, 
        classes=CLASSES, 
        activation=None
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file '{IMAGE_PATH}' not found!")
        return

    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Preprocess Image (Resize & Pad -> Tensor)
    #    A. Augmentation (Resize/Pad)
    aug_fn = get_validation_augmentation()
    sample = aug_fn(image=image)
    image_padded = sample['image']
    
    #    B. Neural Net Preprocessing (Normalize)
    preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    pp_fn = get_preprocessing(preprocess_fn)
    sample_tensor = pp_fn(image=image_padded)
    tensor = torch.from_numpy(sample_tensor['image']).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        output = model(tensor)
        # Convert logits to probability (0 to 1)
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # 5. Clean up the mask
    final_mask = post_process_mask(prob_mask, threshold=0.5)

    # 6. Calculate Density
    #    Total Head Area = Anything that isn't the black background
    gray_image = cv2.cvtColor(image_padded, cv2.COLOR_RGB2GRAY)
    _, head_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    
    total_head_pixels = np.count_nonzero(head_mask)
    bald_pixels = np.count_nonzero(final_mask)
    
    # Handle edge case where image is purely black
    if total_head_pixels == 0:
        density = 0.0
    else:
        hair_pixels = total_head_pixels - bald_pixels
        density = (hair_pixels / total_head_pixels) * 100

    print(f"\n================ PREDICTION RESULTS ================")
    print(f"📊 Total Head Area: {total_head_pixels} pixels")
    print(f"📊 Bald Area:       {bald_pixels} pixels")
    print(f"🔥 Hair Density:    {density:.2f}%")
    print(f"====================================================")

    # 7. Visualization
    plt.figure(figsize=(12, 6))
    
    # Original (Padded) Image
    plt.subplot(1, 3, 1)
    plt.imshow(image_padded)
    plt.title("Input Image (Resized)")
    plt.axis('off')

    # The AI's Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prob_mask, cmap='jet')
    plt.title("AI Heatmap")
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    overlay = image_padded.copy()
    # Paint red where the mask is 1
    overlay[final_mask == 1] = [255, 0, 0] 
    
    result = cv2.addWeighted(image_padded, 0.6, overlay, 0.4, 0)
    plt.imshow(result)
    plt.title(f"Result (Density: {density:.1f}%)")
    plt.axis('off')

    save_path = 'prediction_result.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Result saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    predict()
