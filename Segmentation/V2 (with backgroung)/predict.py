import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= CONFIGURATION =================
# ⚙️ SETTINGS
MODEL_PATH = 'best_skin_model_2ch.pth'  # Your trained 2-channel model
IMAGE_PATH = '../dataset/predict_tests/with_bg/photo_2026-02-13_21-09-50.jpg'  # <--- CHANGE THIS TO TEST DIFFERENT IMAGES
OUTPUT_FILENAME = 'hair_analysis_report.png'

# 🧠 MODEL PARAMETERS (Must match training)
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_SIZE = 512

# ================= 1. HELPER FUNCTIONS =================
def get_preprocessing(preprocessing_fn):
    """Transform image to Tensor for the AI"""
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
        
    return albu.Compose([
        albu.LongestMaxSize(max_size=INPUT_SIZE),
        albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, border_mode=0, value=[0,0,0]),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ])

def crop_to_head(image, head_mask, padding=20):
    """
    Finds the bounding box of the head and crops the image 
    so we zoom in on the important part.
    """
    coords = cv2.findNonZero(head_mask)
    if coords is None:
        return image, head_mask, None # No head found
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img, (x1, y1, x2, y2)

def classify_density(density_percent):
    """Returns a text label based on hair density."""
    if density_percent >= 90: return "Excellent / Thick", "green"
    if density_percent >= 75: return "Good / Normal", "blue"
    if density_percent >= 50: return "Thinning Detected", "orange"
    return "Severe Hair Loss", "red"

# ================= 2. MAIN PREDICTION ENGINE =================
def analyze_hair():
    print(f"🔄 Loading AI Model ({DEVICE})...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    # 1. Load Model
    model = smp.Unet(encoder_name=ENCODER, classes=CLASSES, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load & Preprocess Image
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image '{IMAGE_PATH}' not found!")
        return
        
    original_image = cv2.imread(IMAGE_PATH)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Preprocess for AI
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    transform = get_preprocessing(preprocessing_fn)
    
    # We keep a copy of the padded image for visualization alignment
    aug = albu.Compose([
        albu.LongestMaxSize(max_size=INPUT_SIZE),
        albu.PadIfNeeded(min_height=INPUT_SIZE, min_width=INPUT_SIZE, border_mode=0, value=[0,0,0])
    ])
    padded_image = aug(image=original_image)['image']
    
    # Create tensor
    sample = transform(image=original_image)
    tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(DEVICE)

    # 3. Predict
    print("🧠 Analyzing scalp topology...")
    with torch.no_grad():
        output = model(tensor)
        # Apply Sigmoid to get probabilities 0.0 - 1.0
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

    # 4. Extract Masks
    # Channel 0 = Head Shape
    # Channel 1 = Bald Spot
    head_prob = prob_map[0, :, :]
    bald_prob = prob_map[1, :, :]
    
    # Thresholding & Cleaning
    head_mask = (head_prob > 0.5).astype(np.uint8)
    bald_mask = (bald_prob > 0.5).astype(np.uint8)
    
    # Post-Process: Morphological Cleaning
    kernel = np.ones((5,5), np.uint8)
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bald_mask = cv2.morphologyEx(bald_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # LOGIC: A pixel cannot be bald if it's not part of the head
    bald_mask = cv2.bitwise_and(bald_mask, head_mask)

    # 5. Calculation
    total_head_pixels = np.count_nonzero(head_mask)
    bald_pixels = np.count_nonzero(bald_mask)
    
    if total_head_pixels == 0:
        print("⚠️ No head detected in image.")
        density = 0
        hair_pixels = 0
    else:
        hair_pixels = total_head_pixels - bald_pixels
        density = (hair_pixels / total_head_pixels) * 100

    classification, status_color = classify_density(density)

    # ================= 3. VISUALIZATION =================
    print(f"📊 Calculating visualization...")
    
    # Crop to head for better viewing
    crop_img, bbox = crop_to_head(padded_image, head_mask)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        crop_head_mask = head_mask[y1:y2, x1:x2]
        crop_bald_mask = bald_mask[y1:y2, x1:x2]
    else:
        crop_img = padded_image
        crop_head_mask = head_mask
        crop_bald_mask = bald_mask

    # Create the Figure
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # --- PANEL 1: Input Image ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(crop_img)
    ax1.set_title("Original Input (Zoomed)", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # --- PANEL 2: Digital Segmentation ---
    # Create a clean visualization of what the AI sees
    # Black Background, Green Head, Red Spot
    seg_vis = np.zeros_like(crop_img)
    seg_vis[crop_head_mask == 1] = [50, 205, 50]   # Lime Green (Hair)
    seg_vis[crop_bald_mask == 1] = [220, 20, 60]   # Crimson Red (Bald)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(seg_vis)
    ax2.set_title("AI Segmentation Layer", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Legend for Panel 2
    legend_elements = [
        patches.Patch(facecolor='#32CD32', edgecolor='w', label='Detected Hair'),
        patches.Patch(facecolor='#DC143C', edgecolor='w', label='Detected Baldness')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # --- PANEL 3: Final Overlay Report ---
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Draw transparent overlay
    overlay = crop_img.copy()
    
    # 1. Draw Head Contour (Cyan)
    contours, _ = cv2.findContours(crop_head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2) # Cyan Outline
    
    # 2. Fill Bald Spot (Red Transparent)
    overlay[crop_bald_mask == 1] = [255, 0, 0] # Make bald pixels pure red
    
    # Blend: 70% Original, 30% Overlay
    final_vis = cv2.addWeighted(crop_img, 0.7, overlay, 0.3, 0)
    
    ax3.imshow(final_vis)
    ax3.set_title(f"Density Analysis: {density:.1f}%", fontsize=14, fontweight='bold', color='black')
    ax3.axis('off')

    # Add Stats Box
    stats_text = (
        f"HEAD AREA : {total_head_pixels:,} px\n"
        f"BALD AREA : {bald_pixels:,} px\n"
        f"HAIR AREA : {hair_pixels:,} px\n"
        f"----------------------\n"
        f"DENSITY   : {density:.1f}%\n"
        f"STATUS    : {classification}"
    )
    
    # Place text box in top-left
    ax3.text(10, 30, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

    # Add main title to the whole figure
    plt.suptitle(f"HAIR DENSITY REPORT: {classification.upper()}", 
                 fontsize=20, fontweight='bold', color=status_color, y=1.05)

    # Save and Show
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', dpi=150)
    print(f"✅ Report saved to '{OUTPUT_FILENAME}'")
    plt.show()

if __name__ == "__main__":
    analyze_hair()
