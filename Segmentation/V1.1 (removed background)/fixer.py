import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
IMAGES_DIR = '../dataset/3- resized/'
MASKS_OUTPUT_DIR = './masks/' 

os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)

def create_otsu_mask(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    h, w = img.shape[:2]
    
    # 1. Focus on the Center (Create a circular Region of Interest)
    # This ignores the background corners immediately
    roi_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_circle, (w//2, h//2), int(min(h, w) * 0.45), 255, -1)
    
    # 2. Convert to Grayscale for Brightness Analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply the ROI to the grayscale image (Make background black)
    gray_roi = cv2.bitwise_and(gray, gray, mask=roi_circle)
    
    # 3. Blur slightly to remove single hairs (noise)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # 4. OTSU'S THRESHOLDING (The Magic Step)
    # This automatically finds the best brightness cutoff to separate 
    # the "Light Scalp" from the "Dark Hair/Background"
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. SKIN COLOR FILTER (Secondary Check)
    # We also verify that the bright area is actually skin-colored 
    # (Prevents selecting white shirts or bright lights)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Generous skin range
    lower_skin = np.array([0, 10, 50], dtype=np.uint8)
    upper_skin = np.array([50, 255, 255], dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 6. COMBINE: Must be Bright AND Skin-Colored
    final_mask = cv2.bitwise_and(bright_mask, color_mask)
    
    # 7. Clean up (Morphology)
    # Close small holes, remove small dots
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 8. Keep only the largest blob (The Bald Spot)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(final_mask)
    if num_labels > 1:
        # Find the largest area that isn't the background
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask[labels == largest_label] = 255
        
    return cleaned_mask

def main():
    print("🚀 Running Adaptive Otsu Mask Generation...")
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(IMAGES_DIR, filename)
        mask = create_otsu_mask(img_path)
        
        if mask is not None:
            save_name = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(MASKS_OUTPUT_DIR, save_name), mask)

if __name__ == "__main__":
    main()
