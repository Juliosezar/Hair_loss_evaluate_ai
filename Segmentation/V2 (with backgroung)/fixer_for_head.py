import cv2
import numpy as np
import os
from rembg import remove
from tqdm import tqdm

# ================= CONFIGURATION =================
IMAGES_DIR = '../dataset/3- resized_with_bg/'
OUTPUT_DIR = './mask_head/' 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_head_mask(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    try:
        # rembg returns BGRA. Alpha channel (index 3) is the mask.
        result = remove(img)
        alpha = result[:, :, 3]
        
        # Threshold to make it strictly Binary (0 or 255)
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
        # Fill small holes (morphology)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(img.shape[:2], dtype=np.uint8)

def main():
    print("🚀 Generating Head Masks (White = Head)...")
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(IMAGES_DIR, filename)
        mask = create_head_mask(img_path)
        
        if mask is not None:
            # Save as PNG
            save_name = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), mask)

if __name__ == "__main__":
    main()
