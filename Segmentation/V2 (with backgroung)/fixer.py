import cv2
import numpy as np
import os
from tqdm import tqdm


IMAGES_DIR = '../dataset/3- resized_with_bg/'
HEAD_MASKS_DIR = './mask_head/' 
OUTPUT_DIR = './masks_bald/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fill_holes(mask):
    """
    Finds the contours of the bald spot and fills them in.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    

    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    
    if not valid_contours:
        return filled_mask
        

    cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
    
    return filled_mask

def get_bald_mask_kmeans(img, head_mask):
    """
    Uses K-Means Clustering to separate Hair vs Skin dynamically.
    """

    kernel = np.ones((15, 15), np.uint8) 
    safe_head_mask = cv2.erode(head_mask, kernel, iterations=1)
    

    masked_img = cv2.bitwise_and(img, img, mask=safe_head_mask)
    

    lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)

    pixel_values = lab[safe_head_mask > 0]
    
    if len(pixel_values) == 0:
        return np.zeros_like(head_mask)

    pixel_values = np.float32(pixel_values)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    mean_L_0 = centers[0][0]
    mean_L_1 = centers[1][0]

    bald_label = 0 if mean_L_0 > mean_L_1 else 1
    

    bald_mask = np.zeros_like(head_mask)
    

    bald_pixels = (labels.flatten() == bald_label)

    bald_mask[safe_head_mask > 0] = bald_pixels.astype(np.uint8) * 255


    bald_mask = fill_holes(bald_mask)

    kernel_smooth = np.ones((11, 11), np.uint8)
    bald_mask = cv2.morphologyEx(bald_mask, cv2.MORPH_OPEN, kernel_smooth) # Remove noise
    bald_mask = cv2.morphologyEx(bald_mask, cv2.MORPH_CLOSE, kernel_smooth) # Connect gaps
    
    return bald_mask

def main():
    print("🚀 Running K-MEANS Bald Detection (The Nuclear Option)...")
    
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(IMAGES_DIR, filename)
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None: continue
        
        mask_name = os.path.splitext(filename)[0] + ".png"
        head_mask_path = os.path.join(HEAD_MASKS_DIR, mask_name)
        
        if not os.path.exists(head_mask_path):
            continue
            
        head_mask = cv2.imread(head_mask_path, 0)
        
        bald_mask = get_bald_mask_kmeans(img, head_mask)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, mask_name), bald_mask)

if __name__ == "__main__":
    main()
