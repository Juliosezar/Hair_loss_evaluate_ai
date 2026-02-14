import cv2
import numpy as np
import os
from tqdm import tqdm

IMAGES_DIR = '../dataset/3- resized/'
MASKS_OUTPUT_DIR = './masks/'

os.makedirs(MASKS_OUTPUT_DIR, exist_ok=True)

def create_skin_mask(image_path):
    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    final_mask = np.zeros_like(mask)
    
    if num_labels > 1:
        sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        largest_label = sorted_indices[0]
        final_mask[labels == largest_label] = 255

    return final_mask


def main():
    print("Starting Mask Generation...")
    
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("❌ No images found in IMAGES_DIR!")
        return

    for filename in tqdm(files):
        img_path = os.path.join(IMAGES_DIR, filename)
        
        mask = create_skin_mask(img_path)
        if mask is not None:
            save_name = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(MASKS_OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, mask)
    
    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()
