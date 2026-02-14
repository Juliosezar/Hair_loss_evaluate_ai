import cv2
import os

INPUT_FOLDER = "../dataset/1- raw/"  
OUTPUT_FOLDER = "../dataset/3- resized_with_bg/" 
TARGET_WIDTH = 1024             

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def resize_maintain_aspect(img, target_w):
    h, w = img.shape[:2]
    
    aspect_ratio = h / w
    target_h = int(target_w * aspect_ratio)
    
 
    final_w = (target_w // 32) * 32
    final_h = (target_h // 32) * 32
    
    resized = cv2.resize(img, (final_w, final_h), interpolation=cv2.INTER_AREA)
    return resized

print("🚀 شروع تغییر سایز تصاویر...")

count = 0
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is not None:
            processed_img = resize_maintain_aspect(img, TARGET_WIDTH)
            
            # ذخیره عکس
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(save_path, processed_img)
            count += 1
            print(f"✅ {filename} -> {processed_img.shape[1]}x{processed_img.shape[0]}")

print(f"\n✨ عملیات تمام شد. {count} عکس آماده لیبل‌زنی در پوشه '{OUTPUT_FOLDER}' هستند.")
