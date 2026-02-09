import cv2
import numpy as np
import os

def process_image(input_path, output_path, target_size=(224, 224)):
    # خواندن تصویر
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading {input_path}")
        return

    # 1. پیدا کردن کادر دور سر (جایی که سیاه نیست)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # بزرگترین کانتور احتمالا سر است
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # برش زدن (Crop)
        cropped = img[y:y+h, x:x+w]
        
        # 2. تغییر سایز به استاندارد هوش مصنوعی (مثلا 224x224)
        resized = cv2.resize(cropped, target_size)
        
        # ذخیره
        cv2.imwrite(output_path, resized)
        print(f"Processed: {output_path}")
    else:
        print(f"No content found in {input_path}")


        
if __name__ == '__main__':
    input_dir = '/home/sezar/Downloads/dataset/b'
    output_dir = '/home/sezar/Downloads/dataset/a'
    for i in os.listdir(input_dir):
        try:
            print(i)
            input_path = os.path.join(input_dir, i)
            output_path = os.path.join(output_dir, i)
            process_image(input_path, output_path)
        except:
            print('error')
            continue
