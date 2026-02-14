import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# ================= 1. تنظیمات (مطابق با آموزش) =================
MODEL_PATH = 'best_skin_model.pth' 
IMAGE_PATH = '../dataset/predict_tests/no_bg/photo_2026-02-13_21-09-50.jpg' # عکسی که خودتان بک‌گراندش را سیاه کردید

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER = 'resnet34'
CLASSES = 1 # فقط پوست
INPUT_SIZE = 320
# =========================================================

def preprocess_with_padding(image, target_size):
    """تغییر سایز با حفظ نسبت ابعاد و اضافه کردن پدینگ (دقیقا مثل زمان آموزش)"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # ساخت بوم سیاه 320x320
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # قرار دادن عکس در مرکز بوم
    canvas[:new_h, :new_w] = resized
    
    return canvas, scale

def post_process_mask(mask):
    """حذف لکه‌های نویز و نگه داشتن بزرگترین ناحیه (پوست اصلی)"""
    mask = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask

    # پیدا کردن بزرگترین جزیره (به جز پس‌زمینه)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = np.zeros_like(mask)
    final_mask[labels == largest_label] = 1
    return final_mask

def predict():
    # ۱. بارگذاری مدل
    print(f"🔄 بارگذاری مدل از {MODEL_PATH}...")
    model = smp.Unet(encoder_name=ENCODER, classes=CLASSES, activation=None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # ۲. خواندن تصویر
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"❌ خطا: عکس در مسیر {IMAGE_PATH} پیدا نشد.")
        return
    
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # ۳. محاسبه مساحت فیزیکی کل سر (هر چیزی که سیاه مطلق نیست)
    # آستانه 10 برای حذف نویزهای احتمالی در پیکسل‌های سیاه
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, head_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    total_head_pixels = np.sum(head_mask > 0)
    
    if total_head_pixels == 0:
        print("❌ خطای فیزیکی: هیچ سری در عکس تشخیص داده نشد (عکس کاملا سیاه است؟)")
        return

    # ۴. پیش‌پردازش برای هوش مصنوعی
    img_padded, scale = preprocess_with_padding(img_rgb, INPUT_SIZE)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    img_input = preprocessing_fn(img_padded)
    img_input = img_input.transpose(2, 0, 1).astype('float32')
    tensor_input = torch.from_numpy(img_input).unsqueeze(0).to(DEVICE)

    # ۵. پیش‌بینی لکه‌های طاسی (Skin)
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # ۶. پس‌پردازش و بازگرداندن ماسک به سایز اصلی
    skin_mask_cleaned = post_process_mask(probs)
    
    # بریدن پدینگ‌های اضافی برای بازگشت به ابعاد واقعی
    h_orig, w_orig = img_rgb.shape[:2]
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    skin_mask_cropped = skin_mask_cleaned[:new_h, :new_w]
    skin_mask_final = cv2.resize(skin_mask_cropped, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # ۷. محاسبه نهایی تراکم مو
    skin_pixels = np.sum(skin_mask_final > 0)
    
    # فرمول: (کل سر - ناحیه طاس) / کل سر
    hair_pixels = max(0, total_head_pixels - skin_pixels)
    density = (hair_pixels / total_head_pixels) * 100

    print(f"\n============================")
    print(f"📊 مساحت کل سر: {total_head_pixels} پیکسل")
    print(f"📊 مساحت طاسی: {skin_pixels} پیکسل")
    print(f"🔥 تراکم نهایی مو: {density:.2f}%")
    print(f"============================\n")

    # ۸. نمایش گرافیکی
    overlay = img_rgb.copy()
    overlay[skin_mask_final > 0] = [255, 0, 0] # رنگ قرمز برای طاسی
    
    # ترکیب عکس اصلی با لایه قرمز
    result_view = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(result_view)
    plt.title(f"Predicted Hair Density: {density:.2f}%")
    plt.axis('off')
    plt.savefig('final_prediction.png')
    print("📂 تصویر نتیجه در 'final_prediction.png' ذخیره شد.")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        predict()
    else:
        print(f"❌ فایل {IMAGE_PATH} وجود ندارد.")
