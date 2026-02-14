from PIL import Image
import os

folder_path = './dataset/resized_with_bg/'
print("🚀 در حال پاکسازی فایل‌های JPEG...")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        path = os.path.join(folder_path, filename)
        try:
            # باز کردن با پیلو
            img = Image.open(path)
            # تبدیل به RGB (برای حذف پروفایل‌های رنگی خراب)
            img = img.convert('RGB')
            # ذخیره مجدد روی همان فایل (این کار هدرهای خراب را اصلاح میکند)
            img.save(path, quality=95, subsampling=0)
        except Exception as e:
            print(f"❌ فایل {filename} به شدت خراب است و قابل اصلاح نیست: {e}")

print("✅ پاکسازی تمام شد. حالا دوباره train.py را اجرا کنید.")
