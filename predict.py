import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from rembg import remove # استفاده از همان ابزار حذف پس‌زمینه

# 1. تنظیمات
class_names = ['0_Healthy', '1_Mild_Loss', '2_Severe_Loss', '3_Bald'] 
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. لود مدل
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
try:
    model.load_state_dict(torch.load('baldness_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
except:
    print("خطا: فایل مدل پیدا نشد.")

# 3. تابع پیش‌پردازش (همان کارهایی که موقع آموزش کردیم)
def preprocess_image(image_path):
    # الف) خواندن عکس با OpenCV
    img = cv2.imread(image_path)
    if img is None: return None
    
    # ب) حذف پس‌زمینه (با rembg)
    # چون rembg ورودی بایت می‌خواهد، تبدیلش می‌کنیم
    is_success, buffer = cv2.imencode(".jpg", img)
    output_array = remove(buffer.tobytes())
    img_no_bg = cv2.imdecode(np.frombuffer(output_array, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # ج) برش دور سر (Auto Crop)
    # تبدیل به خاکستری برای پیدا کردن لبه‌ها
    if img_no_bg.shape[2] == 4: # اگر کانال آلفا دارد
        alpha = img_no_bg[:, :, 3]
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        crop_img = img_no_bg[y:y+h, x:x+w]
        
        # تبدیل از OpenCV (BGR) به PIL (RGB)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2RGB)
        pil_image = Image.fromarray(crop_img)
        return pil_image
    else:
        return Image.open(image_path) # اگر نتوانست کراپ کند، اصلش را برگرداند

# 4. ترنسفورم نهایی برای مدل
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_smart(image_path):
    # اول تمیزکاری عکس
    print("در حال پردازش و حذف پس‌زمینه...")
    processed_img = preprocess_image(image_path)
    
    if processed_img is None:
        print("عکس خوانده نشد!")
        return

    # آماده‌سازی برای مدل
    input_tensor = data_transform(processed_img).unsqueeze(0).to(device)

    # پیش‌بینی
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    idx = predicted.item()
    print(f"\n✅ نتیجه نهایی:")
    print(f"کلاس: {class_names[idx]}")
    print(f"میزان اطمینان: {confidence.item()*100:.1f}%")

# اجرا
# نام عکس جدیدی که مدل تا حالا ندیده را اینجا بگذارید
# اول عکس را با همان روش‌های قبلی (rembg) کراپ کنید بهتر است، اما عکس خام هم جواب می‌دهد
predict_smart('/home/sezar/x5.jpg')
predict_smart('/home/sezar/x6.jpg')
