import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from rembg import remove 


class_names = ['0_Healthy', '1_Mild_Loss', '2_Severe_Loss', '3_Bald'] 
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
try:
    model.load_state_dict(torch.load('./best_model_bs64_ep26.pth', map_location=device))
    model = model.to(device)
    model.eval()
except:
    print("model not found")


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    is_success, buffer = cv2.imencode(".jpg", img)
    output_array = remove(buffer.tobytes())
    img_no_bg = cv2.imdecode(np.frombuffer(output_array, np.uint8), cv2.IMREAD_UNCHANGED)
    
    if img_no_bg.shape[2] == 4: 
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
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2RGB)
        pil_image = Image.fromarray(crop_img)
        return pil_image
    else:
        return Image.open(image_path) 



data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_smart(image_path):
    print("⏳ Processing and removing background...")
    processed_img = preprocess_image(image_path)
    
    if processed_img is None:
        print("image not found!")
        return

    input_tensor = data_transform(processed_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    idx = predicted.item()
    print(f"\n✅ Final result:")
    print(f"Class: {class_names[idx]}")
    print(f"confidence: {confidence.item()*100:.1f}%")


if __name__ == "__main__":
    predict_smart('./dataset/predict_tests/no_bg/photo_2026-02-13_21-09-54.jpg')
