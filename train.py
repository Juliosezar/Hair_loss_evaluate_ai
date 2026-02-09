import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# 1. تنظیمات اولیه
data_dir = 'split_dataset' # همان پوشه‌ای که در قدم قبل ساختیم
batch_size = 18            # مدل در هر مرحله چند عکس را با هم ببیند
num_epochs = 12        # مدل چند بار کل عکس‌ها را مرور کند
num_classes = 4            # تعداد گروه‌های شما

# 2. افزایش داده‌ها (Data Augmentation)
# این بخش باعث میشه 150 عکس شما برای مدل شبیه 1500 عکس به نظر برسه!
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # چرخش آینه‌ای تصادفی
        transforms.RandomRotation(20),     # کج کردن تصادفی عکس تا 20 درجه
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),     # برای تست دیگه عکس رو کج نمی‌کنیم
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. لود کردن داده‌ها
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(f"گروه‌های پیدا شده: {class_names}")

# بررسی استفاده از کارت گرافیک یا CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"مدل روی این پردازنده اجرا می‌شود: {device}")

# 4. ساخت مدل (Transfer Learning)
# استفاده از مدل ResNet18 که قبلا هوشمند شده است
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# تغییر لایه آخر مدل برای 4 گروه خودمان
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# تعیین نحوه محاسبه خطا و آپدیت مدل
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. حلقه آموزش (Training Loop)
print("شروع آموزش مدل...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # حالت یادگیری
        else:
            model.eval()   # حالت امتحان دادن

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("آموزش تمام شد!")

# ذخیره کردن مغز مدل!
torch.save(model.state_dict(), 'baldness_model.pth')
print("مدل با نام baldness_model.pth ذخیره شد.")
