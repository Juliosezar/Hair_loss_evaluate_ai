import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import copy

# ==========================================
# 1. تنظیمات اولیه و تشخیص سخت‌افزار
# ==========================================
# مسیر داده‌های تقسیم‌بندی شده (Train/Val)
DATA_DIR = './dataset/5- slplited/' 

# استفاده از GPU اگر موجود باشد
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")

# بررسی وجود دیتاست
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"پوشه {DATA_DIR} پیدا نشد! لطفا ابتدا کد split-folders را اجرا کنید.")

# ==========================================
# 2. تعریف تابع اصلی آموزش (کاملا خودکار)
# ==========================================
def train_model_with_params(batch_size, num_epochs):
    print(f"\n{'='*40}")
    print(f"🚀 STARTING TRAINING | Batch: {batch_size} | Epochs: {num_epochs}")
    print(f"{'='*40}")

    # الف) آماده‌سازی داده‌ها (Data Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # افزایش داده مصنوعی
            transforms.RandomRotation(15),     # کمی چرخش
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    # ب) ساخت مدل خام (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # تغییر لایه آخر
    model = model.to(device)

    # ج) تنظیمات بهینه‌ساز و تابع خطا
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # متغیرهای ذخیره بهترین مدل
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # برای ذخیره نتیجه نهایی این اجرا
    final_train_loss = 0.0
    final_train_acc = 0.0
    final_val_loss = 0.0
    final_val_acc = 0.0

    start_time = time.time()

    # د) حلقه آموزش (Epochs)
    for epoch in range(num_epochs):
        # نمایش وضعیت هر 5 دور یکبار (برای شلوغ نشدن کنسول)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # پیمایش روی دسته‌های داده (Batches)
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # ذخیره آمار آخرین دور برای گزارش
            if phase == 'train':
                final_train_loss = epoch_loss
                final_train_acc = epoch_acc
            else:
                final_val_loss = epoch_loss
                final_val_acc = epoch_acc
                
                # *** بخش طلایی: ذخیره بهترین مدل ***
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # ذخیره موقت در حافظه، در آخر فایل می‌کنیم
                    print(f"   --> New Best Acc: {best_acc:.4f} found at Epoch {epoch+1}")

    # هـ) پایان آموزش و ذخیره فایل
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # بارگذاری بهترین وزن‌ها روی مدل
    model.load_state_dict(best_model_wts)
    
    # ذخیره فایل فیزیکی
    save_path = f"./models/model_bs{batch_size}_ep{num_epochs}_trAcc{final_train_acc:.4f}_valAcc{final_val_acc:.4f}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved Best Model to: {save_path}")

    return {
        'bs': batch_size,
        'ep': num_epochs,
        'train_loss': final_train_loss,
        'train_acc': final_train_acc,
        'val_loss': final_val_loss,
        'val_acc': final_val_acc,
        'best_val_acc': best_acc # دقت بهترین مدلی که ذخیره شد
    }

# ==========================================
# 3. اجرای آزمایش‌های مختلف (Main Block)
# ==========================================
if __name__ == "__main__":
    batch_size_list = [8, 16, 32, 64] 
    epoch_list = [16, 18, 20 ,22, 24, 26, 28, 30, 32, 34, 36, 40, 42 , 44]

    results_list = []

    # اجرای حلقه روی تمام تنظیمات
    for bs in batch_size_list:
        for ep in epoch_list:
            try:
                res = train_model_with_params(bs, ep)
                results_list.append(res)
            except Exception as e:
                print(f"❌ Error with BS={bs}, EP={ep}: {e}")

    # ==========================================
    # 4. نمایش گزارش نهایی (جدول مقایسه)
    # ==========================================
    print("\n\n")
    print("="*80)
    print(f"{'FINAL REPORT':^80}")
    print("="*80)
    print(f"{'Batch':<8} | {'Epoch':<8} | {'Best Val Acc (Saved)':<20} | {'Final Val Loss':<15}")
    print("-" * 80)

    best_config = None
    highest_acc = 0.0

    for r in results_list:
        # تبدیل تنسور به عدد معمولی برای نمایش تمیز
        best_acc_val = r['best_val_acc'].item() if torch.is_tensor(r['best_val_acc']) else r['best_val_acc']
        final_loss_val = r['val_loss'] # Loss معمولا float معمولی است

        print(f"{r['bs']:<8} | {r['ep']:<8} | {best_acc_val:.4f}{' (Highest!)' if best_acc_val > highest_acc else '' :<15} | {final_loss_val:.4f}")
        
        if best_acc_val > highest_acc:
            highest_acc = best_acc_val
            best_config = r

    print("="*80)
    if best_config:
        print(f"🏆 WINNER CONFIGURATION: Batch Size {best_config['bs']} with {best_config['ep']} Epochs")
        print(f"   Please use file: 'best_model_bs{best_config['bs']}_ep{best_config['ep']}.pth' for your app.")
    print("="*80)
