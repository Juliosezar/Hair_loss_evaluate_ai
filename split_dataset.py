
import splitfolders

# این کد پوشه dataset شما را می‌گیرد و یک پوشه جدید به نام split_dataset می‌سازد
# 80 درصد عکس‌ها برای آموزش (train) و 20 درصد برای تست (val)
splitfolders.ratio('dataset', output="split_dataset", seed=42, ratio=(0.8, 0.2))
print("تقسیم داده‌ها با موفقیت انجام شد!")
