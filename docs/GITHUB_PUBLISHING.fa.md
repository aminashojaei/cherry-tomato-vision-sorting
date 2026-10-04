# به‌روزرسانی پروژه در GitHub

این مخزن از قبل در `aminashojaei/cherry-tomato-vision-sorting` ساخته شده است. برای تغییرهای بعدی، تاریخچهٔ آن را با `git clone` بگیرید؛ `git init` و ساخت مخزن تازه لازم نیست.

## ساخت شاخهٔ اصلاح

```bash
git clone https://github.com/aminashojaei/cherry-tomato-vision-sorting.git
cd cherry-tomato-vision-sorting
git switch -c fix/review-findings
```

فایل‌های اصلاح‌شده را روی همین شاخه اعمال کنید. از commit کردن `inputs/`، `outputs/`، `.venv/` و ویدئوهای آزمایشی خودداری کنید.

## بررسی پیش از انتشار

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==1.26.4 PyYAML==6.0.2
python -m unittest discover -s tests -v
git diff --check
git status
```

در Windows PowerShell، فعال‌سازی محیط مجازی با `.\.venv\Scripts\Activate.ps1` انجام می‌شود. تست‌های سبک تمام مدل‌ها را اجرا نمی‌کنند. اگر محیط کامل و ویدئوی ورودی دارید، `pip install -r requirements.txt` و یک اجرای کوتاه `run_pipeline.py --input inputs/sample.mp4 --output-dir outputs/smoke-test --device cpu --max-frames 120` هم انجام دهید.

## ثبت و ارسال

```bash
git add -A
git diff --cached --stat
git commit -m "Fix review findings and add CI"
git push -u origin fix/review-findings
```

بعد از push، در GitHub از شاخهٔ `fix/review-findings` به `main` یک Pull Request بسازید. نتیجهٔ GitHub Actions را بررسی و سپس PR را merge کنید. اگر به این مخزن دسترسی نوشتن ندارید، در حساب خودتان fork بسازید، تغییرات را به fork push کنید و از آن‌جا PR بدهید. برای HTTPS از روش احراز هویت GitHub استفاده کنید؛ گذرواژه یا token را در فایل‌های پروژه ذخیره نکنید.

انتخاب مجوز کد (`LICENSE`) و انتشار dataset باید با توجه به حق مالکیت کد، داده‌ها و وزن‌های مدل انجام شود.
