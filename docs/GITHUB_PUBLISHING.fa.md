# راهنمای انتشار پروژه در GitHub

این راهنما فرض می‌کند ZIP نهایی را دانلود و از حالت فشرده خارج کرده‌اید و پوشهٔ اصلی `cherry-tomato-vision-sorting` است.

## ۱. پیش‌نیازها

Git را نصب کنید و نام و ایمیل commitها را یک‌بار تنظیم کنید:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## ۲. بررسی محلی پیش از انتشار

داخل پوشهٔ پروژه environment بسازید، dependencyها را نصب و testها را اجرا کنید:

```bash
python -m venv .venv
```

فعال‌سازی در Linux یا macOS:

```bash
source .venv/bin/activate
```

فعال‌سازی در Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

سپس:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

در صورت در دسترس بودن یک ویدئوی نمونه، ابتدا یک smoke test کوتاه اجرا کنید:

```bash
python run_pipeline.py --config config/config.yaml --input inputs/sample.mp4 --output-dir outputs/smoke-test --device cpu --max-frames 120
```

## ۳. ساخت repository در GitHub

در GitHub یک repository جدید با نام `cherry-tomato-vision-sorting` بسازید. چون README و `.gitignore` از قبل داخل پروژه هستند، هنگام ساخت repository گزینه‌های افزودن README، license یا `.gitignore` را فعال نکنید.

Description پیشنهادی:

> End-to-end cherry tomato video sorting with YOLO, ByteTrack, ShuffleNetV2, temporal aggregation, and pixel-based size estimation.

Topicهای پیشنهادی:

`computer-vision`, `object-detection`, `object-tracking`, `yolo`, `bytetrack`, `pytorch`, `opencv`, `agritech`, `video-processing`

## ۴. ساخت تاریخچه Git محلی

در ریشهٔ پروژه اجرا کنید:

```bash
git init
git status
git add .
git status
git commit -m "Initial public release"
git branch -M main
```

خروجی `git status` را پیش از commit بخوانید. پوشه‌هایی مثل `.venv`، `outputs`، cacheها یا فایل‌های موقت نباید staged شده باشند.

## ۵. اتصال و push

روش HTTPS:

```bash
git remote add origin https://github.com/YOUR_USERNAME/cherry-tomato-vision-sorting.git
git push -u origin main
```

یا روش SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/cherry-tomato-vision-sorting.git
git push -u origin main
```

`YOUR_USERNAME` را با نام کاربری GitHub خود عوض کنید. برای HTTPS ممکن است به Personal Access Token نیاز داشته باشید؛ برای SSH باید کلید SSH حساب از قبل تنظیم شده باشد.

## ۶. کنترل نسخهٔ منتشرشده

بعد از push این موارد را در صفحهٔ GitHub بررسی کنید:

- GIF بالای بخش Demo بدون خطا نمایش داده شود.
- لینک جابه‌جایی README انگلیسی و فارسی کار کند.
- MP4 قابل دانلود باشد.
- فایل‌های مدل داخل `models/` موجود باشند.
- notebook داخل `notebooks/` در preview گیت‌هاب باز شود.
- فایل‌های خروجی محلی، cacheها و environment وارد repository نشده باشند.
- بخش About، description و topicها تکمیل شده باشند.

## ۷. محدودیت حجم فایل‌ها

فایل‌های فعلی زیر محدودیت ۱۰۰ مگابایت GitHub هستند. برای وزن‌ها یا ویدئوهای بزرگ‌تر در نسخه‌های بعدی از Git LFS یا GitHub Releases استفاده کنید؛ فایل حجیم را مستقیماً وارد history نکنید.

## ۸. به‌روزرسانی‌های بعدی

پس از هر تغییر:

```bash
git status
git add <changed-files>
git commit -m "Describe the change"
git push
```

بهتر است هر commit یک تغییر مشخص داشته باشد و قبل از push، testها دوباره اجرا شوند.
