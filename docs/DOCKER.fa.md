# راهنمای اجرای پروژه با Docker

نسخهٔ Docker این پروژه برای `Linux x86-64` و اجرای CPU ساخته شده است. Docker Desktop همین image را روی Windows و macOS اجرا می‌کند؛ در Apple Silicon اجرای `linux/amd64` از طریق emulation انجام می‌شود و ممکن است کندتر باشد.

## ۱. نصب Docker

در Windows و macOS، Docker Desktop را نصب و اجرا کنید. در Linux می‌توانید Docker Engine و افزونهٔ Compose را نصب کنید. سپس در Terminal یا PowerShell بررسی کنید:

```bash
docker --version
docker compose version
docker info
```

اگر `docker info` خطا داد، Docker daemon یا Docker Desktop هنوز اجرا نشده است.

## ۲. باز کردن پوشهٔ پروژه

ZIP را extract کنید و وارد ریشهٔ پروژه شوید؛ همان پوشه‌ای که `Dockerfile` و `compose.yaml` داخل آن هستند:

```bash
cd cherry-tomato-vision-sorting
```

## ۳. ساخت پوشه‌های ورودی و خروجی

در Linux و macOS:

```bash
mkdir -p inputs outputs
```

در Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force inputs, outputs
```

ویدئوها را داخل `inputs/` بگذارید. برای مثال:

```text
inputs/
├── video_01.mp4
└── video_02.mp4
```

این ویدئوها و خروجی‌ها به‌دلیل تنظیمات `.gitignore` وارد GitHub نمی‌شوند.

## ۴. ساخت image

```bash
docker compose build
```

در اولین build، Python image، PyTorch CPU و dependencyها دانلود می‌شوند؛ بنابراین زمان و فضای بیشتری لازم است. در طول build این مراحل خودکار انجام می‌شوند:

1. نصب dependencyهای pin‌شده از `requirements-docker-cpu.txt`
2. بررسی SHA-256 هر دو model checkpoint
3. کپی testها به stage مخصوص تست
4. اجرای کامل `python -m unittest discover -s tests -v`
5. ساخت runtime image فقط پس از موفقیت تست‌ها

stage نهایی marker تولیدشده در test stage را کپی می‌کند. در نتیجه اگر حتی یک test fail شود، image نهایی ساخته نمی‌شود.

## ۵. اجرای تمام ویدئوهای پوشه

```bash
docker compose run --rm tomato-sorting
```

Compose پوشهٔ `inputs/` را فقط‌خواندنی در `/data/input` و پوشهٔ `outputs/` را در `/data/output` mount می‌کند. نتیجه پس از پایان container داخل `outputs/` سیستم خودتان باقی می‌ماند.

## ۶. مشاهدهٔ خروجی‌ها

ساختار معمول خروجی:

```text
outputs/
├── batch_summary.json
├── video_01/
│   ├── annotated.mp4
│   ├── tomatoes.json
│   ├── size_measurements.csv
│   └── benchmark.json
└── video_02/
    └── ...
```

## ۷. اجرای smoke test روی یک ویدئو

پیش از اجرای کامل می‌توانید فقط ۱۲۰ فریم را پردازش کنید.

Linux و macOS:

```bash
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/inputs:/data/input:ro" \
  -v "$(pwd)/outputs:/data/output" \
  cherry-tomato-vision-sorting:cpu \
  --config /app/config/config.yaml \
  --input /data/input/video_01.mp4 \
  --output-dir /data/output/smoke-test \
  --device cpu \
  --max-frames 120
```

Windows PowerShell:

```powershell
docker run --rm --platform linux/amd64 `
  -v "${PWD}/inputs:/data/input:ro" `
  -v "${PWD}/outputs:/data/output" `
  cherry-tomato-vision-sorting:cpu `
  --config /app/config/config.yaml `
  --input /data/input/video_01.mp4 `
  --output-dir /data/output/smoke-test `
  --device cpu `
  --max-frames 120
```

## ۸. بررسی دستورات و تنظیمات

اعتبارسنجی Compose:

```bash
docker compose config
```

ساخت مستقیم test stage:

```bash
docker build --platform linux/amd64 --target test -t cherry-tomato-vision-sorting:test .
```

نمایش help برنامه داخل container:

```bash
docker run --rm --platform linux/amd64 cherry-tomato-vision-sorting:cpu --help
```

مشاهدهٔ imageهای ساخته‌شده:

```bash
docker image ls cherry-tomato-vision-sorting
```

## ۹. build مجدد پس از تغییر کد

```bash
docker compose build
docker compose run --rm tomato-sorting
```

برای build کاملاً تازه و بدون cache:

```bash
docker compose build --no-cache
```

## ۱۰. پاک‌کردن منابع Docker

حذف containerهای Compose و network مرتبط:

```bash
docker compose down
```

حذف image پروژه:

```bash
docker image rm cherry-tomato-vision-sorting:cpu
```

این دستورات فایل‌های `inputs/` و `outputs/` روی سیستم را حذف نمی‌کنند.

## ۱۱. نکات عیب‌یابی

- اگر `no matching manifest` دیدید، فرمان را با `--platform linux/amd64` اجرا کنید.
- اگر Docker حافظهٔ کافی ندارد، میزان RAM اختصاص‌یافته به Docker Desktop را افزایش دهید.
- اگر هیچ ویدئویی پیدا نشد، پسوند فایل و mount شدن `inputs/` را بررسی کنید.
- اگر نوشتن خروجی در Linux با permission error روبه‌رو شد، دسترسی پوشهٔ `outputs/` را برای کاربر خود اصلاح کنید.
- این image عمداً CPU-only است و گزینهٔ `--device cuda` برای آن مناسب نیست.
