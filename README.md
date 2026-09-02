# CallForge

CallForge یک ابزار خط فرمان محلی و قابل‌ادامه برای ایندکس‌کردن فایل‌های MP3 و اجرای transcription با Codex و Whisper است. نام پروژه وابسته به نام پوشه یا PBX خاصی نیست و می‌توان هر دایرکتوری صوتی را به آن داد.

## قابلیت‌ها

- جست‌وجوی بازگشتی همهٔ فایل‌های `.mp3` با پشتیبانی از حروف بزرگ و کوچک
- استخراج hash، زمان و اندازهٔ فایل، مشخصات صوت و متادیتای نام تماس
- صف پایدار SQLite با retry، lease و تاریخچهٔ هر اجرا
- پردازش batch و اجرای هم‌زمان چند Codex worker
- ساخت Markdown هم‌نام کنار MP3
- ذخیرهٔ کامل همان Markdown به‌صورت نسخه‌دار در دیتابیس و رابطهٔ مستقیم با فایل صوتی
- واردکردن خودکار Markdownهای قبلی هنگام scan
- UI فارسی روی localhost برای جست‌وجو، مشاهدهٔ متادیتا، پخش صوت و خواندن transcript
- اجرای transcription یا transcription مجدد یک فایل مشخص مستقیماً از UI
- آماده برای stageهای آینده از طریق جدول‌های عمومی `jobs`، `processing_runs` و `artifacts`

نگاشت نوع تماس از روی پیشوند نام فایل انجام می‌شود:

- `external-`: تماس ورودی
- `internal-`: تماس داخلی به داخلی
- `out-`: تماس خروجی

## نصب

Python 3.11 یا جدیدتر لازم است. نصب‌کننده Codex CLI، ورود حساب، FFmpeg، backend مناسب Whisper و skill را بررسی می‌کند. نصب خودکار Codex به Node.js/npm نیاز دارد؛ ورود به حساب تعاملی است و در صورت نیاز باید `codex login` را اجرا کنید.

macOS و Linux:

```bash
cd callforge
./scripts/install.sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
cd callforge
.\scripts\install.ps1
.\.venv\Scripts\Activate.ps1
```

برای توسعه:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/callforge setup --yes
```

## شروع سریع

```bash
callforge init "/path/to/audio-root"
callforge status "/path/to/audio-root"
callforge run "/path/to/audio-root" --batch-size 5 --workers 2
callforge ui "/path/to/audio-root"
```

فرمان `ui` ابتدا دایرکتوری را scan می‌کند، مرورگر را باز می‌کند و رابط را به‌صورت پیش‌فرض روی `http://127.0.0.1:8765` بالا می‌آورد. فایل‌های frontend از قبل داخل پکیج build و بسته‌بندی شده‌اند؛ بنابراین نصب Node.js یا اجرای build جداگانه برای UI لازم نیست.

پس از انتخاب هر تماس، دکمهٔ «تبدیل به متن» همان فایل را در background پردازش می‌کند. اگر transcript فعلی وجود داشته باشد دکمه به «تبدیل مجدد» تغییر می‌کند؛ نسخهٔ قبلی تا موفق‌شدن پردازش جدید حفظ می‌شود و سپس transcript تازه به‌عنوان نسخهٔ جاری ثبت خواهد شد. تعداد پردازش‌های هم‌زمان UI از مقدار `workers` در `.callforge/config.toml` پیروی می‌کند.

برای انتخاب port یا جلوگیری از بازشدن خودکار مرورگر:

```bash
callforge ui "/path/to/audio-root" --port 9090 --no-open
```

سرور عمداً فقط به localhost متصل می‌شود. اگر `--host 0.0.0.0` انتخاب شود، فایل‌های صوتی و transcriptها بدون احراز هویت در شبکه قابل دسترسی خواهند بود.

`batch-size` تعداد فایل‌هایی است که در هر بار از صف برداشته می‌شود. `workers` تعداد پردازش‌های هم‌زمان است. این دو عمداً جدا هستند، چون مدل‌های بزرگ Whisper حافظهٔ زیادی مصرف می‌کنند.

برای دیدن کار قابل انجام بدون اجرای Codex:

```bash
callforge run "/path/to/audio-root" --batch-size 5 --workers 2 --dry-run
```

فرمان‌های دیگر:

```bash
callforge doctor
callforge scan "/path/to/audio-root"
callforge transcripts "/path/to/audio-root" --limit 20
callforge retry "/path/to/audio-root"
```

## فایل‌های هر workspace

با `init`، پوشهٔ زیر داخل دایرکتوری ورودی ساخته می‌شود:

```text
.callforge/
├── config.toml
├── callforge.sqlite3
├── logs/
├── models/
└── runs/
```

تنظیمات پیش‌فرض در `.callforge/config.toml` قابل تغییر‌اند:

```toml
[callforge]
batch_size = 5
workers = 2
max_attempts = 3
lease_seconds = 7200
language = "fa"
skill_name = "pbx-call-transcriber"
```

## مدل داده

```text
audio_files 1 ─── * jobs 1 ─── * processing_runs
     │                            │
     └──── 1 ─── * transcripts ──┘
     │                    │
     └──── 1 ─── * artifacts ────┘
```

`transcripts.content` متن کامل Markdown را نگه می‌دارد و `markdown_path` مسیر نسخهٔ کنار فایل را ثبت می‌کند. هر ویرایش جدید یک نسخهٔ تازه می‌سازد و فقط یکی `is_current=1` است. در نتیجه فایل کناری خروجی قابل‌خواندن برای انسان است، ولی دیتابیس منبع ساخت‌یافتهٔ مرحله‌های بعدی باقی می‌ماند.

## رفتار پردازش

هر worker، skill نصب‌شده را با `codex exec` صدا می‌زند. skill صوت را محلی decode و تقویت می‌کند، چند پاس مستقل Whisper می‌گیرد، بخش‌های مبهم را بازبینی می‌کند و Markdown را می‌سازد. CallForge بعد از اعتبارسنجی فایل، محتوا را در یک تراکنش به transcript و artifact مرتبط می‌کند و سپس job را کامل علامت می‌زند. خروجی و خطای هر Codex run در `.callforge/logs` نگه‌داری می‌شود.

فایل صوتی برای speech-to-text به سرویس transcription خارجی ارسال نمی‌شود. دسترسی شبکهٔ worker فقط برای دانلود اولیهٔ مدل‌های Whisper و ارتباط خود Codex فعال است و cache مدل داخل `.callforge/models` قرار می‌گیرد.

## تست

```bash
python -m pytest
```

تست‌ها Codex و دانلود مدل را mock می‌کنند؛ بنابراین سریع‌اند و فایل‌های واقعی را تغییر نمی‌دهند.
