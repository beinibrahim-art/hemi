# ============================================================================
# دليل تثبيت المكتبات في البيئة الافتراضية
# ============================================================================

## الطريقة الأولى: استخدام السكريبت التلقائي (موصى به)

```bash
# تشغيل السكريبت
./install_requirements.sh
```

## الطريقة الثانية: التثبيت اليدوي

### 1. إنشاء البيئة الافتراضية (إذا لم تكن موجودة)
```bash
python3 -m venv venv
```

### 2. تفعيل البيئة الافتراضية

**على Linux/macOS:**
```bash
source venv/bin/activate
```

**على Windows:**
```bash
venv\Scripts\activate
```

### 3. تحديث pip
```bash
pip install --upgrade pip
```

### 4. تثبيت المكتبات من requirements.txt
```bash
pip install -r requirements.txt
```

## الطريقة الثالثة: التثبيت التدريجي (للمشاكل)

إذا واجهت مشاكل في التثبيت، يمكنك تثبيت المكتبات بشكل منفصل:

```bash
# تفعيل البيئة أولاً
source venv/bin/activate  # Linux/macOS
# أو
venv\Scripts\activate     # Windows

# Core Web Framework
pip install Flask==3.0.0 Werkzeug==3.0.1

# Media Download
pip install --upgrade "yt-dlp>=2024.1.0"

# Audio/Video Processing
pip install moviepy==1.0.3 pydub==0.25.1 ffmpeg-python==0.2.0

# Speech Recognition
pip install "openai-whisper==20231117"
pip install "faster-whisper>=1.0.0"  # اختياري

# Translation
pip install "deep-translator==1.11.4" "googletrans==4.0.0rc1"

# Web Scraping
pip install "requests==2.31.0" "beautifulsoup4==4.12.2" "lxml==4.9.3"

# Additional Dependencies
pip install "numpy==1.24.3" "torch==2.1.0" "torchaudio==2.1.0" "Pillow==10.1.0"

# Subtitle Processing
pip install "pysrt==1.1.2" "asstosrt==0.1.6" "webvtt-py==0.4.6" "tqdm==4.66.1" "jsonschema==4.19.0"
```

## التحقق من التثبيت

```bash
# تشغيل سكريبت التحقق
python3 check_requirements.py
```

## تشغيل التطبيق

بعد التثبيت، تأكد من تفعيل البيئة الافتراضية ثم شغّل:

```bash
# تفعيل البيئة
source venv/bin/activate  # Linux/macOS
# أو
venv\Scripts\activate     # Windows

# تشغيل التطبيق
python3 app.py
```

## إيقاف البيئة الافتراضية

```bash
deactivate
```

## ملاحظات مهمة

1. **ffmpeg**: يجب تثبيته بشكل منفصل:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: تحميل من https://ffmpeg.org/download.html

2. **torch و torchaudio**: قد تكون كبيرة الحجم (عدة GB)، تأكد من وجود مساحة كافية

3. **Faster Whisper**: اختياري لكنه أسرع من Whisper العادي

4. **pyannote.audio**: معطل في requirements.txt لأنه يتطلب تسجيل في Hugging Face

## حل المشاكل الشائعة

### مشكلة: pip غير موجود
```bash
# Ubuntu/Debian
sudo apt install python3-pip

# macOS
brew install python3
```

### مشكلة: python3-venv غير موجود
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# Fedora
sudo dnf install python3-venv
```

### مشكلة: فشل تثبيت torch
```bash
# تثبيت torch بدون CUDA (أصغر حجماً)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### مشكلة: فشل تثبيت whisper
```bash
# تثبيت Whisper مع جميع التبعيات
pip install openai-whisper[all]
```
