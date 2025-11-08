# حل مشكلة تثبيت Whisper على Python 3.13 (macOS)

## المشكلة
`openai-whisper==20231117` لا يدعم Python 3.13 بشكل كامل ويسبب خطأ `KeyError: '__version__'`

## الحلول السريعة

### ✅ الحل 1: استخدام Faster Whisper فقط (موصى به - الأفضل!)
```bash
source venv/bin/activate
pip install faster-whisper
```
**مميزات Faster Whisper:**
- أسرع بـ 4-5x من Whisper العادي
- متوافق مع Python 3.13
- أفضل أداء

### ✅ الحل 2: تثبيت Whisper من GitHub (أحدث إصدار)
```bash
source venv/bin/activate
pip install git+https://github.com/openai/whisper.git
```

### ✅ الحل 3: تثبيت جميع المكتبات بدون Whisper
```bash
source venv/bin/activate
pip install -r requirements.txt --ignore-installed openai-whisper
pip install faster-whisper  # استخدم هذا بدلاً منه
```

## تثبيت سريع لجميع المكتبات (Python 3.13)

```bash
# تأكد من تفعيل البيئة أولاً
source venv/bin/activate

# 1. المكتبات الأساسية
pip install Flask==3.0.0 Werkzeug==3.0.1
pip install --upgrade yt-dlp
pip install requests beautifulsoup4 lxml tqdm jsonschema

# 2. NumPy و Pillow
pip install "numpy>=1.24.3" "Pillow>=10.1.0"

# 3. معالجة الفيديو
pip install pydub ffmpeg-python moviepy

# 4. الترجمة
pip install deep-translator googletrans

# 5. معالجة الترجمة
pip install pysrt asstosrt webvtt-py

# 6. PyTorch (CPU version - أصغر حجماً على macOS)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 7. Faster Whisper (أفضل من Whisper العادي)
pip install faster-whisper

# 8. Whisper (اختياري - محاولة من GitHub)
pip install git+https://github.com/openai/whisper.git || echo "Whisper اختياري - Faster Whisper كافٍ"
```

## استخدام السكريبت المحدث

```bash
./install_requirements.sh
```

السكريبت المحدث يتعامل مع Python 3.13 تلقائياً ويستخدم Faster Whisper إذا فشل تثبيت Whisper العادي.

## ملاحظات مهمة

1. **Faster Whisper كافٍ تماماً** - لا تحتاج Whisper العادي إذا كان Faster Whisper مثبتاً
2. **التطبيق يعمل بدون Whisper** - إذا كان Faster Whisper مثبتاً
3. **PyTorch CPU** - أصغر حجماً على macOS (عدة GB أقل)

## التحقق من التثبيت

```bash
source venv/bin/activate
python3 check_requirements.py
```

إذا ظهر Faster Whisper كـ ✅، فكل شيء جاهز!
