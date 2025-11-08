# 🚀 دليل التشغيل - التطبيق المتكامل للترجمة والتحميل

## الطريقة الأولى: التشغيل التلقائي (موصى به) ⭐

### على Linux/macOS:
```bash
cd "/workspace/untitled folder"
chmod +x start_enhanced.sh
./start_enhanced.sh
```

### على Windows:
```cmd
cd "untitled folder"
start_enhanced.sh
```

**ماذا يفعل السكريبت:**
- ✅ يفحص Python
- ✅ ينشئ البيئة الافتراضية (إذا لم تكن موجودة)
- ✅ يفعّلها تلقائياً
- ✅ يثبت جميع المكتبات المطلوبة
- ✅ يشغّل التطبيق

---

## الطريقة الثانية: التشغيل اليدوي

### الخطوة 1: إنشاء البيئة الافتراضية (إذا لم تكن موجودة)

**Linux/macOS:**
```bash
cd "/workspace/untitled folder"
python3 -m venv venv
```

**Windows:**
```cmd
cd "untitled folder"
python -m venv venv
```

### الخطوة 2: تفعيل البيئة الافتراضية

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

**ملاحظة:** بعد التفعيل، ستظهر `(venv)` في بداية السطر

### الخطوة 3: تثبيت المكتبات

**الطريقة السريعة:**
```bash
pip install -r requirements.txt
```

**أو استخدام السكريبت:**
```bash
# Linux/macOS
./install_requirements.sh

# Windows
install_requirements.bat
```

### الخطوة 4: تشغيل التطبيق

```bash
python3 app.py
```

**أو:**
```bash
python app.py
```

---

## الطريقة الثالثة: استخدام سكريبت التثبيت أولاً

### 1. تثبيت المكتبات:
```bash
# Linux/macOS
./install_requirements.sh

# Windows
install_requirements.bat
```

### 2. ثم تشغيل التطبيق:
```bash
source venv/bin/activate  # Linux/macOS
# أو
venv\Scripts\activate     # Windows

python3 app.py
```

---

## 📍 الوصول للتطبيق

بعد التشغيل، افتح المتصفح على:

- **الصفحة الرئيسية:** http://localhost:5000
- **محرر الترجمة:** http://localhost:5000/subtitle-editor

### من جهاز آخر في نفس الشبكة:
- http://YOUR_IP:5000
- (سيظهر IP تلقائياً في Terminal)

---

## 🛑 إيقاف التطبيق

اضغط: **CTRL + C**

---

## 🔍 التحقق من المكتبات قبل التشغيل

```bash
source venv/bin/activate  # تفعيل البيئة أولاً
python3 check_requirements.py
```

---

## ⚠️ حل المشاكل الشائعة

### المشكلة: "command not found: python3"
**الحل:**
```bash
# استخدم python بدلاً من python3
python app.py
```

### المشكلة: "No module named 'flask'"
**الحل:**
```bash
# تأكد من تفعيل البيئة الافتراضية
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# ثم ثبت المكتبات
pip install -r requirements.txt
```

### المشكلة: "Port 5000 already in use"
**الحل:**
```bash
# غيّر المنفذ في app.py
# ابحث عن: app.run(..., port=5000)
# غيّره إلى: app.run(..., port=5001)
```

### المشكلة: Whisper لا يعمل على Python 3.13
**الحل:**
```bash
source venv/bin/activate
pip install faster-whisper  # أسرع وأفضل
# أو
pip install git+https://github.com/openai/whisper.git
```

---

## 📝 ملاحظات مهمة

1. **تأكد من تفعيل البيئة الافتراضية** قبل تشغيل التطبيق
2. **ffmpeg** يجب تثبيته بشكل منفصل:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: تحميل من https://ffmpeg.org/download.html
3. **البيئة الافتراضية** تبقى مفعّلة حتى تغلق Terminal أو تكتب `deactivate`

---

## 🎯 الطريقة الأسرع (موصى به)

```bash
cd "/workspace/untitled folder"
./start_enhanced.sh
```

هذا كل شيء! السكريبت يقوم بكل شيء تلقائياً.
