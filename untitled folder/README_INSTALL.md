# 📦 دليل تثبيت المكتبات

## الطريقة السريعة (موصى بها)

### على Linux/macOS:
```bash
./install_requirements.sh
```

### على Windows:
```cmd
install_requirements.bat
```

## الطريقة اليدوية

### 1. إنشاء البيئة الافتراضية
```bash
python3 -m venv venv
```

### 2. تفعيل البيئة

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### 3. تحديث pip
```bash
pip install --upgrade pip
```

### 4. تثبيت المكتبات
```bash
pip install -r requirements.txt
```

## التحقق من التثبيت

```bash
python3 check_requirements.py
```

## تشغيل التطبيق

```bash
# تأكد من تفعيل البيئة أولاً
source venv/bin/activate  # Linux/macOS
# أو
venv\Scripts\activate     # Windows

# ثم شغّل التطبيق
python3 app.py
```

## ملاحظات

- **ffmpeg** يجب تثبيته بشكل منفصل
- **torch** قد يكون كبير الحجم (عدة GB)
- راجع `INSTALL.md` للتفاصيل الكاملة
