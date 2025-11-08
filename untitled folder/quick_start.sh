#!/bin/bash
# أمر تشغيل سريع - Quick Start Command

cd "$(dirname "$0")" || exit

# التحقق من Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 غير مثبت!"
    exit 1
fi

# إنشاء البيئة الافتراضية إذا لم تكن موجودة
if [ ! -d "venv" ]; then
    echo "📦 إنشاء البيئة الافتراضية..."
    python3 -m venv venv
fi

# تفعيل البيئة
source venv/bin/activate

# تثبيت المكتبات إذا لم تكن مثبتة
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 تثبيت المكتبات..."
    pip install -q -r requirements.txt 2>/dev/null || pip install -q Flask yt-dlp
fi

# تشغيل التطبيق
echo "🚀 تشغيل التطبيق..."
echo "🌐 افتح المتصفح على: http://localhost:5000"
echo "🛑 للإيقاف: اضغط CTRL+C"
echo ""

python3 app.py
