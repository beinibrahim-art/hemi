@echo off
chcp 65001 >nul
cls
title التطبيق المتكامل المحسن v4.0

echo ╔═══════════════════════════════════════════════════════════════════════════╗
echo ║                                                                           ║
echo ║        🎬 التطبيق المتكامل المحسن - v4.0                               ║
echo ║        دعم محسن لجميع المنصات + محرر الترجمة الاحترافي                 ║
echo ║                                                                           ║
echo ╚═══════════════════════════════════════════════════════════════════════════╝
echo.
echo.

REM Check Python
echo [1/5] فحص Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python غير مثبت!
    echo يرجى تثبيته من: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo ✅ Python جاهز
echo.

REM Virtual Environment
echo [2/5] إعداد البيئة الافتراضية...
if not exist "venv" (
    echo 📦 إنشاء البيئة الافتراضية...
    python -m venv venv
    if %errorlevel% equ 0 (
        echo ✅ تم إنشاء البيئة
    ) else (
        echo ❌ فشل الإنشاء
        pause
        exit /b 1
    )
) else (
    echo ✅ البيئة موجودة
)
echo.

REM Activate
echo [3/5] تفعيل البيئة...
call venv\Scripts\activate.bat
echo ✅ تم التفعيل
python -m pip install --upgrade pip -q
echo.

REM Install Libraries
echo [4/5] تثبيت المكتبات المطلوبة...
python -c "import flask" 2>nul
if %errorlevel% neq 0 (
    echo 📦 تثبيت المكتبات (قد يستغرق 3-10 دقائق)...
    
    echo   ⏳ Flask و Werkzeug...
    pip install Flask==3.0.0 Werkzeug==3.0.1 -q
    echo   ✅ Flask
    
    echo   ⏳ yt-dlp (محسن)...
    pip install --upgrade yt-dlp -q
    echo   ✅ yt-dlp
    
    echo   ⏳ Whisper (قد يستغرق وقتاً)...
    pip install openai-whisper -q 2>nul
    if %errorlevel% neq 0 (
        echo   ⚠️ Whisper اختياري - يمكن المتابعة بدونه
    ) else (
        echo   ✅ Whisper
    )
    
    echo   ⏳ مكتبات معالجة الفيديو...
    pip install moviepy pydub ffmpeg-python -q 2>nul
    echo   ✅ معالجة الفيديو
    
    echo   ⏳ الترجمة والترجمات...
    pip install deep-translator pysrt -q
    echo   ✅ الترجمة
    
    echo   ⏳ مكتبات إضافية...
    pip install requests beautifulsoup4 lxml tqdm -q
    echo   ✅ المكتبات الإضافية
) else (
    echo ✅ المكتبات مثبتة مسبقاً
    
    echo ⏳ التحقق من التحديثات...
    pip install --upgrade yt-dlp -q
    echo ✅ تم التحديث
)
echo.

REM Check Files
echo [5/5] التحقق من الملفات...
if not exist "unified_app_enhanced.py" (
    echo ❌ unified_app_enhanced.py غير موجود!
    pause
    exit /b 1
)
echo ✅ الملف الرئيسي موجود

REM Create directories
if not exist "templates" mkdir templates
if not exist "downloads" mkdir downloads
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "subtitles" mkdir subtitles
if not exist "static" mkdir static
echo ✅ المجلدات جاهزة

REM Check HTML files
if not exist "templates\index.html" (
    echo ⚠️ templates\index.html غير موجود
    echo يتم إنشاؤه تلقائياً عند التشغيل...
)
if not exist "templates\subtitle_editor.html" (
    echo ⚠️ templates\subtitle_editor.html غير موجود
    echo يتم إنشاؤه تلقائياً عند التشغيل...
)
echo.

REM Check ffmpeg
echo فحص الأدوات الإضافية...
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ ffmpeg متوفر - ممتاز لدمج الترجمة!
) else (
    echo ⚠️ ffmpeg غير متوفر
    echo.
    echo لتثبيت ffmpeg:
    echo 1. تحميل من: https://ffmpeg.org/download.html
    echo 2. فك الضغط وإضافة المسار إلى PATH
    echo.
    echo يمكن المتابعة بدونه لكن دمج الترجمة لن يعمل
)
echo.

REM Final message
echo ╔═══════════════════════════════════════════════════════════════════════════╗
echo ║                                                                           ║
echo ║                    ✅ كل شيء جاهز! جاري التشغيل...                       ║
echo ║                                                                           ║
echo ╚═══════════════════════════════════════════════════════════════════════════╝
echo.
echo ════════════════════════════════════════════════════════════════════════════
echo 🎬 التطبيق المتكامل المحسن v4.0
echo ════════════════════════════════════════════════════════════════════════════
echo.
echo ✅ البيئة الافتراضية مفعّلة
echo ✅ جميع المكتبات الأساسية مثبتة
echo.
echo 🌐 افتح المتصفح على:
echo.
echo    👉 http://localhost:5000 (الصفحة الرئيسية)
echo    👉 http://localhost:5000/subtitle-editor (محرر الترجمة)
echo.
echo 💡 المميزات الجديدة:
echo   ✨ دعم محسن لـ TikTok بطرق متعددة
echo   🎯 تحكم كامل في جودة التحميل
echo   🎨 محرر ترجمة احترافي مع معاينة حية
echo   🎬 دمج الترجمة بجودات متعددة (أصلي/عالي/متوسط/منخفض)
echo   🎨 تخصيص كامل للترجمة (الخط/الحجم/اللون/الخلفية/الموضع)
echo   📱 متوافق مع Mac و Windows
echo.
echo 🛑 لإيقاف الخادم: CTRL+C
echo ════════════════════════════════════════════════════════════════════════════
echo.

REM Open browser
timeout /t 2 /nobreak >nul
start http://localhost:5000

REM Run the app
python unified_app_enhanced.py
