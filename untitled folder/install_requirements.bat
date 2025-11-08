@echo off
REM سكريبت لتثبيت المكتبات في البيئة الافتراضية - Windows

echo ================================================================================
echo        تثبيت المكتبات في البيئة الافتراضية
echo ================================================================================
echo.

REM التحقق من Python
echo [1/4] فحص Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python غير مثبت!
    pause
    exit /b 1
)
python --version
echo.

REM إنشاء البيئة الافتراضية
echo [2/4] إعداد البيئة الافتراضية...
if not exist "venv" (
    echo إنشاء البيئة الافتراضية...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] فشل إنشاء البيئة الافتراضية
        pause
        exit /b 1
    )
    echo تم إنشاء البيئة
) else (
    echo البيئة موجودة
)
echo.

REM تفعيل البيئة
echo [3/4] تفعيل البيئة الافتراضية...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] فشل تفعيل البيئة
    pause
    exit /b 1
)
echo تم التفعيل
echo.

REM تحديث pip
echo تحديث pip...
python -m pip install --upgrade pip -q
echo.

REM تثبيت المكتبات
echo [4/4] تثبيت المكتبات من requirements.txt...
echo.
echo جاري التثبيت... قد يستغرق هذا 5-15 دقيقة
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [WARNING] حدثت بعض الأخطاء أثناء التثبيت
    echo يمكنك المحاولة مرة أخرى
) else (
    echo.
    echo ================================================================================
    echo                    تم تثبيت جميع المكتبات بنجاح!
    echo ================================================================================
    echo.
    echo ملاحظات:
    echo   - البيئة الافتراضية مفعّلة الآن
    echo   - لتشغيل التطبيق: python app.py
    echo   - لإيقاف البيئة: deactivate
    echo.
)

pause
