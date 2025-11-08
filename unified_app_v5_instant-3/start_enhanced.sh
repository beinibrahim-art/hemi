#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${PURPLE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                           ║${NC}"
echo -e "${CYAN}║        🎬 التطبيق المتكامل للترجمة والتحميل - v5.0                               ║${NC}"
echo -e "${CYAN}║        دعم محسن لجميع المنصات + محرر الترجمة الاحترافي                 ║${NC}"
echo -e "${CYAN}║                                                                           ║${NC}"
echo -e "${PURPLE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
echo -e "${BLUE}[1/5] فحص Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 غير مثبت!${NC}"
    echo ""
    echo "للتثبيت:"
    echo "  • Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  • macOS: brew install python3"
    echo "  • Fedora: sudo dnf install python3"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Python جاهز: ${PYTHON_VERSION}${NC}"
echo ""

# Virtual Environment
echo -e "${BLUE}[2/5] إعداد البيئة الافتراضية...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 إنشاء البيئة الافتراضية...${NC}"
    
    # Check if venv module is available
    if ! python3 -m venv --help &> /dev/null; then
        echo -e "${YELLOW}⚠️ python3-venv غير متوفر، محاولة التثبيت...${NC}"
        
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y python3-venv
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3-venv
        elif command -v brew &> /dev/null; then
            echo -e "${GREEN}✅ venv متوفر على macOS${NC}"
        else
            echo -e "${RED}❌ يرجى تثبيت python3-venv يدوياً${NC}"
            exit 1
        fi
    fi
    
    python3 -m venv venv
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ تم إنشاء البيئة${NC}"
    else
        echo -e "${RED}❌ فشل الإنشاء${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ البيئة موجودة${NC}"
fi
echo ""

# Activate
echo -e "${BLUE}[3/5] تفعيل البيئة...${NC}"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ تم التفعيل${NC}"
else
    echo -e "${RED}❌ فشل التفعيل${NC}"
    exit 1
fi
echo -e "${YELLOW}⏳ تحديث pip...${NC}"
pip install --upgrade pip -q
echo ""

# Install Libraries
echo -e "${BLUE}[4/5] تثبيت المكتبات المطلوبة...${NC}"

# Check if libraries are installed
python3 -c "import flask" 2>/dev/null
FLASK_CHECK=$?

if [ $FLASK_CHECK -ne 0 ]; then
    echo -e "${YELLOW}📦 تثبيت المكتبات (قد يستغرق 3-10 دقائق)...${NC}"
    echo ""
    
    echo -e "${CYAN}  ⏳ Flask و Werkzeug...${NC}"
    pip install Flask==3.0.0 Werkzeug==3.0.1 -q
    echo -e "${GREEN}  ✅ Flask${NC}"
    
    echo -e "${CYAN}  ⏳ yt-dlp (محسن)...${NC}"
    pip install --upgrade yt-dlp -q
    echo -e "${GREEN}  ✅ yt-dlp${NC}"
    
    echo -e "${CYAN}  ⏳ Whisper (قد يستغرق وقتاً)...${NC}"
    pip install openai-whisper -q 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}  ⚠️ Whisper اختياري - يمكن المتابعة بدونه${NC}"
    else
        echo -e "${GREEN}  ✅ Whisper${NC}"
    fi
    
    echo -e "${CYAN}  ⏳ مكتبات معالجة الفيديو...${NC}"
    pip install moviepy pydub ffmpeg-python -q 2>/dev/null
    echo -e "${GREEN}  ✅ معالجة الفيديو${NC}"
    
    echo -e "${CYAN}  ⏳ الترجمة والترجمات...${NC}"
    pip install deep-translator pysrt -q
    echo -e "${GREEN}  ✅ الترجمة${NC}"
    
    echo -e "${CYAN}  ⏳ مكتبات إضافية...${NC}"
    pip install requests beautifulsoup4 lxml tqdm -q
    echo -e "${GREEN}  ✅ المكتبات الإضافية${NC}"
else
    echo -e "${GREEN}✅ المكتبات مثبتة مسبقاً${NC}"
    
    echo -e "${YELLOW}⏳ التحقق من التحديثات...${NC}"
    pip install --upgrade yt-dlp -q
    echo -e "${GREEN}✅ تم التحديث${NC}"
fi
echo ""

# Check Files
echo -e "${BLUE}[5/5] التحقق من الملفات...${NC}"
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ app.py غير موجود!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ الملف الرئيسي موجود${NC}"

# Create directories
mkdir -p templates downloads uploads outputs subtitles static
echo -e "${GREEN}✅ المجلدات جاهزة${NC}"

# Check HTML files
if [ ! -f "templates/index.html" ]; then
    echo -e "${YELLOW}⚠️ templates/index.html غير موجود${NC}"
    echo "يتم إنشاؤه تلقائياً عند التشغيل..."
fi
if [ ! -f "templates/subtitle_editor.html" ]; then
    echo -e "${YELLOW}⚠️ templates/subtitle_editor.html غير موجود${NC}"
    echo "يتم إنشاؤه تلقائياً عند التشغيل..."
fi
echo ""

# Check ffmpeg
echo "فحص الأدوات الإضافية..."
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✅ ffmpeg متوفر - ممتاز لدمج الترجمة!${NC}"
else
    echo -e "${YELLOW}⚠️ ffmpeg غير متوفر${NC}"
    echo ""
    echo "لتثبيت ffmpeg:"
    echo "  • Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  • macOS: brew install ffmpeg"
    echo "  • Fedora: sudo dnf install ffmpeg"
    echo ""
    echo "يمكن المتابعة بدونه لكن دمج الترجمة لن يعمل"
fi
echo ""

# Final message
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                           ║${NC}"
echo -e "${GREEN}║                    ✅ كل شيء جاهز! جاري التشغيل...                       ║${NC}"
echo -e "${GREEN}║                                                                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🎬 التطبيق المتكامل المحسن v4.0${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}✅ البيئة الافتراضية مفعّلة${NC}"
echo -e "${GREEN}✅ جميع المكتبات الأساسية مثبتة${NC}"
echo ""
echo -e "${YELLOW}🌐 افتح المتصفح على:${NC}"
echo ""
echo -e "${PURPLE}   👉 http://localhost:5000${NC} (الصفحة الرئيسية)"
echo -e "${PURPLE}   👉 http://localhost:5000/subtitle-editor${NC} (محرر الترجمة)"
echo ""

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || ipconfig getifaddr en0 2>/dev/null || echo "YOUR_IP")
if [ ! -z "$LOCAL_IP" ] && [ "$LOCAL_IP" != "YOUR_IP" ]; then
    echo "أو من جهاز آخر في نفس الشبكة:"
    echo ""
    echo -e "${PURPLE}   👉 http://${LOCAL_IP}:5000${NC}"
    echo ""
fi

echo -e "${YELLOW}💡 المميزات الجديدة:${NC}"
echo "  ✨ دعم محسن لـ TikTok بطرق متعددة"
echo "  🎯 تحكم كامل في جودة التحميل"
echo "  🎨 محرر ترجمة احترافي مع معاينة حية"
echo "  🎬 دمج الترجمة بجودات متعددة (أصلي/عالي/متوسط/منخفض)"
echo "  🎨 تخصيص كامل للترجمة (الخط/الحجم/اللون/الخلفية/الموضع)"
echo "  📱 متوافق مع Mac و Windows"
echo ""
echo -e "${YELLOW}🛑 لإيقاف الخادم: CTRL+C${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Open browser automatically
sleep 2
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000 2>/dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:5000 2>/dev/null &
fi

# Run the app
echo -e "${CYAN}🔥 بدء التشغيل...${NC}"
echo ""
python3 app.py
