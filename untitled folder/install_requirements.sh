#!/bin/bash
# سكريبت تثبيت محسّن لـ macOS مع Python 3.13

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                           ║${NC}"
echo -e "${CYAN}║        📦 تثبيت المكتبات في البيئة الافتراضية (macOS)                    ║${NC}"
echo -e "${CYAN}║                                                                           ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# التحقق من Python
echo -e "${BLUE}[1/5] فحص Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 غير مثبت!${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Python جاهز: ${PYTHON_VERSION}${NC}"

# التحقق من إصدار Python
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo -e "${YELLOW}⚠️  Python 3.13+ - سيتم استخدام إصدارات متوافقة${NC}"
fi
echo ""

# إنشاء البيئة الافتراضية
echo -e "${BLUE}[2/5] إعداد البيئة الافتراضية...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 إنشاء البيئة الافتراضية...${NC}"
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ تم إنشاء البيئة${NC}"
    else
        echo -e "${RED}❌ فشل إنشاء البيئة${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ البيئة موجودة${NC}"
fi
echo ""

# تفعيل البيئة
echo -e "${BLUE}[3/5] تفعيل البيئة الافتراضية...${NC}"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ تم التفعيل${NC}"
else
    echo -e "${RED}❌ فشل التفعيل${NC}"
    exit 1
fi

# تحديث pip
echo -e "${YELLOW}⏳ تحديث pip...${NC}"
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}✅ pip محدث${NC}"
echo ""

# تثبيت المكتبات الأساسية أولاً
echo -e "${BLUE}[4/5] تثبيت المكتبات...${NC}"
echo ""

# Core Web Framework
echo -e "${CYAN}[1/10] Flask و Werkzeug...${NC}"
pip install Flask==3.0.0 Werkzeug==3.0.1 -q
echo -e "${GREEN}✅ Flask${NC}"

# Media Download
echo -e "${CYAN}[2/10] yt-dlp...${NC}"
pip install --upgrade "yt-dlp>=2024.1.0" -q
echo -e "${GREEN}✅ yt-dlp${NC}"

# Basic libraries first
echo -e "${CYAN}[3/10] المكتبات الأساسية...${NC}"
pip install "requests==2.31.0" "beautifulsoup4==4.12.2" "tqdm==4.66.1" "jsonschema==4.19.0" -q
# محاولة تثبيت lxml (قد يفشل على macOS - اختياري)
pip install "lxml==4.9.3" -q 2>/dev/null || echo -e "${YELLOW}  ⚠️ lxml - اختياري (يمكن استخدام html.parser)${NC}"
echo -e "${GREEN}✅ المكتبات الأساسية${NC}"

# NumPy and Pillow
echo -e "${CYAN}[4/10] NumPy و Pillow...${NC}"
pip install "numpy>=1.24.3" "Pillow>=10.1.0" -q
echo -e "${GREEN}✅ NumPy و Pillow${NC}"

# Audio/Video Processing
echo -e "${CYAN}[5/10] معالجة الفيديو...${NC}"
pip install "pydub==0.25.1" "ffmpeg-python==0.2.0" -q
pip install "moviepy==1.0.3" -q 2>/dev/null || pip install moviepy -q
echo -e "${GREEN}✅ معالجة الفيديو${NC}"

# Translation
echo -e "${CYAN}[6/10] الترجمة...${NC}"
pip install "deep-translator==1.11.4" "googletrans==4.0.0rc1" -q
echo -e "${GREEN}✅ الترجمة${NC}"

# Subtitle Processing
echo -e "${CYAN}[7/10] معالجة الترجمة...${NC}"
pip install "pysrt==1.1.2" "asstosrt==0.1.6" "webvtt-py==0.4.6" -q
echo -e "${GREEN}✅ معالجة الترجمة${NC}"

# PyTorch (CPU version for macOS - smaller)
echo -e "${CYAN}[8/10] PyTorch (قد يستغرق وقتاً)...${NC}"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || pip install "torch>=2.1.0" "torchaudio>=2.1.0" -q
echo -e "${GREEN}✅ PyTorch${NC}"

# Whisper - محاولة إصدارات متعددة
echo -e "${CYAN}[9/10] Whisper (قد يستغرق وقتاً)...${NC}"
# محاولة faster-whisper أولاً (أفضل وأسرع)
pip install "faster-whisper>=1.0.0" -q 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Faster Whisper${NC}"
else
    echo -e "${YELLOW}⚠️  Faster Whisper - اختياري${NC}"
fi

# محاولة openai-whisper من GitHub مباشرة (أحدث إصدار)
echo -e "${CYAN}      محاولة تثبيت openai-whisper...${NC}"
pip install git+https://github.com/openai/whisper.git -q 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Whisper (من GitHub)${NC}"
else
    # محاولة الإصدار القديم
    pip install "openai-whisper>=20231117" -q 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Whisper${NC}"
    else
        echo -e "${YELLOW}⚠️  Whisper - يمكن المتابعة مع Faster Whisper فقط${NC}"
    fi
fi

# SpeechRecognition
echo -e "${CYAN}[10/10] SpeechRecognition...${NC}"
pip install "SpeechRecognition==3.10.0" -q 2>/dev/null
echo -e "${GREEN}✅ SpeechRecognition${NC}"

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# التحقق من التثبيت
echo -e "${BLUE}[5/5] التحقق من المكتبات...${NC}"
python3 check_requirements.py
CHECK_RESULT=$?

echo ""
if [ $CHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                           ║${NC}"
    echo -e "${GREEN}║                    ✅ تم تثبيت جميع المكتبات بنجاح!                      ║${NC}"
    echo -e "${GREEN}║                                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}⚠️  بعض المكتبات لم يتم تثبيتها${NC}"
    echo ""
    echo "يمكنك المحاولة مرة أخرى أو تثبيت المكتبات المفقودة يدوياً"
fi

echo ""
echo -e "${CYAN}💡 ملاحظات:${NC}"
echo "  • البيئة الافتراضية مفعّلة الآن"
echo "  • لتشغيل التطبيق: python3 app.py"
echo "  • لإيقاف البيئة: deactivate"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
