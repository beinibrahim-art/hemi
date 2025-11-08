#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت للتحقق من المكتبات المطلوبة
"""

import sys

# قائمة المكتبات المطلوبة
REQUIRED_PACKAGES = {
    # Core Web Framework
    'flask': 'Flask',
    'werkzeug': 'Werkzeug',
    
    # Media Download
    'yt_dlp': 'yt-dlp',
    
    # Speech Recognition
    'whisper': 'openai-whisper',
    'faster_whisper': 'faster-whisper',
    
    # Translation
    'deep_translator': 'deep-translator',
    
    # Audio/Video Processing
    'moviepy': 'moviepy',
    'pydub': 'pydub',
    
    # Additional
    'numpy': 'numpy',
    'torch': 'torch',
    'torchaudio': 'torchaudio',
    'PIL': 'Pillow',
    'requests': 'requests',
    'bs4': 'beautifulsoup4',
    'pysrt': 'pysrt',
    'tqdm': 'tqdm',
}

# قائمة المكتبات الاختيارية
OPTIONAL_PACKAGES = {
    'pyannote': 'pyannote.audio',
    'googletrans': 'googletrans',
    'webvtt': 'webvtt-py',
    'asstosrt': 'asstosrt',
    'ffmpeg': 'ffmpeg-python',
    'SpeechRecognition': 'SpeechRecognition',
    'jsonschema': 'jsonschema',
}

def check_package(package_name, display_name):
    """التحقق من وجود مكتبة"""
    try:
        __import__(package_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("فحص المكتبات المطلوبة للتطبيق")
    print("=" * 70)
    print()
    
    missing_required = []
    missing_optional = []
    
    # فحص المكتبات المطلوبة
    print("📦 المكتبات المطلوبة:")
    print("-" * 70)
    for package_name, display_name in REQUIRED_PACKAGES.items():
        is_installed, error = check_package(package_name, display_name)
        if is_installed:
            print(f"✅ {display_name:30s} - متوفر")
        else:
            print(f"❌ {display_name:30s} - غير متوفر")
            missing_required.append(display_name)
    print()
    
    # فحص المكتبات الاختيارية
    print("📦 المكتبات الاختيارية:")
    print("-" * 70)
    for package_name, display_name in OPTIONAL_PACKAGES.items():
        is_installed, error = check_package(package_name, display_name)
        if is_installed:
            print(f"✅ {display_name:30s} - متوفر")
        else:
            print(f"⚠️  {display_name:30s} - غير متوفر (اختياري)")
            missing_optional.append(display_name)
    print()
    
    # النتيجة النهائية
    print("=" * 70)
    if missing_required:
        print("❌ بعض المكتبات المطلوبة غير متوفرة!")
        print("\nالمكتبات المفقودة:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nلتثبيت المكتبات المفقودة:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("✅ جميع المكتبات المطلوبة متوفرة!")
        if missing_optional:
            print(f"\n⚠️  {len(missing_optional)} مكتبة اختيارية غير متوفرة (لا يؤثر على التشغيل)")
        return 0

if __name__ == '__main__':
    sys.exit(main())
