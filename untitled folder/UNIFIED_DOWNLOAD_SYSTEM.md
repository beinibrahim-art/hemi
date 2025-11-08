# نظام التحميل الموحد - Unified Download System

## نظرة عامة

تم توحيد جميع عمليات التحميل في نظام مركزي واحد (`UnifiedDownloadManager`) يوفر:

- **دالتان أساسيتان**: `start_download()` و `execute_download()`
- **دالة تتبع موحدة**: `get_progress()`
- **دعم جميع أنواع الوسائط**: فيديو، صوت، تفريغ نصي
- **تنسيق استجابة موحد** في جميع أنواع التحميل
- **قابل للتوسع** بسهولة (إضافة جودة أو نوع ملف جديد)

## الاستخدام الأساسي

### 1. بدء التحميل

```python
from app import unified_downloader

# تحميل فيديو
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=...",
    quality="1080p",  # auto, best, 4k, 2160p, 1440p, 1080p, 720p, 480p, audio
    media_type=unified_downloader.MEDIA_TYPE_VIDEO
)

# تحميل صوت فقط
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=...",
    quality="audio",
    media_type=unified_downloader.MEDIA_TYPE_AUDIO
)

# تحميل + تفريغ نصي
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=...",
    quality="720p",
    media_type=unified_downloader.MEDIA_TYPE_TRANSCRIBE,
    options={
        'language': 'ar',  # أو 'auto' للكشف التلقائي
        'model_size': 'base'  # tiny, base, small, medium, large
    }
)
```

### 2. تتبع التقدم

```python
download_id = result['download_id']

# الحصول على حالة التقدم
progress = unified_downloader.get_progress(download_id)

# progress يحتوي على:
# {
#     'success': bool,
#     'status': 'starting' | 'downloading' | 'completed' | 'error',
#     'percent': '0%' | '50%' | '100%',
#     'message': str,
#     'download_id': str,
#     'url': str,
#     'quality': str,
#     'media_type': str,
#     'file': str | dict | None,
#     'error': str | None,
#     'started_at': str,
#     'completed_at': str | None,
#     'failed_at': str | None,
#     'method': str
# }
```

## API Endpoints الموحدة

### POST `/api/media/download`

بدء التحميل

**Request:**
```json
{
    "url": "https://youtube.com/watch?v=...",
    "quality": "1080p",
    "media_type": "video",  // "video" | "audio" | "transcribe"
    "options": {
        "language": "ar",
        "model_size": "base"
    }
}
```

**Response:**
```json
{
    "success": true,
    "download_id": "abc123...",
    "message": "تم بدء التحميل",
    "status": "started"
}
```

### GET `/api/media/progress/<download_id>`

الحصول على حالة التقدم

**Response:**
```json
{
    "success": true,
    "status": "completed",
    "percent": "100%",
    "message": "تم التحميل بنجاح!",
    "download_id": "abc123...",
    "url": "https://youtube.com/watch?v=...",
    "quality": "1080p",
    "media_type": "video",
    "file": "/path/to/file.mp4",
    "started_at": "2025-01-01T12:00:00",
    "completed_at": "2025-01-01T12:05:00"
}
```

## الجودات المدعومة

النظام يدعم الجودات التالية (قابلة للتوسع):

- `auto` - أفضل جودة متاحة (افتراضي)
- `best` - أفضل جودة
- `4k` / `2160p` - 4K Ultra HD
- `1440p` - QHD
- `1080p` - Full HD
- `720p` - HD
- `480p` - SD
- `360p` - منخفضة
- `audio` - صوت فقط

### إضافة جودة جديدة

```python
unified_downloader.add_quality_preset(
    quality_id="8k",
    format_command="bestvideo[height<=4320]+bestaudio/best[height<=4320]"
)
```

## أنواع الوسائط

### MEDIA_TYPE_VIDEO
تحميل فيديو عادي

### MEDIA_TYPE_AUDIO
تحميل صوت فقط (MP3)

### MEDIA_TYPE_TRANSCRIBE
تحميل فيديو + تفريغ نصي تلقائي (تحويل الصوت إلى نص)

## التوسع

النظام قابل للتوسع بسهولة:

1. **إضافة جودة جديدة**: استخدم `add_quality_preset()`
2. **إضافة نوع وسائط جديد**: أضف ثابت جديد في `MEDIA_TYPE_*` وعدّل `execute_download()`
3. **تخصيص التنسيق**: استخدم `format_command` مباشرة بدلاً من `quality`

## المزايا

✅ **نقطة مركزية واحدة** - جميع عمليات التحميل من مكان واحد  
✅ **تنسيق موحد** - نفس البنية للاستجابات  
✅ **غير متزامن** - التحميل في threads منفصلة  
✅ **قابل للتوسع** - إضافة جودة أو نوع جديد بسهولة  
✅ **تتبع التقدم** - دالة واحدة موحدة  
✅ **دعم متعدد** - فيديو، صوت، تفريغ نصي  

## أمثلة الاستخدام

### مثال 1: تحميل فيديو بجودة 1080p

```python
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=abc123",
    quality="1080p"
)

if result['success']:
    download_id = result['download_id']
    
    # تتبع التقدم
    while True:
        progress = unified_downloader.get_progress(download_id)
        print(f"التقدم: {progress['percent']}")
        
        if progress['status'] == 'completed':
            print(f"تم التحميل: {progress['file']}")
            break
        elif progress['status'] == 'error':
            print(f"خطأ: {progress['error']}")
            break
        
        import time
        time.sleep(1)
```

### مثال 2: تحميل صوت

```python
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=abc123",
    quality="audio",
    media_type=unified_downloader.MEDIA_TYPE_AUDIO
)
```

### مثال 3: تحميل + تفريغ نصي

```python
result = unified_downloader.start_download(
    url="https://youtube.com/watch?v=abc123",
    quality="720p",
    media_type=unified_downloader.MEDIA_TYPE_TRANSCRIBE,
    options={
        'language': 'ar',
        'model_size': 'base'
    }
)

# بعد اكتمال التحميل
progress = unified_downloader.get_progress(result['download_id'])
if progress['status'] == 'completed':
    file_info = progress['file']
    print(f"الفيديو: {file_info['video']}")
    print(f"الترجمة: {file_info['transcript']}")
    print(f"النص: {file_info['text']}")
```

## ملاحظات

- النظام يستخدم `SmartMediaDownloader` كـ backend للتحميل الفعلي
- جميع عمليات التحميل غير متزامنة (asynchronous)
- التقدم يتم تتبعه تلقائياً عبر `download_progress` و `progress_tracker`
- النظام متوافق مع الكود القديم (`downloader` ما زال متاحاً)
