#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
التطبيق المتكامل للترجمة والتحميل v5.0
- تحميل الفيديو من جميع المنصات
- الترجمة الفورية (تحميل + تحويل صوت + ترجمة + دمج)
- دمج الترجمة مع الفيديو
- تحويل الصوت/الفيديو إلى نص
- ترجمة النص
"""

import os
import sys
import json
import logging
import subprocess
import traceback
import re
import secrets
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import yt_dlp

# محاولة استيراد المكتبات الاختيارية
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper غير متوفر - لن يعمل تحويل الكلام إلى نص")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("✅ Faster Whisper متوفر - سيتم استخدامه (أسرع بـ 4-5x)")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("⚠️ Faster Whisper غير متوفر - سيتم استخدام Whisper العادي")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ deep-translator غير متوفر - لن تعمل الترجمة")

# إعدادات التطبيق
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SUBTITLE_FOLDER'] = 'subtitles'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# تقليل حجم session cookie
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 دقيقة

# إعدادات السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# إنشاء المجلدات المطلوبة
for folder in ['uploads', 'downloads', 'outputs', 'subtitles', 'templates', 'static']:
    Path(folder).mkdir(exist_ok=True)


def transcribe_audio(audio_file: str, model_size: str = 'base', language: str = 'auto', use_faster: bool = True):
    """
    تحويل الصوت إلى نص باستخدام Faster Whisper أو Whisper العادي
    
    Args:
        audio_file: مسار ملف الصوت
        model_size: حجم النموذج (tiny, base, small, medium, large)
        language: اللغة (auto للكشف التلقائي)
        use_faster: استخدام Faster Whisper إذا كان متاحاً
    
    Returns:
        dict: {'text': str, 'language': str, 'segments': list}
    """
    # محاولة استخدام Faster Whisper أولاً (أسرع بـ 4-5x)
    if use_faster and FASTER_WHISPER_AVAILABLE:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            logger.info(f"استخدام Faster Whisper مع device: {device}, compute_type: {compute_type}")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            # تحويل اللغة
            language_code = None if language == 'auto' else language
            
            # تحويل الصوت إلى نص
            segments, info = model.transcribe(
                audio_file,
                language=language_code,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,  # إزالة الضوضاء
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # تجميع النص والـ segments
            full_text = ""
            segments_list = []
            
            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:
                    full_text += segment_text + " "
                    segments_list.append({
                        'id': len(segments_list) + 1,
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment_text,
                        'words': [
                            {
                                'word': word.word,
                                'start': word.start,
                                'end': word.end
                            }
                            for word in segment.words
                        ] if hasattr(segment, 'words') else []
                    })
            
            detected_language = info.language if hasattr(info, 'language') else language
            
            logger.info(f"تم التحويل بنجاح باستخدام Faster Whisper - اللغة المكتشفة: {detected_language}")
            
            return {
                'text': full_text.strip(),
                'language': detected_language,
                'segments': segments_list
            }
        except Exception as e:
            logger.warning(f"فشل استخدام Faster Whisper: {e} - سيتم استخدام Whisper العادي")
            # الاستمرار إلى Whisper العادي
    
    # استخدام Whisper العادي كـ fallback
    if WHISPER_AVAILABLE:
        try:
            logger.info("استخدام Whisper العادي")
            model = whisper.load_model(model_size)
            
            import torch
            use_fp16 = torch.cuda.is_available()
            
            options = {
                'language': None if language == 'auto' else language,
                'task': 'transcribe',
                'fp16': use_fp16,
                'beam_size': 5,
                'best_of': 5,
                'temperature': 0.0,
                'word_timestamps': True
            }
            
            result = model.transcribe(audio_file, **options)
            
            # معالجة segments لضمان وجود word timestamps
            processed_segments = []
            for seg in result.get('segments', []):
                processed_seg = {
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '').strip(),
                    'words': seg.get('words', [])  # word timestamps من Whisper
                }
                processed_segments.append(processed_seg)
            
            return {
                'text': result['text'],
                'language': result.get('language', language),
                'segments': processed_segments
            }
        except Exception as e:
            logger.error(f"فشل استخدام Whisper: {e}")
            raise Exception(f"فشل تحويل الصوت إلى نص: {str(e)}")
    else:
        raise Exception("لا توجد مكتبة متاحة لتحويل الصوت إلى نص")


# Store progress for downloads
download_progress = {}

class SmartMediaDownloader:
    """محمل الوسائط الذكي مع استراتيجيات متعددة للتحميل"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = app.config.get('DOWNLOAD_FOLDER', 'downloads')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.check_dependencies()
    
    def check_dependencies(self):
        """التحقق من الأدوات المطلوبة"""
        self.available_tools = {
            'yt-dlp': self._check_command('yt-dlp'),
            'ffmpeg': self._check_command('ffmpeg')
        }
        
        if not self.available_tools['yt-dlp']:
            self._install_yt_dlp()
    
    def _check_command(self, cmd: str) -> bool:
        try:
            subprocess.run([cmd, '--version'], 
                         capture_output=True, 
                         check=True, 
                         timeout=5)
            return True
        except:
            return False
    
    def _install_yt_dlp(self):
        try:
            subprocess.run(['pip', 'install', '-U', 'yt-dlp', '-q'],
                         check=True)
            self.available_tools['yt-dlp'] = True
        except:
            pass
    
    def detect_platform(self, url: str) -> str:
        """الكشف عن المنصة من الرابط"""
        url_lower = url.lower()
        
        if any(x in url_lower for x in ['twitter.com', 'x.com', 't.co']):
            return 'twitter'
        elif any(x in url_lower for x in ['youtube.com', 'youtu.be']):
            return 'youtube'
        elif 'instagram.com' in url_lower:
            return 'instagram'
        elif 'tiktok.com' in url_lower:
            return 'tiktok'
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            return 'facebook'
        else:
            return 'unknown'
    
    def get_available_formats(self, url: str) -> dict:
        """الحصول على جميع التنسيقات المتاحة للفيديو"""
        try:
            # استخدام -F فقط للحصول على قائمة التنسيقات (أبسط وأكثر موثوقية)
            cmd = ['yt-dlp', '-F', '--no-warnings', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return {'success': False, 'error': 'Could not fetch formats'}
            
            formats = self._parse_formats(result.stdout)
            info = self._get_video_info(url)
            
            return {
                'success': True,
                'formats': formats,
                'info': info,
                'platform': self.detect_platform(url)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_formats(self, output: str) -> dict:
        """تحليل مخرجات yt-dlp format - نسخة بسيطة وموثوقة"""
        formats = {
            'video_audio': [],
            'video_only': [],
            'audio_only': [],
            'best': None
        }
        
        lines = output.split('\n')
        
        for line in lines:
            if not line.strip() or 'format code' in line.lower():
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            format_id = parts[0]
            quality_match = re.search(r'(\d+)x(\d+)', line)
            resolution = None
            
            if quality_match:
                width, height = quality_match.groups()
                resolution = f"{height}p"
            
            format_info = {
                'id': format_id,
                'resolution': resolution,
                'note': '',
                'filesize': self._extract_filesize(line)
            }
            
            line_lower = line.lower()
            
            if 'audio only' in line_lower or 'm4a' in line_lower or 'mp3' in line_lower:
                bitrate = re.search(r'(\d+)k', line)
                if bitrate:
                    format_info['bitrate'] = bitrate.group(1) + 'kbps'
                format_info['note'] = 'Audio Only'
                formats['audio_only'].append(format_info)
                
            elif 'video only' in line_lower:
                if resolution:
                    format_info['note'] = f'{resolution} (No Audio)'
                    formats['video_only'].append(format_info)
                    
            elif resolution:
                format_info['note'] = f'{resolution}'
                formats['video_audio'].append(format_info)
        
        formats['presets'] = self._create_smart_presets(formats)
        
        return formats
    
    def _create_smart_presets(self, formats: dict) -> list:
        """إنشاء presets ذكية للتحميل - نسخة بسيطة وموثوقة"""
        presets = []
        
        presets.append({
            'id': 'best',
            'name': 'أفضل جودة',
            'description': 'أعلى جودة متاحة (تلقائي)',
            'icon': 'crown',
            'command': 'bestvideo+bestaudio/best'
        })
        
        # التحقق من وجود 4K (2160p أو 4320p)
        has_4k = any(f.get('resolution') in ['2160p', '4320p'] 
                     for f in formats['video_audio'] + formats['video_only'])
        if has_4k:
            presets.append({
                'id': '4k',
                'name': '4K Ultra HD',
                'description': '2160p - جودة فائقة',
                'icon': 'sparkles',
                'command': 'bestvideo[height<=2160]+bestaudio/best[height<=2160]'
            })
        
        # التحقق من وجود 1440p
        has_1440p = any(f.get('resolution') == '1440p' 
                        for f in formats['video_audio'] + formats['video_only'])
        if has_1440p:
            presets.append({
                'id': '1440p',
                'name': '1440p QHD',
                'description': 'جودة عالية جداً',
                'icon': 'star',
                'command': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]'
            })
        
        # التحقق من وجود 1080p
        has_1080p = any(f.get('resolution') == '1080p' 
                        for f in formats['video_audio'] + formats['video_only'])
        if has_1080p:
            presets.append({
                'id': '1080p',
                'name': '1080p Full HD',
                'description': 'جودة ممتازة',
                'icon': 'video',
                'command': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
            })
        
        # التحقق من وجود 720p
        has_720p = any(f.get('resolution') == '720p' 
                       for f in formats['video_audio'] + formats['video_only'])
        if has_720p:
            presets.append({
                'id': '720p',
                'name': '720p HD',
                'description': 'جودة جيدة - حجم متوازن',
                'icon': 'film',
                'command': 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            })
        
        # التحقق من وجود 480p
        has_480p = any(f.get('resolution') == '480p' 
                       for f in formats['video_audio'] + formats['video_only'])
        if has_480p:
            presets.append({
                'id': '480p',
                'name': '480p SD',
                'description': 'جودة متوسطة - حجم صغير',
                'icon': 'smartphone',
                'command': 'bestvideo[height<=480]+bestaudio/best[height<=480]'
            })
        
        if formats['audio_only']:
            presets.append({
                'id': 'audio',
                'name': 'صوت فقط',
                'description': 'MP3 بأفضل جودة',
                'icon': 'music',
                'command': 'audio'
            })
        
        return presets
    
    def _extract_filesize(self, line: str) -> str:
        size_match = re.search(r'(\d+\.?\d*\s*[KMG]iB)', line)
        if size_match:
            return size_match.group(1)
        return 'Unknown'
    
    def _get_video_info(self, url: str) -> dict:
        """الحصول على معلومات الفيديو"""
        try:
            cmd = ['yt-dlp', '--dump-json', '--no-warnings', '--skip-download', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'description': info.get('description', '')[:200]
                }
        except:
            pass
        
        return {
            'title': 'Unknown',
            'uploader': 'Unknown',
            'duration': 0,
            'thumbnail': '',
            'description': ''
        }
    
    def download_with_format(self, url: str, format_command: str, 
                            download_id: str, is_audio: bool = False):
        """التحميل مع تنسيق محدد باستخدام استراتيجيات متعددة"""
        try:
            download_progress[download_id] = {
                'status': 'starting',
                'percent': '0%',
                'method': 'Attempting download...'
            }
            
            success = self._download_strategy_1(url, format_command, download_id, is_audio)
            
            if not success:
                download_progress[download_id]['method'] = 'Trying with cookies...'
                success = self._download_strategy_2(url, format_command, download_id, is_audio)
            
            if not success:
                download_progress[download_id]['method'] = 'Trying compatible format...'
                success = self._download_strategy_3(url, download_id, is_audio)
            
            if not success:
                download_progress[download_id]['method'] = 'Last attempt...'
                success = self._download_strategy_4(url, download_id, is_audio)
            
            if success:
                download_progress[download_id] = {
                    'status': 'completed',
                    'percent': '100%',
                    'message': 'تم التحميل بنجاح!'
                }
                return {'success': True}
            else:
                download_progress[download_id] = {
                    'status': 'error',
                    'message': 'فشل التحميل بجميع الطرق'
                }
                return {'success': False, 'error': 'All strategies failed'}
                
        except Exception as e:
            download_progress[download_id] = {
                'status': 'error',
                'message': str(e)
            }
            return {'success': False, 'error': str(e)}
    
    def _download_strategy_1(self, url: str, format_cmd: str, download_id: str, is_audio: bool) -> bool:
        try:
            output_template = str(self.output_dir / '%(title)s.%(ext)s')
            
            cmd = ['yt-dlp']
            
            if is_audio or format_cmd == 'audio':
                cmd.extend(['-x', '--audio-format', 'mp3', '--audio-quality', '0'])
            else:
                cmd.extend(['-f', format_cmd, '--merge-output-format', 'mp4'])
            
            cmd.extend([
                '-o', output_template,
                '--no-warnings',
                '--newline',
                url
            ])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            return self._monitor_download(process, download_id)
            
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")
            return False
    
    def _download_strategy_2(self, url: str, format_cmd: str, download_id: str, is_audio: bool) -> bool:
        try:
            output_template = str(self.output_dir / '%(title)s.%(ext)s')
            
            cmd = ['yt-dlp', '--cookies-from-browser', 'chrome']
            
            if is_audio or format_cmd == 'audio':
                cmd.extend(['-x', '--audio-format', 'mp3'])
            else:
                cmd.extend(['-f', format_cmd])
            
            cmd.extend(['-o', output_template, '--no-warnings', url])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            return self._monitor_download(process, download_id)
            
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")
            return False
    
    def _download_strategy_3(self, url: str, download_id: str, is_audio: bool) -> bool:
        try:
            output_template = str(self.output_dir / '%(title)s.%(ext)s')
            
            cmd = ['yt-dlp']
            
            if is_audio:
                cmd.extend(['-x', '--audio-format', 'mp3'])
            else:
                cmd.extend(['-f', 'best[ext=mp4]/best'])
            
            cmd.extend(['-o', output_template, '--no-warnings', url])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            return self._monitor_download(process, download_id)
            
        except Exception as e:
            logger.error(f"Strategy 3 failed: {e}")
            return False
    
    def _download_strategy_4(self, url: str, download_id: str, is_audio: bool) -> bool:
        try:
            output_template = str(self.output_dir / '%(title)s.%(ext)s')
            
            cmd = ['yt-dlp', '-o', output_template, '--no-warnings', url]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            return self._monitor_download(process, download_id)
            
        except Exception as e:
            logger.error(f"Strategy 4 failed: {e}")
            return False
    
    def _monitor_download(self, process, download_id: str) -> bool:
        try:
            for line in process.stdout:
                if '[download]' in line and '%' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part:
                            download_progress[download_id] = {
                                'status': 'downloading',
                                'percent': part
                            }
                            break
            
            process.wait()
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Monitor failed: {e}")
            return False
    
    def download(self, url: str, quality: str = 'best') -> Dict:
        """تحميل الفيديو - دالة متوافقة مع الكود القديم"""
        result = {
            'success': False,
            'message': '',
            'file': None,
            'info': {}
        }
        
        try:
            # تحويل quality إلى format command
            format_command = quality
            is_audio = (quality == 'audio')
            
            if quality == 'best':
                format_command = 'bestvideo+bestaudio/best'
            elif quality == '720p' or quality == 'medium':
                format_command = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            elif quality == '480p' or quality == 'low':
                format_command = 'bestvideo[height<=480]+bestaudio/best[height<=480]'
            elif quality.startswith('1080p'):
                format_command = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
            elif quality.startswith('1440p'):
                format_command = 'bestvideo[height<=1440]+bestaudio/best[height<=1440]'
            elif quality.startswith('2160p') or quality == '4k':
                format_command = 'bestvideo[height<=2160]+bestaudio/best[height<=2160]'
            
            download_id = secrets.token_hex(8)
            
            # استخدام download_with_format في thread منفصل
            download_result = self.download_with_format(url, format_command, download_id, is_audio)
            
            if download_result.get('success'):
                # البحث عن الملف المحمّل
                info = self._get_video_info(url)
                title = info.get('title', 'video')
                
                # البحث عن الملف الأحدث في مجلد التحميل
                download_folder = self.output_dir
                if download_folder.exists():
                    video_files = []
                    for file in download_folder.iterdir():
                        if file.is_file() and file.suffix.lower() in ['.mp4', '.webm', '.mkv', '.mp3', '.m4a', '.mov', '.avi']:
                            video_files.append((file, file.stat().st_mtime))
                    
                    if video_files:
                        video_files.sort(key=lambda x: x[1], reverse=True)
                        latest_file = video_files[0][0]
                        
                        result['success'] = True
                        result['message'] = 'تم التحميل بنجاح'
                        result['file'] = str(latest_file)
                        result['info'] = {
                            'title': title,
                            'duration': info.get('duration', 0),
                            'platform': self.detect_platform(url),
                            'quality': quality
                        }
                        return result
            
            # إذا فشل التحميل
            result['message'] = download_result.get('error', 'فشل التحميل')
            
        except Exception as e:
            result['message'] = f'خطأ في التحميل: {str(e)}'
            logger.error(f"Download error: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def is_youtube_shorts(self, url: str) -> bool:
        """التحقق من إذا كان الرابط YouTube Shorts"""
        return '/shorts/' in url.lower() or 'youtube.com/shorts/' in url.lower()
    
    def get_ydl_opts(self, platform: str, quality: str = 'best', player_client: str = 'web') -> dict:
        """الحصول على إعدادات yt-dlp - للتوافق مع الكود القديم"""
        # هذه الدالة محتفظ بها للتوافق فقط
        # الكود الجديد يستخدم download_with_format مباشرة
        opts = {
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'concurrent_fragment_downloads': 16,
            'http_chunk_size': 10485760,
            'ignoreerrors': False,
            'no_check_certificate': True,
            'prefer_insecure': False,
            'extractor_args': {
                'youtube': {
                    'player_client': [player_client],
                }
            },
            'format_sort': ['res', 'ext:mp4:m4a', 'codec', 'size'],
        }
        
        if quality == 'best':
            opts['format'] = 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif quality == '720p' or quality == 'medium':
            opts['format'] = 'bestvideo[height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best'
        elif quality == '480p' or quality == 'low':
            opts['format'] = 'bestvideo[height<=480][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best'
        elif quality == 'audio':
            opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio'
            opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:
            opts['format'] = quality
        
        if platform == 'tiktok':
            opts['http_headers'] = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
                'Referer': 'https://www.tiktok.com/',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
            }
            opts['extractor_args'] = {
                'tiktok': {
                    'webpage_download': True,
                }
            }
            if quality != 'audio':
                opts['format'] = 'best[ext=mp4]/best'
        
        return opts


class UnifiedDownloadManager:
    """
    مدير التحميل الموحد - نقطة مركزية واحدة لجميع عمليات التحميل
    يدعم: فيديو، صوت، تفريغ نصي
    """
    
    # Quality presets mapping - قابل للتوسع بسهولة
    QUALITY_PRESETS = {
        'auto': 'bestvideo+bestaudio/best',
        'best': 'bestvideo+bestaudio/best',
        '4k': 'bestvideo[height<=2160]+bestaudio/best[height<=2160]',
        '2160p': 'bestvideo[height<=2160]+bestaudio/best[height<=2160]',
        '1440p': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
        '1080p': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        '360p': 'bestvideo[height<=360]+bestaudio/best[height<=360]',
        'audio': 'audio',
        'audio_best': 'bestaudio/bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio'
    }
    
    # Media types
    MEDIA_TYPE_VIDEO = 'video'
    MEDIA_TYPE_AUDIO = 'audio'
    MEDIA_TYPE_TRANSCRIBE = 'transcribe'  # تحميل + تفريغ نصي
    
    def __init__(self, output_dir: str = None):
        """تهيئة المدير الموحد"""
        if output_dir is None:
            output_dir = app.config.get('DOWNLOAD_FOLDER', 'downloads')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # استخدام SmartMediaDownloader كـ backend
        self.downloader = SmartMediaDownloader(output_dir)
        
        # تتبع التقدم - بنية موحدة
        self.progress_tracker = {}
    
    def start_download(self, url: str, quality: str = 'auto', 
                      media_type: str = MEDIA_TYPE_VIDEO, 
                      options: dict = None) -> dict:
        """
        بدء عملية التحميل - الدالة الأساسية الأولى
        
        Args:
            url: رابط الوسائط
            quality: الجودة (auto, best, 4k, 1080p, 720p, 480p, audio, etc.)
            media_type: نوع الوسائط (video, audio, transcribe)
            options: خيارات إضافية (language, model_size, etc.)
        
        Returns:
            dict: {'success': bool, 'download_id': str, 'message': str}
        """
        if not url:
            return {
                'success': False,
                'error': 'No URL provided',
                'message': 'الرجاء إدخال رابط'
            }
        
        # توليد معرف فريد للتحميل
        download_id = secrets.token_hex(8)
        
        # تحويل quality إلى format command
        format_command = self._quality_to_format(quality, media_type)
        
        # تحديد نوع التحميل
        is_audio = (media_type == self.MEDIA_TYPE_AUDIO) or (quality == 'audio')
        
        # تهيئة تتبع التقدم
        self.progress_tracker[download_id] = {
            'status': 'starting',
            'percent': '0%',
            'message': 'بدء التحميل...',
            'url': url,
            'quality': quality,
            'media_type': media_type,
            'format_command': format_command,
            'options': options or {},
            'started_at': datetime.now().isoformat(),
            'file': None,
            'error': None
        }
        
        # بدء التحميل في thread منفصل (غير متزامن)
        thread = threading.Thread(
            target=self._execute_download_worker,
            args=(download_id, url, format_command, is_audio, media_type, options)
        )
        thread.daemon = True
        thread.start()
        
        return {
            'success': True,
            'download_id': download_id,
            'message': 'تم بدء التحميل',
            'status': 'started'
        }
    
    def _execute_download_worker(self, download_id: str, url: str, 
                                 format_command: str, is_audio: bool,
                                 media_type: str, options: dict):
        """
        عامل التحميل - يتم تنفيذه في thread منفصل
        يستدعي execute_download للتنفيذ الفعلي
        """
        try:
            result = self.execute_download(
                download_id=download_id,
                url=url,
                format_command=format_command,
                is_audio=is_audio,
                media_type=media_type,
                options=options or {}
            )
            
            # تحديث التقدم النهائي
            if result['success']:
                self.progress_tracker[download_id].update({
                    'status': 'completed',
                    'percent': '100%',
                    'message': 'تم التحميل بنجاح!',
                    'file': result.get('file'),
                    'completed_at': datetime.now().isoformat()
                })
            else:
                self.progress_tracker[download_id].update({
                    'status': 'error',
                    'message': result.get('message', 'فشل التحميل'),
                    'error': result.get('error', 'Unknown error'),
                    'failed_at': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Download worker error: {e}")
            logger.error(traceback.format_exc())
            self.progress_tracker[download_id].update({
                'status': 'error',
                'message': f'خطأ في التحميل: {str(e)}',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })
    
    def execute_download(self, download_id: str, url: str, 
                        format_command: str, is_audio: bool = False,
                        media_type: str = MEDIA_TYPE_VIDEO,
                        options: dict = None) -> dict:
        """
        تنفيذ التحميل الفعلي - الدالة الأساسية الثانية
        
        Args:
            download_id: معرف التحميل
            url: رابط الوسائط
            format_command: أمر التنسيق لـ yt-dlp
            is_audio: هل التحميل صوت فقط؟
            media_type: نوع الوسائط
            options: خيارات إضافية
        
        Returns:
            dict: {'success': bool, 'file': str, 'message': str, 'error': str}
        """
        options = options or {}
        
        try:
            # تحديث حالة البدء
            if download_id in self.progress_tracker:
                self.progress_tracker[download_id].update({
                    'status': 'downloading',
                    'message': 'جاري التحميل...'
                })
            
            # استخدام SmartMediaDownloader للتحميل الفعلي
            download_result = self.downloader.download_with_format(
                url=url,
                format_command=format_command,
                download_id=download_id,
                is_audio=is_audio
            )
            
            if not download_result.get('success'):
                return {
                    'success': False,
                    'message': download_result.get('error', 'فشل التحميل'),
                    'error': download_result.get('error', 'Unknown error')
                }
            
            # البحث عن الملف المحمّل
            downloaded_file = self._find_downloaded_file(url)
            
            if not downloaded_file:
                return {
                    'success': False,
                    'message': 'تم التحميل لكن الملف غير موجود',
                    'error': 'File not found'
                }
            
            # إذا كان النوع transcribe، قم بالتفريغ النصي
            if media_type == self.MEDIA_TYPE_TRANSCRIBE:
                transcribe_result = self._transcribe_downloaded_file(
                    downloaded_file, options
                )
                if not transcribe_result.get('success'):
                    return {
                        'success': False,
                        'message': f'تم التحميل لكن فشل التفريغ: {transcribe_result.get("error")}',
                        'error': transcribe_result.get('error'),
                        'file': downloaded_file  # الملف موجود لكن التفريغ فشل
                    }
                # إضافة معلومات التفريغ
                downloaded_file = {
                    'video': downloaded_file,
                    'transcript': transcribe_result.get('transcript_file'),
                    'text': transcribe_result.get('text')
                }
            
            return {
                'success': True,
                'file': downloaded_file,
                'message': 'تم التحميل بنجاح'
            }
            
        except Exception as e:
            logger.error(f"Execute download error: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f'خطأ في التحميل: {str(e)}',
                'error': str(e)
            }
    
    def get_progress(self, download_id: str) -> dict:
        """
        الحصول على حالة التقدم - دالة موحدة لتتبع التقدم
        
        Args:
            download_id: معرف التحميل
        
        Returns:
            dict: حالة التقدم الموحدة
        """
        if download_id not in self.progress_tracker:
            return {
                'status': 'unknown',
                'percent': '0%',
                'message': 'التحميل غير موجود',
                'error': 'Download ID not found'
            }
        
        progress = self.progress_tracker[download_id].copy()
        
        # دمج مع download_progress من SmartMediaDownloader إذا كان موجوداً
        if download_id in download_progress:
            smart_progress = download_progress[download_id]
            progress.update({
                'percent': smart_progress.get('percent', progress.get('percent', '0%')),
                'method': smart_progress.get('method', '')
            })
            if smart_progress.get('status'):
                progress['status'] = smart_progress['status']
        
        # تنسيق موحد للاستجابة
        return {
            'success': progress.get('status') == 'completed',
            'status': progress.get('status', 'unknown'),
            'percent': progress.get('percent', '0%'),
            'message': progress.get('message', ''),
            'download_id': download_id,
            'url': progress.get('url', ''),
            'quality': progress.get('quality', ''),
            'media_type': progress.get('media_type', ''),
            'file': progress.get('file'),
            'error': progress.get('error'),
            'started_at': progress.get('started_at'),
            'completed_at': progress.get('completed_at'),
            'failed_at': progress.get('failed_at'),
            'method': progress.get('method', '')
        }
    
    def _quality_to_format(self, quality: str, media_type: str) -> str:
        """تحويل quality string إلى format command"""
        # إذا كان audio
        if media_type == self.MEDIA_TYPE_AUDIO or quality == 'audio':
            return self.QUALITY_PRESETS.get('audio_best', 'audio')
        
        # البحث في presets
        quality_lower = quality.lower().strip()
        if quality_lower in self.QUALITY_PRESETS:
            return self.QUALITY_PRESETS[quality_lower]
        
        # إذا كان quality هو format command مباشرة
        if '[' in quality or '+' in quality or '/' in quality:
            return quality
        
        # افتراضي
        return self.QUALITY_PRESETS.get('auto', 'bestvideo+bestaudio/best')
    
    def _find_downloaded_file(self, url: str) -> Optional[str]:
        """البحث عن الملف المحمّل"""
        try:
            # الحصول على معلومات الفيديو
            info = self.downloader._get_video_info(url)
            title = info.get('title', 'video')
            
            # البحث عن الملف الأحدث
            download_folder = self.downloader.output_dir
            if download_folder.exists():
                video_files = []
                for file in download_folder.iterdir():
                    if file.is_file():
                        ext = file.suffix.lower()
                        if ext in ['.mp4', '.webm', '.mkv', '.mp3', '.m4a', '.mov', '.avi', '.flv']:
                            video_files.append((file, file.stat().st_mtime))
                
                if video_files:
                    video_files.sort(key=lambda x: x[1], reverse=True)
                    return str(video_files[0][0])
        except Exception as e:
            logger.error(f"Error finding downloaded file: {e}")
        
        return None
    
    def _transcribe_downloaded_file(self, video_file: str, options: dict) -> dict:
        """تفريغ نصي للملف المحمّل"""
        try:
            if not WHISPER_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Whisper not available'
                }
            
            # استخراج الصوت
            audio_file = video_file.rsplit('.', 1)[0] + '_audio.wav'
            
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-threads', '0',
                '-y',
                audio_file
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            
            # تحويل الصوت إلى نص
            model_size = options.get('model_size', 'base')
            language = options.get('language', 'auto')
            
            transcribe_result = transcribe_audio(audio_file, model_size, language, use_faster=True)
            
            # إنشاء ملف SRT
            srt_content = SubtitleProcessor.create_srt(
                transcribe_result['text'],
                duration=transcribe_result.get('duration'),
                segments=transcribe_result.get('segments', [])
            )
            
            srt_filename = f"{os.path.splitext(os.path.basename(video_file))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # تنظيف ملف الصوت المؤقت
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
            
            return {
                'success': True,
                'text': transcribe_result['text'],
                'language': transcribe_result.get('language', language),
                'transcript_file': srt_filename,
                'transcript_path': srt_path
            }
            
        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_quality_preset(self, quality_id: str, format_command: str):
        """إضافة جودة جديدة - قابل للتوسع"""
        self.QUALITY_PRESETS[quality_id.lower()] = format_command
        logger.info(f"Added quality preset: {quality_id} -> {format_command}")
    
    def get_available_qualities(self) -> list:
        """الحصول على قائمة الجودات المتاحة"""
        return list(self.QUALITY_PRESETS.keys())


# Initialize downloader - النظام القديم (للتوافق)
downloader = SmartMediaDownloader()

# Initialize unified download manager - النظام الموحد الجديد
unified_downloader = UnifiedDownloadManager()


class SubtitleProcessor:
    """معالج الترجمة"""
    
    @staticmethod
    def split_long_segments(segments: List[Dict], max_duration: float = 5.0, max_chars: int = 80) -> List[Dict]:
        """
        تقسيم segments الطويلة إلى أجزاء أصغر بناءً على الكلام والوقفات الطبيعية
        
        Args:
            segments: قائمة segments (يجب أن تحتوي على 'words' مع timestamps إذا كانت متاحة)
            max_duration: الحد الأقصى لمدة كل segment بالثواني
            max_chars: الحد الأقصى لعدد الأحرف في كل segment
        
        Returns:
            قائمة segments مقسمة بناءً على الكلام الفعلي
        """
        split_segments = []
        
        for segment in segments:
            start_time = float(segment.get('start', 0))
            end_time = float(segment.get('end', start_time + 3))
            text = segment.get('text', '').strip()
            words = segment.get('words', [])  # word timestamps إذا كانت متاحة
            
            if not text:
                continue
            
            duration = end_time - start_time
            
            # إذا كان segment قصيراً وعدد الأحرف معقول، استخدمه كما هو
            if duration <= max_duration and len(text) <= max_chars:
                split_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
                continue
            
            # إذا كانت word timestamps متاحة، استخدمها لتقسيم ذكي
            if words and len(words) > 0:
                split_segments.extend(
                    SubtitleProcessor._split_by_word_timestamps(
                        words, text, start_time, end_time, max_duration, max_chars
                    )
                )
            else:
                # تقسيم بناءً على النص فقط
                split_segments.extend(
                    SubtitleProcessor._split_by_text_only(
                        text, start_time, end_time, max_duration, max_chars
                    )
                )
        
        return split_segments


    @staticmethod
    def _split_by_word_timestamps(words: List[Dict], text: str, start_time: float, end_time: float, 
                                   max_duration: float, max_chars: int) -> List[Dict]:
        """تقسيم بناءً على word timestamps والوقفات الطبيعية"""
        split_segments = []
        
        # تجميع الكلمات مع timestamps
        word_list = []
        for word_info in words:
            if isinstance(word_info, dict):
                word = word_info.get('word', '').strip()
                word_start = float(word_info.get('start', 0))
                word_end = float(word_info.get('end', 0))
            else:
                # إذا كان word_info كائن وليس dict
                word = getattr(word_info, 'word', '').strip()
                word_start = float(getattr(word_info, 'start', 0))
                word_end = float(getattr(word_info, 'end', 0))
            
            if word:
                word_list.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })
        
        if not word_list:
            return SubtitleProcessor._split_by_text_only(text, start_time, end_time, max_duration, max_chars)
        
        # تقسيم بناءً على الوقفات الطبيعية والحدود
        current_segment_words = []
        current_start = word_list[0]['start']
        current_text = ""
        pause_threshold = 0.5  # وقفة 0.5 ثانية تعتبر نقطة تقسيم جيدة
        
        for i, word_data in enumerate(word_list):
            word = word_data['word']
            word_start = word_data['start']
            word_end = word_data['end']
            
            # حساب الوقفة قبل هذه الكلمة
            if i > 0:
                prev_word_end = word_list[i-1]['end']
                pause_duration = word_start - prev_word_end
            else:
                pause_duration = 0
            
            # إضافة الكلمة للـ segment الحالي
            current_segment_words.append(word)
            current_text = ' '.join(current_segment_words)
            current_end = word_end
            
            # تحديد ما إذا كان يجب إنهاء الـ segment الحالي
            should_split = False
            
            # 1. إذا كانت المدة تجاوزت الحد الأقصى
            if current_end - current_start > max_duration:
                should_split = True
            
            # 2. إذا كان عدد الأحرف تجاوز الحد الأقصى
            elif len(current_text) > max_chars:
                should_split = True
            
            # 3. إذا كانت هناك وقفة طويلة (pause) - نقطة تقسيم طبيعية
            elif pause_duration > pause_threshold and len(current_segment_words) > 3:
                should_split = True
            
            # 4. إذا انتهت الجملة بعلامة توقف (نقطة، علامة استفهام، إلخ)
            elif word.rstrip().endswith(('.', '!', '?', '،', '؛')) and len(current_segment_words) >= 3:
                should_split = True
            
            # 5. إذا كانت هناك فاصلة ووصلنا لطول معقول
            elif word.rstrip().endswith((',', '،', ';', '؛')) and len(current_text) > max_chars * 0.7:
                should_split = True
            
            if should_split:
                # حفظ الـ segment الحالي
                segment_text = ' '.join(current_segment_words).strip()
                if segment_text:
                    split_segments.append({
                        'start': current_start,
                        'end': current_end,
                        'text': segment_text
                    })
                
                # بدء segment جديد
                current_segment_words = []
                current_start = word_start
                current_text = ""
        
        # إضافة الـ segment الأخير
        if current_segment_words:
            segment_text = ' '.join(current_segment_words).strip()
            if segment_text:
                final_end = word_list[-1]['end']
                split_segments.append({
                    'start': current_start,
                    'end': final_end,
                    'text': segment_text
                })
        
        return split_segments
    
    @staticmethod
    def _split_by_text_only(text: str, start_time: float, end_time: float, 
                            max_duration: float, max_chars: int) -> List[Dict]:
        """تقسيم بناءً على النص فقط (fallback عندما لا تكون word timestamps متاحة)"""
        split_segments = []
        duration = end_time - start_time
        
        # تقسيم على علامات التوقف الطبيعية
        # تقسيم على علامات التوقف مع الاحتفاظ بها
        sentences = re.split(r'([.!?،؛]\s*)', text)
        
        current_sentence = ""
        current_start = start_time
        total_chars = len(text)
        
        for i, part in enumerate(sentences):
            if not part.strip():
                continue
            
            current_sentence += part
            
            # حساب الوقت بناءً على نسبة الأحرف
            chars_ratio = len(current_sentence) / total_chars if total_chars > 0 else 1.0
            current_end = start_time + (duration * chars_ratio)
            
            # تحديد ما إذا كان يجب إنهاء الـ segment
            should_split = False
            
            # إذا انتهت الجملة بعلامة توقف
            if part.strip().endswith(('.', '!', '?', '،', '؛')):
                should_split = True
            
            # إذا تجاوزت المدة أو الأحرف الحد الأقصى
            elif current_end - current_start > max_duration or len(current_sentence) >= max_chars:
                should_split = True
            
            if should_split and current_sentence.strip():
                # التأكد من أن المدة معقولة
                if current_end - current_start > max_duration:
                    current_end = current_start + max_duration
                
                split_segments.append({
                    'start': current_start,
                    'end': current_end,
                    'text': current_sentence.strip()
                })
                
                current_start = current_end
                current_sentence = ""
        
        # إضافة أي نص متبقي
        if current_sentence.strip():
            split_segments.append({
                'start': current_start,
                'end': end_time,
                'text': current_sentence.strip()
            })
        
        return split_segments
    
    @staticmethod
    def create_srt(text: str, duration: float = None, segments: List[Dict] = None) -> str:
        """إنشاء ملف SRT"""
        srt_content = []
        
        if segments:
            for i, segment in enumerate(segments):
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 3))
                text_segment = segment.get('text', '').strip()
                
                if end_time <= start_time:
                    end_time = start_time + 3.0
                
                start_str = SubtitleProcessor.seconds_to_srt_time(start_time)
                end_str = SubtitleProcessor.seconds_to_srt_time(end_time)
                
                srt_content.append(f"{i + 1}")
                srt_content.append(f"{start_str} --> {end_str}")
                srt_content.append(text_segment)
                srt_content.append("")
            
            return '\n'.join(srt_content)
        
        # تقسيم النص
        lines = text.split('\n')
        segments_list = []
        current_segment = []
        char_count = 0
        
        for line in lines:
            if not line.strip():
                continue
            words = line.split()
            for word in words:
                current_segment.append(word)
                char_count += len(word) + 1
                
                if char_count >= 40 or word.endswith(('.', '!', '?', '،', '؛')):
                    segments_list.append(' '.join(current_segment))
                    current_segment = []
                    char_count = 0
        
        if current_segment:
            segments_list.append(' '.join(current_segment))
        
        if not segments_list:
            return ""
        
        # حساب التوقيتات
        if duration and duration > 0:
            segment_duration = max(duration / len(segments_list), 2.0)
        else:
            segment_duration = 3.0
        
        for i, segment in enumerate(segments_list):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration) if duration else (i + 1) * segment_duration
            
            start_str = SubtitleProcessor.seconds_to_srt_time(start_time)
            end_str = SubtitleProcessor.seconds_to_srt_time(end_time)
            
            srt_content.append(f"{i + 1}")
            srt_content.append(f"{start_str} --> {end_str}")
            srt_content.append(segment.strip())
            srt_content.append("")
        
        return '\n'.join(srt_content)
    
    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        """تحويل الثواني إلى تنسيق SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs % 1) * 1000)
        secs = int(secs)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    @staticmethod
    def create_ass_subtitle(
        srt_content: str,
        font_size: int = 20,
        font_color: str = "FFFFFF",
        bg_color: str = "000000",
        bg_opacity: int = 128,
        position: str = "bottom",
        font_name: str = "Arial",
        vertical_offset: int = 0
    ) -> str:
        """إنشاء ملف ASS"""
        # ASS يستخدم تنسيق BGR (Blue-Green-Red) بدلاً من RGB
        # تحويل من RGB hex إلى BGR hex
        if len(font_color) == 6:
            # RGB: RRGGBB -> BGR: BBGGRR
            r = font_color[0:2]
            g = font_color[2:4]
            b = font_color[4:6]
            ass_font_color = f"&H00{b}{g}{r}"  # BGR format
        else:
            ass_font_color = "&H00FFFFFF"  # أبيض افتراضي
        
        if len(bg_color) == 6:
            # RGB: RRGGBB -> BGR: BBGGRR
            r = bg_color[0:2]
            g = bg_color[2:4]
            b = bg_color[4:6]
            # bg_opacity في ASS هو قيمة hex (00-FF)
            opacity_hex = format(min(255, max(0, bg_opacity)), '02X')
            ass_bg_color = f"&H{opacity_hex}{b}{g}{r}"  # BGR format with opacity
        else:
            ass_bg_color = f"&H{format(min(255, max(0, bg_opacity)), '02X')}000000"  # أسود افتراضي
        
        alignment = {'top': '8', 'center': '5', 'bottom': '2'}.get(position, '2')
        
        # حساب margin_v بناءً على الموضع و vertical_offset
        if position == 'top':
            margin_v = max(10, 10 - vertical_offset)  # سالب يرفع لأعلى
        elif position == 'center':
            margin_v = 10
        else:  # bottom
            margin_v = max(10, 10 + vertical_offset)  # موجب يخفض لأسفل
        
        ass_header = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{ass_font_color},&H000000FF,&H00000000,{ass_bg_color},0,0,0,0,100,100,0,0,3,2,1,{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        events = []
        lines = srt_content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            if lines[i].strip().isdigit():
                if i + 2 < len(lines):
                    timing = lines[i + 1].strip()
                    text = lines[i + 2].strip()
                    
                    if ' --> ' in timing:
                        start, end = timing.split(' --> ')
                        start = start.replace(',', '.').strip()
                        end = end.replace(',', '.').strip()
                        
                        start_ass = SubtitleProcessor.srt_time_to_ass(start)
                        end_ass = SubtitleProcessor.srt_time_to_ass(end)
                        
                        # استخدام Effect لتطبيق vertical_offset بدقة
                        effect = ''
                        if vertical_offset != 0:
                            if position == 'center':
                                # للمركز: استخدام pos لتحديد الموضع بدقة
                                y_pos = 540 - vertical_offset  # 540 هو منتصف 1080
                                effect = f"\\pos(960,{y_pos})"
                            elif position == 'top':
                                # للأعلى: استخدام an=8 و pos
                                y_pos = 50 - vertical_offset
                                effect = f"\\an8\\pos(960,{y_pos})"
                            else:  # bottom
                                # للأسفل: استخدام an=2 و pos
                                y_pos = 1030 - vertical_offset  # 1030 = 1080 - 50
                                effect = f"\\an2\\pos(960,{y_pos})"
                        
                        events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,{effect},{text}")
                    
                    i += 4
                else:
                    i += 1
            else:
                i += 1
        
        return ass_header + '\n'.join(events)
    
    @staticmethod
    def srt_time_to_ass(srt_time: str) -> str:
        """تحويل وقت SRT إلى ASS"""
        parts = srt_time.replace(',', '.').split(':')
        if len(parts) >= 3:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            return f"{h}:{m:02d}:{s:05.2f}"
        return "0:00:00.00"


class VideoProcessor:
    """معالج الفيديو"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """التحقق من توفر ffmpeg"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """الحصول على مدة الفيديو"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return None
    
    @staticmethod
    def get_video_dimensions(video_path: str) -> Dict:
        """الحصول على مقاسات الفيديو"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'streams' in data and len(data['streams']) > 0:
                    width = int(data['streams'][0].get('width', 0))
                    height = int(data['streams'][0].get('height', 0))
                    return {'width': width, 'height': height, 'aspect_ratio': width / height if height > 0 else 1.0}
        except:
            pass
        return {'width': 1920, 'height': 1080, 'aspect_ratio': 16/9}
    
    @staticmethod
    def create_subtitle_filter(subtitle_path: str, settings: Dict = None) -> str:
        """إنشاء فلتر الترجمة"""
        if not settings:
            settings = {}
        
        # التأكد من أن المسار موجود
        if not os.path.exists(subtitle_path):
            logger.error(f"Subtitle file not found: {subtitle_path}")
            raise Exception(f"ملف الترجمة غير موجود: {subtitle_path}")
        
        # استخدام مسار مطلق
        abs_subtitle_path = os.path.abspath(subtitle_path)
        logger.info(f"Using subtitle path: {abs_subtitle_path}")
        
        # Escape المسار بشكل صحيح لـ FFmpeg (يعمل على Windows و Mac)
        # FFmpeg يتعامل مع المسارات بشكل مختلف حسب النظام
        import platform
        if platform.system() == 'Windows':
            # Windows: استخدام backslash مع escape
            subtitle_path_escaped = abs_subtitle_path.replace("\\", "\\\\")
            subtitle_path_escaped = subtitle_path_escaped.replace(":", "\\:")
        else:
            # Unix/Mac: استخدام forward slash
            subtitle_path_escaped = abs_subtitle_path.replace("\\", "/")
            subtitle_path_escaped = subtitle_path_escaped.replace(":", "\\:")
        
        # Escape للـ shell
        subtitle_path_escaped = subtitle_path_escaped.replace("'", "'\\''")
        subtitle_path_escaped = f"'{subtitle_path_escaped}'"
        
        if subtitle_path.endswith('.ass'):
            filter_str = f"ass={subtitle_path_escaped}"
            logger.info(f"ASS filter: {filter_str}")
            return filter_str
        
        filter_str = f"subtitles={subtitle_path_escaped}"
        style_options = []
        
        font_name = settings.get('font_name', 'Arial')
        style_options.append(f"FontName={font_name}")
        
        if settings.get('font_size'):
            style_options.append(f"FontSize={int(settings['font_size'])}")
        
        if settings.get('font_color'):
            color = settings['font_color'].replace('#', '')
            if len(color) == 6:
                r = color[0:2]
                g = color[2:4]
                b = color[4:6]
                style_options.append(f"PrimaryColour=&H00{b}{g}{r}")
        
        if settings.get('bg_color'):
            bg_color = settings['bg_color'].replace('#', '')
            opacity = int(settings.get('bg_opacity', 180))
            if len(bg_color) == 6:
                r = bg_color[0:2]
                g = bg_color[2:4]
                b = bg_color[4:6]
                opacity_hex = format(opacity, '02X')
                style_options.append(f"BackColour=&H{opacity_hex}{b}{g}{r}")
        
        alignment = {'top': '8', 'center': '5', 'bottom': '2'}.get(settings.get('position', 'bottom'), '2')
        style_options.append(f"Alignment={alignment}")
        
        if style_options:
            style_str = ','.join(style_options)
            filter_str = f"{filter_str}:force_style='{style_str}'"
        
        logger.info(f"Subtitle filter: {filter_str}")
        return filter_str
    
    @staticmethod
    def get_quality_settings(quality: str, use_filter: bool = False) -> List[str]:
        """الحصول على إعدادات الجودة"""
        if quality == 'original' and not use_filter:
            return ['-c:v', 'copy']
        elif quality == 'original' and use_filter:
            return [
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '20',
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'
            ]
        elif quality == 'high':
            return [
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '20',
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'
            ]
        elif quality == 'medium':
            return [
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'
            ]
        elif quality == 'low':
            return [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                '-profile:v', 'baseline',
                '-level', '3.1',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=w=min(iw\\,1280):h=-2',
                '-threads', '0'
            ]
        else:
            return ['-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-threads', '0']
    
    @staticmethod
    def merge_subtitles(
        video_path: str,
        subtitle_path: str,
        output_path: str,
        quality: str = "original",
        subtitle_settings: Dict = None
    ) -> Dict:
        """دمج الترجمة مع الفيديو"""
        result = {
            'success': False,
            'message': '',
            'output_file': None
        }
        
        try:
            if not VideoProcessor.check_ffmpeg():
                raise Exception("ffmpeg غير متوفر. يرجى تثبيته أولاً")
            
            subtitle_filter = VideoProcessor.create_subtitle_filter(subtitle_path, subtitle_settings)
            quality_settings = VideoProcessor.get_quality_settings(quality, use_filter=True)
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', subtitle_filter,
                '-c:a', 'copy',
                '-threads', '0',
                '-movflags', '+faststart',
            ]
            cmd.extend(quality_settings)
            cmd.extend(['-y', output_path])
            
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=600)
                
                if process.returncode == 0:
                    result['success'] = True
                    result['message'] = 'تم دمج الترجمة بنجاح'
                    result['output_file'] = output_path
                else:
                    error_msg = stderr or stdout
                    logger.error(f"FFmpeg error: {error_msg}")
                    raise Exception(f"FFmpeg error: {error_msg}")
            except subprocess.TimeoutExpired:
                process.kill()
                raise Exception("انتهت مهلة المعالجة. الملف كبير جداً")
        
        except Exception as e:
            result['message'] = f'خطأ في دمج الترجمة: {str(e)}'
            logger.error(f"Merge error: {e}")
            logger.error(traceback.format_exc())
        
        return result


# تهيئة الكائنات
downloader = SmartMediaDownloader()


# === Routes ===

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html',
                         whisper_available=WHISPER_AVAILABLE,
                         translator_available=TRANSLATOR_AVAILABLE)


@app.route('/api/instant-translate', methods=['POST'])
def api_instant_translate():
    """API للترجمة الفورية"""
    try:
        data = request.json
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'success': False, 'message': 'لم يتم استلام بيانات'}), 400
        
        step = data.get('step')
        
        if not step:
            logger.error(f"Missing step in request data: {data.keys()}")
            return jsonify({'success': False, 'message': 'خطوة غير محددة'}), 400
        
        logger.info(f"Processing step: {step}")
        logger.debug(f"Request data keys: {data.keys()}")
        
        if step == 'download':
            url = data.get('url')
            quality = data.get('quality', '720p')
            
            if not url:
                return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
            
            # استخدام النظام الموحد
            result = unified_downloader.start_download(
                url=url,
                quality=quality,
                media_type=unified_downloader.MEDIA_TYPE_VIDEO
            )
            
            if result['success']:
                download_id = result.get('download_id')
                
                # الانتظار قليلاً ثم التحقق من التقدم
                import time
                time.sleep(3)  # انتظار قصير للبدء
                
                # الحصول على حالة التحميل
                progress = unified_downloader.get_progress(download_id)
                
                if progress.get('status') == 'completed' and progress.get('file'):
                    video_file = progress.get('file')
                    if isinstance(video_file, dict):
                        video_file = video_file.get('video', video_file)
                    
                    if not os.path.isabs(video_file):
                        video_file = os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file))
                    video_file = os.path.normpath(video_file)
                    
                    if not os.path.exists(video_file):
                        basename = os.path.basename(video_file)
                        possible_paths = [
                            os.path.join(app.config['DOWNLOAD_FOLDER'], basename),
                            video_file
                        ]
                        for path in possible_paths:
                            if os.path.exists(path):
                                video_file = path
                                break
                    
                    # استخدام ملف مؤقت بدلاً من session
                    temp_file = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    with open(temp_file, 'w') as f:
                        f.write(video_file)
                    
                    return jsonify({
                        'success': True,
                        'file': video_file,
                        'info': progress.get('info', {}),
                        'temp_file': os.path.basename(temp_file),
                        'download_id': download_id
                    })
                else:
                    # التحميل ما زال قيد التنفيذ
                    return jsonify({
                        'success': True,
                        'download_id': download_id,
                        'status': 'downloading',
                        'message': 'جاري التحميل...',
                        'check_progress': f'/api/media/progress/{download_id}'
                    })
            else:
                error_message = result.get('message', 'حدث خطأ غير معروف أثناء التحميل')
                logger.error(f"Download failed: {error_message}")
                return jsonify({
                    'success': False,
                    'message': error_message
                }), 400
        
        elif step == 'extract_audio':
            video_file = data.get('video_file')
            
            # قراءة من الملف المؤقت إذا لزم الأمر
            if not video_file and data.get('temp_video_file'):
                temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], data['temp_video_file'])
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        video_file = f.read().strip()
            
            if video_file and not os.path.isabs(video_file):
                normalized = video_file.replace('downloads/', '').replace('downloads\\', '')
                possible_paths = [
                    os.path.join(app.config['DOWNLOAD_FOLDER'], normalized),
                    os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file)),
                    video_file,
                    normalized
                ]
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        video_file = abs_path
                        break
            
            if not video_file or not os.path.exists(video_file):
                return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
            
            audio_file = video_file.rsplit('.', 1)[0] + '_audio.wav'
            
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-threads', '0',
                '-y',
                audio_file
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                # حفظ في ملف مؤقت بدلاً من session
                temp_file = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(temp_file, 'w') as f:
                    f.write(audio_file)
                
                return jsonify({
                    'success': True,
                    'audio_file': audio_file,
                    'temp_file': os.path.basename(temp_file)
                })
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr}")
                return jsonify({'success': False, 'message': f'خطأ في استخراج الصوت: {e.stderr}'}), 500
            except subprocess.TimeoutExpired:
                return jsonify({'success': False, 'message': 'انتهت مهلة استخراج الصوت'}), 500
        
        elif step == 'transcribe':
            if not WHISPER_AVAILABLE:
                return jsonify({'success': False, 'message': 'Whisper غير متوفر'}), 503
            
            audio_file = data.get('audio_file')
            model_size = data.get('model', 'base')
            language = data.get('language', 'auto')
            
            # قراءة من الملف المؤقت إذا لزم الأمر
            if not audio_file and data.get('temp_audio_file'):
                temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], data['temp_audio_file'])
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        audio_file = f.read().strip()
            
            if not audio_file or not os.path.exists(audio_file):
                return jsonify({'success': False, 'message': 'ملف الصوت غير موجود'}), 400
            
            # استخدام الدالة المساعدة التي تستخدم Faster Whisper تلقائياً
            result = transcribe_audio(audio_file, model_size, language, use_faster=True)
            
            # حفظ في ملف مؤقت بدلاً من session
            temp_file = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'text': result['text'],
                    'language': result.get('language', language),
                    'segments': result.get('segments', [])
                }, f, ensure_ascii=False)
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'language': result.get('language', language),
                'segments': result.get('segments', []),
                'temp_file': os.path.basename(temp_file)
            })
        
        elif step == 'translate':
            if not TRANSLATOR_AVAILABLE:
                return jsonify({'success': False, 'message': 'المترجم غير متوفر'}), 503
            
            text = data.get('text') or session.get('transcript')
            source_lang = data.get('source_lang', 'auto')
            
            if not text:
                return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
            
            translator = GoogleTranslator(source=source_lang, target='ar')
            translated = translator.translate(text)
            
            # حفظ في ملف مؤقت بدلاً من session
            temp_file = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(translated)
            
            return jsonify({
                'success': True,
                'translated_text': translated,
                'temp_file': os.path.basename(temp_file)
            })
        
        elif step == 'merge':
            video_file = data.get('video_file')
            subtitle_text = data.get('subtitle_text')  # هذا للنص الكامل فقط (fallback)
            settings = data.get('settings', {})
            quality = data.get('quality', 'original')
            
            # قراءة من الملفات المؤقتة إذا لزم الأمر
            if not video_file and data.get('temp_video_file'):
                temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], data['temp_video_file'])
                if os.path.exists(temp_path):
                    with open(temp_path, 'r') as f:
                        video_file = f.read().strip()
            
            if not subtitle_text and data.get('temp_translated_file'):
                temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], data['temp_translated_file'])
                if os.path.exists(temp_path):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        subtitle_text = f.read().strip()
            
            if video_file and not os.path.isabs(video_file):
                possible_paths = [
                    os.path.join(app.config['DOWNLOAD_FOLDER'], video_file),
                    os.path.join(app.config['UPLOAD_FOLDER'], video_file),
                    video_file
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        video_file = path
                        break
            
            if not video_file or not os.path.exists(video_file):
                return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
            
            video_duration = VideoProcessor.get_video_duration(video_file)
            
            # قراءة البيانات من الملفات المؤقتة بدلاً من session
            whisper_segments = None
            original_text = ''
            source_language = 'auto'
            
            # البحث عن ملفات مؤقتة
            temp_transcript_file = data.get('temp_transcript_file')
            if temp_transcript_file:
                temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], temp_transcript_file)
                if os.path.exists(temp_path):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                        whisper_segments = transcript_data.get('segments', [])
                        original_text = transcript_data.get('text', '')
                        source_language = transcript_data.get('language', 'auto')
            
            # استخدام segments المترجمة فقط - لا نستخدم subtitle_text
            if whisper_segments and len(whisper_segments) > 0:
                # تقسيم segments الطويلة إلى أجزاء أصغر بناءً على الكلام والوقفات
                logger.info(f"Original segments count: {len(whisper_segments)}")
                
                # التحقق من وجود word timestamps
                has_word_timestamps = any(seg.get('words') for seg in whisper_segments)
                logger.info(f"Has word timestamps: {has_word_timestamps}")
                
                whisper_segments = SubtitleProcessor.split_long_segments(
                    whisper_segments, 
                    max_duration=5.0,  # 5 ثواني كحد أقصى
                    max_chars=80       # 80 حرف كحد أقصى
                )
                logger.info(f"After intelligent splitting: {len(whisper_segments)} segments")
                
                segments_for_srt = []
                translator = GoogleTranslator(source=source_language, target='ar')
                
                logger.info(f"Translating {len(whisper_segments)} segments...")
                
                for i, segment in enumerate(whisper_segments):
                    start_time = float(segment.get('start', 0))
                    end_time = float(segment.get('end', start_time + 3))
                    original_segment_text = segment.get('text', '').strip()
                    
                    if not original_segment_text:
                        continue
                    
                    # التأكد من أن المدة معقولة
                    if end_time - start_time > 7.0:
                        # تقسيم إضافي إذا كانت المدة طويلة جداً
                        words = original_segment_text.split()
                        words_per_second = len(words) / (end_time - start_time) if (end_time - start_time) > 0 else 2
                        target_words = int(words_per_second * 5)  # 5 ثواني كحد أقصى
                        
                        if len(words) > target_words:
                            # تقسيم إلى جمل أصغر
                            sub_segments = []
                            current_words = []
                            current_start = start_time
                            word_duration = (end_time - start_time) / len(words) if len(words) > 0 else 0.5
                            
                            for j, word in enumerate(words):
                                current_words.append(word)
                                
                                # إذا وصلنا للحد الأقصى أو انتهت الجملة
                                if len(current_words) >= target_words or word.endswith(('.', '!', '?', '،', '؛')):
                                    sub_text = ' '.join(current_words)
                                    sub_end = current_start + (len(current_words) * word_duration)
                                    
                                    sub_segments.append({
                                        'start': current_start,
                                        'end': sub_end,
                                        'text': sub_text
                                    })
                                    
                                    current_start = sub_end
                                    current_words = []
                            
                            # إضافة أي كلمات متبقية
                            if current_words:
                                sub_text = ' '.join(current_words)
                                sub_segments.append({
                                    'start': current_start,
                                    'end': end_time,
                                    'text': sub_text
                                })
                            
                            # ترجمة sub_segments
                            for sub_seg in sub_segments:
                                try:
                                    translated_text = translator.translate(sub_seg['text'])
                                    segments_for_srt.append({
                                        'start': sub_seg['start'],
                                        'end': sub_seg['end'],
                                        'text': translated_text.strip()
                                    })
                                except Exception as e:
                                    logger.warning(f"Translation failed: {e}")
                                    segments_for_srt.append({
                                        'start': sub_seg['start'],
                                        'end': sub_seg['end'],
                                        'text': sub_seg['text']
                                    })
                            continue
                    
                    try:
                        translated_segment_text = translator.translate(original_segment_text)
                    except Exception as e:
                        logger.warning(f"Translation failed for segment {i+1}: {e}")
                        translated_segment_text = original_segment_text
                    
                    if translated_segment_text.strip():
                        segments_for_srt.append({
                            'start': start_time,
                            'end': end_time,
                            'text': translated_segment_text.strip()
                        })
                
                # استخدام segments المترجمة فقط لإنشاء SRT
                if segments_for_srt:
                    logger.info(f"Creating SRT from {len(segments_for_srt)} translated segments")
                    srt_content = SubtitleProcessor.create_srt(
                        '',  # نص فارغ - نستخدم segments فقط
                        duration=video_duration,
                        segments=segments_for_srt
                    )
                else:
                    # إذا لم تكن هناك segments، استخدم النص المترجم (fallback)
                    logger.warning("No segments available, using full text as fallback")
                    srt_content = SubtitleProcessor.create_srt(
                        subtitle_text or original_text,
                        duration=video_duration
                    )
            else:
                # إذا لم تكن هناك segments، استخدم النص المترجم فقط
                logger.warning("No whisper segments found, using full text")
                srt_content = SubtitleProcessor.create_srt(
                    subtitle_text or '',
                    duration=video_duration
                )
            
            srt_filename = f"instant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # استخراج الإعدادات من data مباشرة إذا لم تكن في settings
            font_size = int(data.get('font_size') or settings.get('font_size', 22))
            font_color = data.get('font_color') or settings.get('font_color', '#FFFFFF')
            bg_color = data.get('bg_color') or settings.get('bg_color', '#000000')
            bg_opacity = int(data.get('bg_opacity') or settings.get('bg_opacity', 128))
            position = data.get('position') or settings.get('position', 'bottom')
            font_name = data.get('font_name') or settings.get('font_name', 'Arial')
            vertical_offset = int(data.get('vertical_offset') or settings.get('vertical_offset', 0))
            
            # التأكد من أن font_color و bg_color بدون #
            font_color_clean = font_color.replace('#', '') if font_color else 'FFFFFF'
            bg_color_clean = bg_color.replace('#', '') if bg_color else '000000'
            
            ass_content = SubtitleProcessor.create_ass_subtitle(
                srt_content,
                font_size=font_size,
                font_color=font_color_clean,
                bg_color=bg_color_clean,
                bg_opacity=bg_opacity,
                position=position,
                font_name=font_name,
                vertical_offset=vertical_offset
            )
            
            ass_path = srt_path.replace('.srt', '.ass')
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            # التأكد من أن الملف تم إنشاؤه بنجاح
            if not os.path.exists(ass_path):
                logger.error(f"Failed to create ASS file: {ass_path}")
                return jsonify({'success': False, 'message': 'فشل إنشاء ملف الترجمة'}), 500
            
            logger.info(f"ASS file created successfully: {ass_path}")
            logger.info(f"ASS file size: {os.path.getsize(ass_path)} bytes")
            
            output_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            subtitle_settings = {
                'font_name': font_name,
                'font_size': font_size,
                'font_color': font_color,
                'bg_color': bg_color,
                'bg_opacity': bg_opacity,
                'position': position,
                'vertical_offset': vertical_offset
            }
            
            # التأكد من أن ملف ASS موجود قبل الدمج
            if not os.path.exists(ass_path):
                logger.error(f"ASS file not found: {ass_path}")
                return jsonify({'success': False, 'message': f'ملف الترجمة غير موجود: {ass_path}'}), 500
            
            logger.info(f"Merging video: {video_file}")
            logger.info(f"Using ASS file: {ass_path}")
            logger.info(f"ASS file exists: {os.path.exists(ass_path)}")
            
            result = VideoProcessor.merge_subtitles(
                video_file,
                ass_path,
                output_path,
                quality=quality,
                subtitle_settings=subtitle_settings
            )
            
            if result['success']:
                # تنظيف الملفات المؤقتة
                try:
                    temp_files = [
                        data.get('temp_video_file'),
                        data.get('temp_audio_file'),
                        data.get('temp_transcript_file'),
                        data.get('temp_translated_file')
                    ]
                    for temp_file in temp_files:
                        if temp_file:
                            temp_path = os.path.join(app.config['DOWNLOAD_FOLDER'], temp_file)
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                except:
                    pass
                
                return jsonify({
                    'success': True,
                    'message': 'تمت الترجمة الفورية بنجاح!',
                    'output_file': output_filename,
                    'download_url': f'/download/{output_filename}'
                })
            else:
                return jsonify({'success': False, 'message': result['message']}), 500
        
        else:
            return jsonify({'success': False, 'message': f'خطوة غير صحيحة: {step}'}), 400
    
    except Exception as e:
        logger.error(f"Instant translate error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/media/analyze', methods=['POST'])
def api_analyze_url():
    """تحليل الرابط والحصول على التنسيقات المتاحة"""
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    result = downloader.get_available_formats(url)
    return jsonify(result)


@app.route('/api/media/download', methods=['POST'])
def api_start_media_download():
    """بدء التحميل مع تنسيق محدد - API موحد"""
    data = request.json
    url = data.get('url', '')
    quality = data.get('quality', data.get('format', 'auto'))
    media_type = data.get('media_type', unified_downloader.MEDIA_TYPE_VIDEO)
    options = data.get('options', {})
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    # استخدام النظام الموحد
    result = unified_downloader.start_download(
        url=url,
        quality=quality,
        media_type=media_type,
        options=options
    )
    
    return jsonify(result)


@app.route('/api/media/progress/<download_id>')
def api_get_download_progress(download_id):
    """الحصول على حالة التحميل - API موحد"""
    progress = unified_downloader.get_progress(download_id)
    return jsonify(progress)


@app.route('/api/download', methods=['POST'])
def api_download():
    """API للتحميل - موحد مع النظام الجديد"""
    try:
        data = request.json
        url = data.get('url')
        quality = data.get('quality', 'auto')
        media_type = data.get('media_type', unified_downloader.MEDIA_TYPE_VIDEO)
        options = data.get('options', {})
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        # استخدام النظام الموحد
        result = unified_downloader.start_download(
            url=url,
            quality=quality,
            media_type=media_type,
            options=options
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Download API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get-video-thumbnail', methods=['POST'])
def api_get_video_thumbnail():
    """استخراج صورة مصغرة من الفيديو"""
    try:
        data = request.json
        video_file = data.get('video_file') or session.get('video_file')
        
        if video_file and not os.path.isabs(video_file):
            normalized = video_file.replace('downloads/', '').replace('downloads\\', '')
            possible_paths = [
                os.path.join(app.config['DOWNLOAD_FOLDER'], normalized),
                os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file)),
                video_file,
                normalized
            ]
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    video_file = abs_path
                    break
        
        if not video_file or not os.path.exists(video_file):
            return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
        
        if not VideoProcessor.check_ffmpeg():
            return jsonify({'success': False, 'message': 'ffmpeg غير متوفر'}), 503
        
        dimensions = VideoProcessor.get_video_dimensions(video_file)
        
        thumbnail_filename = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], thumbnail_filename)
        
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-ss', '00:00:01',
            '-vframes', '1',
            '-vf', 'scale=640:-1',
            '-threads', '0',
            '-y',
            thumbnail_path
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if process.returncode == 0 and os.path.exists(thumbnail_path):
            return jsonify({
                'success': True,
                'thumbnail_url': f'/download/{thumbnail_filename}',
                'thumbnail_path': thumbnail_path,
                'dimensions': dimensions
            })
        else:
            return jsonify({'success': False, 'message': 'فشل استخراج الصورة المصغرة'}), 500
    
    except Exception as e:
        logger.error(f"Thumbnail extraction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/transcribe-from-url', methods=['POST'])
def api_transcribe_from_url():
    """تحويل الفيديو من رابط إلى نص - موحد"""
    if not WHISPER_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
        return jsonify({'success': False, 'message': 'لا توجد مكتبة متاحة لتحويل الصوت إلى نص'}), 503
    
    try:
        data = request.json
        url = data.get('url')
        language = data.get('language', 'auto')
        model_size = data.get('model', 'base')
        quality = data.get('quality', '720p')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        # استخدام النظام الموحد مع media_type=transcribe
        result = unified_downloader.start_download(
            url=url,
            quality=quality,
            media_type=unified_downloader.MEDIA_TYPE_TRANSCRIBE,
            options={
                'language': language,
                'model_size': model_size
            }
        )
        
        if not result.get('success'):
            return jsonify(result), 400
        
        download_id = result.get('download_id')
        
        # إرجاع download_id للتحقق من التقدم لاحقاً
        return jsonify({
            'success': True,
            'download_id': download_id,
            'message': 'تم بدء التحميل والتفريغ النصي',
            'check_progress': f'/api/media/progress/{download_id}'
        })
        
    except Exception as e:
        logger.error(f"Transcribe from URL error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """تحويل الصوت/الفيديو إلى نص"""
    if not WHISPER_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
        return jsonify({'success': False, 'message': 'لا توجد مكتبة متاحة لتحويل الصوت إلى نص'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        language = request.form.get('language', 'auto')
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'لم يتم اختيار ملف'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        model_size = request.form.get('model', 'base')
        
        # استخدام الدالة المساعدة التي تستخدم Faster Whisper تلقائياً
        result = transcribe_audio(filepath, model_size, language, use_faster=True)
        
        # إنشاء ملف SRT مع segments
        srt_content = SubtitleProcessor.create_srt(
            result['text'],
            duration=result.get('duration'),
            segments=result.get('segments', [])
        )
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', language),
            'srt_file': srt_filename,
            'segments': result.get('segments', [])
        })
    
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def api_translate():
    """ترجمة النص"""
    if not TRANSLATOR_AVAILABLE:
        return jsonify({'success': False, 'message': 'المترجم غير متوفر'}), 503
    
    try:
        data = request.json
        text = data.get('text')
        target_lang = data.get('target_lang', 'ar')
        source_lang = data.get('source_lang', 'auto')
        
        if not text:
            return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        return jsonify({
            'success': True,
            'translated_text': translated,
            'source_lang': source_lang,
            'target_lang': target_lang
        })
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/merge-subtitle', methods=['POST'])
def api_merge_subtitle():
    """دمج الترجمة مع الفيديو"""
    try:
        if 'video' not in request.files or 'subtitle' not in request.files:
            return jsonify({'success': False, 'message': 'الرجاء رفع الفيديو والترجمة'}), 400
        
        video_file = request.files['video']
        subtitle_file = request.files['subtitle']
        
        settings = {
            'font_size': int(request.form.get('font_size', 20)),
            'font_color': request.form.get('font_color', '#FFFFFF'),
            'bg_color': request.form.get('bg_color', '#000000'),
            'bg_opacity': int(request.form.get('bg_opacity', 128)),
            'position': request.form.get('position', 'bottom'),
            'font_name': request.form.get('font_name', 'Arial'),
            'quality': request.form.get('quality', 'medium')
        }
        
        video_filename = secure_filename(video_file.filename)
        subtitle_filename = secure_filename(subtitle_file.filename)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        subtitle_path = os.path.join(app.config['SUBTITLE_FOLDER'], subtitle_filename)
        
        video_file.save(video_path)
        subtitle_file.save(subtitle_path)
        
        if subtitle_filename.endswith('.srt'):
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            ass_content = SubtitleProcessor.create_ass_subtitle(
                srt_content,
                font_size=settings['font_size'],
                font_color=settings['font_color'],
                bg_color=settings['bg_color'],
                bg_opacity=settings['bg_opacity'],
                position=settings['position'],
                font_name=settings['font_name']
            )
            
            ass_path = subtitle_path.replace('.srt', '.ass')
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            subtitle_path = ass_path
        
        output_filename = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        result = VideoProcessor.merge_subtitles(
            video_path,
            subtitle_path,
            output_path,
            quality=settings['quality'],
            subtitle_settings=settings
        )
        
        os.remove(video_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': result['message'],
                'output_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"Merge subtitle error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/storage-info', methods=['GET'])
def api_storage_info():
    """الحصول على معلومات الحجم المستخدم"""
    try:
        total_size = 0
        
        folders = [
            app.config['DOWNLOAD_FOLDER'],
            app.config['UPLOAD_FOLDER'],
            app.config['OUTPUT_FOLDER'],
            app.config['SUBTITLE_FOLDER']
        ]
        
        for folder in folders:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except:
                            pass
        
        size_mb = total_size / (1024 * 1024)
        
        return jsonify({
            'success': True,
            'size_bytes': total_size,
            'size_mb': round(size_mb, 2),
            'size_gb': round(size_mb / 1024, 2)
        })
    except Exception as e:
        logger.error(f"Storage info error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get-qualities', methods=['POST'])
def api_get_qualities():
    """الحصول على الجودات المتاحة للفيديو - نسخة محسّنة واحترافية"""
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        platform = downloader.detect_platform(url)
        logger.info(f"Platform detected: {platform} for URL: {url}")
        
        # استخدام get_available_formats الجديدة
        result = downloader.get_available_formats(url)
        
        if result.get('success'):
            formats_data = result.get('formats', {})
            
            # إضافة معلومات إضافية للتصحيح
            debug_info = {
                'max_height': formats_data.get('max_height', 0),
                'all_heights': formats_data.get('all_heights', []),
                'video_formats_count': len(formats_data.get('video_audio', [])),
                'video_only_count': len(formats_data.get('video_only', []))
            }
            
            return jsonify({
                'success': True,
                'formats': formats_data,
                'info': result.get('info', {}),
                'platform': result.get('platform', 'unknown'),
                'debug': debug_info  # معلومات للتصحيح
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('error', 'فشل الحصول على التنسيقات')
            }), 500
    
    except Exception as e:
        logger.error(f"Get qualities error: {e}")
        return jsonify({
            'success': False, 
            'message': f'خطأ في فحص الجودات: {str(e)}',
            'qualities': [
                {'id': 'best', 'label': '⭐ أفضل جودة', 'ext': 'mp4'},
                {'id': 'medium', 'label': 'HD (720p)', 'ext': 'mp4'},
                {'id': 'low', 'label': 'SD (480p)', 'ext': 'mp4'},
                {'id': 'audio', 'label': 'صوت فقط', 'ext': 'mp3'}
            ]
        }), 500


@app.route('/api/cleanup', methods=['POST'])
@app.route('/api/cleanup-files', methods=['POST'])
def api_cleanup_files():
    """حذف الملفات"""
    try:
        data = request.json or {}
        cleanup_type = data.get('type', 'all')
        
        deleted_count = 0
        deleted_size = 0
        
        folders_to_clean = []
        
        if cleanup_type == 'all':
            folders_to_clean = [
                app.config['DOWNLOAD_FOLDER'],
                app.config['UPLOAD_FOLDER'],
                app.config['OUTPUT_FOLDER'],
                app.config['SUBTITLE_FOLDER']
            ]
        elif cleanup_type == 'downloads':
            folders_to_clean = [app.config['DOWNLOAD_FOLDER']]
        elif cleanup_type == 'uploads':
            folders_to_clean = [app.config['UPLOAD_FOLDER']]
        elif cleanup_type == 'outputs':
            folders_to_clean = [app.config['OUTPUT_FOLDER']]
        elif cleanup_type == 'subtitles':
            folders_to_clean = [app.config['SUBTITLE_FOLDER']]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_count += 1
                            deleted_size += file_size
                        except Exception as e:
                            logger.warning(f"Could not delete {file_path}: {e}")
        
        deleted_mb = deleted_size / (1024 * 1024)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'deleted_size_mb': round(deleted_mb, 2)
        })
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    """تحميل الملفات"""
    # معالجة المسارات المشفرة وإزالة "downloads/" من البداية
    filename = filename.replace('%2F', '/').replace('%5C', '\\')
    
    # إزالة "downloads/" من البداية إذا كان موجوداً
    if filename.startswith('downloads/'):
        filename = filename.replace('downloads/', '', 1)
    if filename.startswith('downloads\\'):
        filename = filename.replace('downloads\\', '', 1)
    
    folders = [
        app.config['OUTPUT_FOLDER'],
        app.config['DOWNLOAD_FOLDER'],
        app.config['SUBTITLE_FOLDER']
    ]
    
    # محاولة البحث في المجلدات
    for folder in folders:
        # محاولة بالمسار الكامل
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        
        # محاولة بالاسم فقط
        basename = os.path.basename(filename)
        filepath = os.path.join(folder, basename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
    
    # البحث في جميع الملفات
    for folder in folders:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file == os.path.basename(filename) or file == filename:
                        return send_file(os.path.join(root, file), as_attachment=True)
    
    logger.error(f"File not found: {filename}")
    return "File not found", 404


@app.route('/download/subtitle/<filename>')
def download_subtitle(filename):
    """تحميل ملفات الترجمة"""
    filepath = os.path.join(app.config['SUBTITLE_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 التطبيق المتكامل للترجمة والتحميل v5.0")
    print("="*60)
    print(f"✅ Whisper: {'متوفر' if WHISPER_AVAILABLE else 'غير متوفر'}")
    print(f"✅ Faster Whisper: {'متوفر (سيتم استخدامه - أسرع بـ 4-5x)' if FASTER_WHISPER_AVAILABLE else 'غير متوفر'}")
    print(f"✅ المترجم: {'متوفر' if TRANSLATOR_AVAILABLE else 'غير متوفر'}")
    print(f"✅ FFmpeg: {'متوفر' if VideoProcessor.check_ffmpeg() else 'غير متوفر'}")
    print("\n🌐 الخادم يعمل على: http://localhost:5000")
    print("\n🛑 لإيقاف الخادم: CTRL+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
