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


class VideoDownloader:
    """محمل الفيديو من جميع المنصات"""
    
    def __init__(self):
        self.session_cache = {}
    
    def detect_platform(self, url: str) -> str:
        """الكشف عن المنصة من الرابط"""
        platforms = {
            'tiktok.com': 'tiktok',
            'vm.tiktok.com': 'tiktok',
            'instagram.com': 'instagram',
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'youtube.com': 'youtube',
            'youtu.be': 'youtube',
            'facebook.com': 'facebook',
            'fb.watch': 'facebook'
        }
        
        for domain, platform in platforms.items():
            if domain in url.lower():
                return platform
        return 'generic'
    
    def get_ydl_opts(self, platform: str, quality: str = 'best') -> dict:
        """الحصول على إعدادات yt-dlp"""
        opts = {
            'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            'concurrent_fragment_downloads': 16,
            'http_chunk_size': 10485760,
        }
        
        # إعدادات الجودة
        if quality == 'best':
            opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif quality == 'medium':
            opts['format'] = 'bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720][ext=mp4]/best'
        elif quality == 'low':
            opts['format'] = 'bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480][ext=mp4]/best'
        elif quality == 'audio':
            opts['format'] = 'bestaudio[ext=m4a]/bestaudio'
            opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        
        # إعدادات خاصة بمنصة TikTok
        if platform == 'tiktok':
            opts['http_headers'] = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
                'Referer': 'https://www.tiktok.com/',
            }
        
        return opts
    
    def download(self, url: str, quality: str = 'best') -> Dict:
        """تحميل الفيديو"""
        result = {
            'success': False,
            'message': '',
            'file': None,
            'info': {}
        }
        
        try:
            platform = self.detect_platform(url)
            ydl_opts = self.get_ydl_opts(platform, quality)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # استخراج المعلومات أولاً
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    raise Exception("لم يتم العثور على معلومات الفيديو")
                
                # التحقق من وجود الملف
                expected_filename = ydl.prepare_filename(info)
                if quality == 'audio':
                    expected_filename = expected_filename.rsplit('.', 1)[0] + '.mp3'
                
                expected_basename = os.path.basename(expected_filename)
                existing_file = os.path.join(app.config['DOWNLOAD_FOLDER'], expected_basename)
                
                if os.path.exists(existing_file):
                    logger.info(f"الملف موجود مسبقاً: {existing_file}")
                    filename = existing_file
                else:
                    # تحميل الفيديو
                    ydl.download([url])
                    filename = ydl.prepare_filename(info)
                    if quality == 'audio':
                        filename = filename.rsplit('.', 1)[0] + '.mp3'
                    
                    filename = os.path.normpath(filename)
                    
                    if not os.path.isabs(filename):
                        basename = os.path.basename(filename)
                        full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], basename)
                        if os.path.exists(full_path):
                            filename = full_path
                        else:
                            # البحث عن الملف الأحدث
                            download_folder = app.config['DOWNLOAD_FOLDER']
                            if os.path.exists(download_folder):
                                video_files = []
                                for file in os.listdir(download_folder):
                                    file_path = os.path.join(download_folder, file)
                                    if os.path.isfile(file_path) and file.endswith(('.mp4', '.webm', '.mkv', '.mp3', '.m4a')):
                                        video_files.append((file_path, os.path.getmtime(file_path)))
                                
                                if video_files:
                                    video_files.sort(key=lambda x: x[1], reverse=True)
                                    filename = video_files[0][0]
                
                if not os.path.exists(filename):
                    raise Exception(f"لم يتم العثور على الملف بعد التحميل: {filename}")
                
                result['success'] = True
                result['message'] = 'تم التحميل بنجاح'
                result['file'] = filename
                result['info'] = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'platform': platform,
                    'quality': quality
                }
        
        except Exception as e:
            result['message'] = f'خطأ في التحميل: {str(e)}'
            logger.error(f"Download error: {e}")
            logger.error(traceback.format_exc())
        
        return result


class SubtitleProcessor:
    """معالج الترجمة"""
    
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
        ass_font_color = f"&H00{font_color[-2:]}{font_color[2:4]}{font_color[:2]}"
        ass_bg_color = f"&H{hex(bg_opacity)[2:].upper():0>2}{bg_color[-2:]}{bg_color[2:4]}{bg_color[:2]}"
        
        alignment = {'top': '8', 'center': '5', 'bottom': '2'}.get(position, '2')
        
        if position == 'top':
            margin_v = max(10, 10 + vertical_offset)
        elif position == 'center':
            margin_v = 10
        else:
            margin_v = max(10, 10 - vertical_offset)
        
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
                        
                        effect = ''
                        if position == 'center' and vertical_offset != 0:
                            effect = f"\\pos(960,{540 + vertical_offset})"
                        
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
        
        subtitle_path_escaped = subtitle_path.replace("'", "'\\''")
        subtitle_path_escaped = f"'{subtitle_path_escaped}'"
        
        if subtitle_path.endswith('.ass'):
            return f"ass={subtitle_path_escaped}"
        
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
downloader = VideoDownloader()


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
        step = data.get('step')
        
        if not step:
            return jsonify({'success': False, 'message': 'خطوة غير محددة'}), 400
        
        logger.info(f"Processing step: {step}")
        
        if step == 'download':
            url = data.get('url')
            quality = data.get('quality', '720p')
            
            if not url:
                return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
            
            quality_map = {
                'best': 'best',
                '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]'
            }
            
            result = downloader.download(url, quality_map.get(quality, '720p'))
            
            if result['success']:
                video_file = result['file']
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
                
                # استخدام ملف مؤقت بدلاً من session لتقليل حجم cookie
                temp_file = os.path.join(app.config['DOWNLOAD_FOLDER'], f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(temp_file, 'w') as f:
                    f.write(video_file)
                
                return jsonify({
                    'success': True,
                    'file': video_file,
                    'info': result['info'],
                    'temp_file': os.path.basename(temp_file)
                })
            else:
                return jsonify({'success': False, 'message': result['message']}), 400
        
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
            
            model = whisper.load_model(model_size)
            
            import torch
            use_fp16 = torch.cuda.is_available()
            
            options = {
                'language': None if language == 'auto' else language,
                'task': 'transcribe',
                'fp16': use_fp16,
                'beam_size': 3,
                'best_of': 2,
                'temperature': 0.0,
                'word_timestamps': True
            }
            
            result = model.transcribe(audio_file, **options)
            
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
            subtitle_text = data.get('subtitle_text')
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
            
            if not subtitle_text:
                return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
            
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
            
            if whisper_segments and len(whisper_segments) > 0:
                segments_for_srt = []
                translator = GoogleTranslator(source=source_language, target='ar')
                
                for i, segment in enumerate(whisper_segments):
                    start_time = float(segment.get('start', 0))
                    end_time = float(segment.get('end', start_time + 3))
                    original_segment_text = segment.get('text', '').strip()
                    
                    if not original_segment_text:
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
                
                if segments_for_srt:
                    srt_content = SubtitleProcessor.create_srt(
                        subtitle_text,
                        duration=video_duration,
                        segments=segments_for_srt
                    )
                else:
                    srt_content = SubtitleProcessor.create_srt(
                        subtitle_text,
                        duration=video_duration
                    )
            else:
                srt_content = SubtitleProcessor.create_srt(
                    subtitle_text,
                    duration=video_duration
                )
            
            srt_filename = f"instant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            font_size = int(settings.get('font_size', 22))
            font_color = settings.get('font_color', '#FFFFFF')
            bg_color = settings.get('bg_color', '#000000')
            bg_opacity = int(settings.get('bg_opacity', 180))
            position = settings.get('position', 'bottom')
            font_name = settings.get('font_name', 'Arial')
            vertical_offset = int(settings.get('vertical_offset', 0))
            
            ass_content = SubtitleProcessor.create_ass_subtitle(
                srt_content,
                font_size=font_size,
                font_color=font_color.replace('#', ''),
                bg_color=bg_color.replace('#', ''),
                bg_opacity=bg_opacity,
                position=position,
                font_name=font_name,
                vertical_offset=vertical_offset
            )
            
            ass_path = srt_path.replace('.srt', '.ass')
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            output_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            subtitle_settings = {
                'font_name': font_name,
                'font_size': font_size,
                'font_color': font_color,
                'bg_color': bg_color,
                'bg_opacity': bg_opacity,
                'position': position
            }
            
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


@app.route('/api/download', methods=['POST'])
def api_download():
    """API للتحميل"""
    try:
        data = request.json
        url = data.get('url')
        quality = data.get('quality', 'best')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        result = downloader.download(url, quality)
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


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """تحويل الصوت/الفيديو إلى نص"""
    if not WHISPER_AVAILABLE:
        return jsonify({'success': False, 'message': 'Whisper غير متوفر'}), 503
    
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
        model = whisper.load_model(model_size)
        
        import torch
        use_fp16 = torch.cuda.is_available()
        
        options = {
            'language': None if language == 'auto' else language,
            'task': 'transcribe',
            'fp16': use_fp16,
            'beam_size': 3,
            'best_of': 2,
            'temperature': 0.0,
            'word_timestamps': True
        }
        
        result = model.transcribe(filepath, **options)
        
        srt_content = SubtitleProcessor.create_srt(result['text'])
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', language),
            'srt_file': srt_filename
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


@app.route('/api/cleanup-files', methods=['POST'])
def api_cleanup_files():
    """حذف الملفات"""
    try:
        data = request.json
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
    print(f"✅ المترجم: {'متوفر' if TRANSLATOR_AVAILABLE else 'غير متوفر'}")
    print(f"✅ FFmpeg: {'متوفر' if VideoProcessor.check_ffmpeg() else 'غير متوفر'}")
    print("\n🌐 الخادم يعمل على: http://localhost:5000")
    print("\n🛑 لإيقاف الخادم: CTRL+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
