#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
التطبيق المتكامل المحسن v4.0
- دعم محسن لجميع المنصات
- صفحة مخصصة لدمج الترجمة
- تحكم كامل في خصائص الترجمة
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import yt_dlp

# Try to import optional libraries
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

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SUBTITLE_FOLDER'] = 'subtitles'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create necessary folders
for folder in ['uploads', 'downloads', 'outputs', 'subtitles', 'templates', 'static']:
    Path(folder).mkdir(exist_ok=True)

# Platform-specific download configurations
PLATFORM_CONFIGS = {
    'tiktok': {
        'cookies_file': 'tiktok_cookies.txt',
        'user_agents': [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)',
            'TikTok 26.1.3 rv:261303 (iPhone; iOS 14.6; en_US)',
        ],
        'headers': {
            'Referer': 'https://www.tiktok.com/',
            'Origin': 'https://www.tiktok.com'
        },
        'extractors': ['TikTok', 'TikTokVM', 'TikTokLive'],
        'methods': ['yt-dlp', 'api', 'mobile', 'web_scrape']
    },
    'instagram': {
        'cookies_file': 'instagram_cookies.txt',
        'login_required': True,
        'headers': {
            'User-Agent': 'Instagram 219.0.0.12.117'
        }
    },
    'twitter': {
        'cookies_file': 'twitter_cookies.txt',
        'extractors': ['Twitter', 'TwitterSpaces']
    },
    'youtube': {
        'extractors': ['Youtube', 'YoutubePlaylist', 'YoutubeShorts']
    },
    'facebook': {
        'cookies_file': 'facebook_cookies.txt',
        'extractors': ['Facebook', 'FacebookReel']
    }
}

class EnhancedDownloader:
    """محمل محسن يدعم طرق متعددة للتحميل"""
    
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
        """الحصول على إعدادات yt-dlp المناسبة للمنصة"""
        
        # Base options
        opts = {
            'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'no_color': True,
            'geo_bypass': True,
            'socket_timeout': 30,
            'retries': 10,
            'fragment_retries': 10,
            'concurrent_fragment_downloads': 5
        }
        
        # Quality settings
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
        
        # Platform-specific settings
        if platform in PLATFORM_CONFIGS:
            config = PLATFORM_CONFIGS[platform]
            
            # Add cookies if available
            cookies_file = config.get('cookies_file')
            if cookies_file and os.path.exists(cookies_file):
                opts['cookiefile'] = cookies_file
            
            # Add headers
            if 'headers' in config:
                opts['http_headers'] = config['headers']
            
            # TikTok specific
            if platform == 'tiktok':
                opts.update({
                    'http_headers': {
                        'User-Agent': config['user_agents'][0],
                        'Referer': 'https://www.tiktok.com/',
                    },
                    'extractor_args': {
                        'tiktok': {
                            'api_hostname': 'api16-normal-c-useast1a.tiktokv.com',
                            'app_version': '26.1.3',
                        }
                    }
                })
        
        return opts
    
    def download_with_ytdlp(self, url: str, quality: str = 'best') -> Dict:
        """التحميل باستخدام yt-dlp"""
        platform = self.detect_platform(url)
        ydl_opts = self.get_ydl_opts(platform, quality)
        
        result = {
            'success': False,
            'message': '',
            'file': None,
            'info': {}
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    raise Exception("لم يتم العثور على معلومات الفيديو")
                
                # Download the video
                ydl.download([url])
                
                # Get the filename
                filename = ydl.prepare_filename(info)
                if quality == 'audio':
                    filename = filename.rsplit('.', 1)[0] + '.mp3'
                
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
            
            # Try alternative method for TikTok
            if platform == 'tiktok':
                result = self.download_tiktok_alternative(url, quality)
        
        return result
    
    def download_tiktok_alternative(self, url: str, quality: str) -> Dict:
        """طريقة بديلة لتحميل TikTok"""
        result = {
            'success': False,
            'message': '',
            'file': None,
            'info': {}
        }
        
        try:
            # Method 1: Using TikTokApi (if available)
            # This would require additional setup
            
            # Method 2: Using requests with proper headers
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # Try different TikTok download services as fallback
            services = [
                f"https://ssstik.io/api/download?url={url}",
                f"https://ttdownloader.com/dl.php?v={url}",
                f"https://musicaldown.com/download?url={url}"
            ]
            
            for service_url in services:
                try:
                    response = requests.get(service_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        # Parse the response and download the video
                        # Implementation depends on the service response format
                        pass
                except:
                    continue
            
            # Method 3: Using gallery-dl or youtube-dl fork
            alternative_opts = {
                'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
                'format': 'best',
                'http_headers': {
                    'User-Agent': 'TikTok 26.1.3 rv:261303 (iPhone; iOS 14.6; en_US)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                }
            }
            
            with yt_dlp.YoutubeDL(alternative_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                result['success'] = True
                result['message'] = 'تم التحميل بطريقة بديلة'
                result['file'] = filename
                result['info'] = {
                    'title': info.get('title', 'TikTok Video'),
                    'platform': 'tiktok'
                }
                
        except Exception as e:
            result['message'] = f'فشلت جميع طرق التحميل: {str(e)}'
            logger.error(f"Alternative download failed: {e}")
        
        return result
    
    def get_available_qualities(self, url: str) -> List[Dict]:
        """الحصول على قائمة الجودات المتاحة"""
        platform = self.detect_platform(url)
        ydl_opts = self.get_ydl_opts(platform, 'best')
        ydl_opts['listformats'] = True
        
        qualities = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                formats = info.get('formats', [])
                seen = set()
                
                for f in formats:
                    if f.get('height'):
                        quality_label = f"{f['height']}p"
                        if quality_label not in seen:
                            qualities.append({
                                'id': f['format_id'],
                                'label': quality_label,
                                'ext': f.get('ext', 'mp4'),
                                'filesize': f.get('filesize', 0)
                            })
                            seen.add(quality_label)
                
                # Add audio option
                qualities.append({
                    'id': 'audio',
                    'label': 'صوت فقط (MP3)',
                    'ext': 'mp3'
                })
                
        except Exception as e:
            logger.error(f"Error getting qualities: {e}")
            # Return default qualities
            qualities = [
                {'id': 'best', 'label': 'أفضل جودة', 'ext': 'mp4'},
                {'id': 'medium', 'label': 'جودة متوسطة (720p)', 'ext': 'mp4'},
                {'id': 'low', 'label': 'جودة منخفضة (480p)', 'ext': 'mp4'},
                {'id': 'audio', 'label': 'صوت فقط (MP3)', 'ext': 'mp3'}
            ]
        
        return qualities

class SubtitleProcessor:
    """معالج الترجمة المتقدم"""
    
    @staticmethod
    def create_srt(text: str, duration: int = None) -> str:
        """إنشاء ملف SRT من النص"""
        lines = text.split('\n')
        srt_content = []
        
        # تقسيم النص إلى أجزاء مناسبة
        segments = []
        current_segment = []
        char_count = 0
        
        for line in lines:
            words = line.split()
            for word in words:
                current_segment.append(word)
                char_count += len(word) + 1
                
                # إنشاء جزء جديد كل 40 حرف تقريباً
                if char_count >= 40 or word.endswith('.') or word.endswith('!') or word.endswith('?'):
                    segments.append(' '.join(current_segment))
                    current_segment = []
                    char_count = 0
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        # حساب التوقيت لكل جزء
        if not segments:
            return ""
        
        segment_duration = (duration or 60) / len(segments) if duration else 3
        
        for i, segment in enumerate(segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            start_str = SubtitleProcessor.seconds_to_srt_time(start_time)
            end_str = SubtitleProcessor.seconds_to_srt_time(end_time)
            
            srt_content.append(f"{i + 1}")
            srt_content.append(f"{start_str} --> {end_str}")
            srt_content.append(segment)
            srt_content.append("")
        
        return '\n'.join(srt_content)
    
    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        """تحويل الثواني إلى تنسيق SRT للوقت"""
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
        font_name: str = "Arial"
    ) -> str:
        """إنشاء ملف ASS متقدم للترجمة مع التنسيق"""
        
        # Convert colors from hex to ASS format
        ass_font_color = f"&H00{font_color[-2:]}{font_color[2:4]}{font_color[:2]}"
        ass_bg_color = f"&H{hex(bg_opacity)[2:].upper():0>2}{bg_color[-2:]}{bg_color[2:4]}{bg_color[:2]}"
        
        # Determine alignment based on position
        alignment = {
            'top': '8',
            'center': '5',
            'bottom': '2'
        }.get(position, '2')
        
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
Style: Default,{font_name},{font_size},{ass_font_color},&H000000FF,&H00000000,{ass_bg_color},0,0,0,0,100,100,0,0,3,2,1,{alignment},10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Parse SRT and convert to ASS events
        events = []
        lines = srt_content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            if lines[i].strip().isdigit():
                # Found subtitle number
                if i + 2 < len(lines):
                    timing = lines[i + 1].strip()
                    text = lines[i + 2].strip()
                    
                    # Parse timing
                    if ' --> ' in timing:
                        start, end = timing.split(' --> ')
                        start = start.replace(',', '.').strip()
                        end = end.replace(',', '.').strip()
                        
                        # Convert to ASS format
                        start_ass = SubtitleProcessor.srt_time_to_ass(start)
                        end_ass = SubtitleProcessor.srt_time_to_ass(end)
                        
                        events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
                    
                    i += 4  # Skip to next subtitle
                else:
                    i += 1
            else:
                i += 1
        
        return ass_header + '\n'.join(events)
    
    @staticmethod
    def srt_time_to_ass(srt_time: str) -> str:
        """تحويل وقت SRT إلى تنسيق ASS"""
        # SRT: 00:00:00,000
        # ASS: 0:00:00.00
        parts = srt_time.replace(',', '.').split(':')
        if len(parts) >= 3:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
            return f"{h}:{m:02d}:{s:05.2f}"
        return "0:00:00.00"

class VideoProcessor:
    """معالج الفيديو لدمج الترجمة"""
    
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
            # Check if ffmpeg is available
            if not VideoProcessor.check_ffmpeg():
                raise Exception("ffmpeg غير متوفر. يرجى تثبيته أولاً")
            
            # Prepare subtitle filter
            subtitle_filter = VideoProcessor.create_subtitle_filter(subtitle_path, subtitle_settings)
            
            # Prepare quality settings
            quality_settings = VideoProcessor.get_quality_settings(quality)
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', subtitle_filter,
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-metadata:s:s:0', 'language=ara',  # Set subtitle language to Arabic
                '-sub_charenc', 'UTF-8'  # Force UTF-8 encoding
            ]
            
            # Add quality settings
            cmd.extend(quality_settings)
            
            # Add output
            cmd.extend([
                '-y',  # Overwrite output
                output_path
            ])
            
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Execute ffmpeg
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if process.returncode == 0:
                result['success'] = True
                result['message'] = 'تم دمج الترجمة بنجاح'
                result['output_file'] = output_path
            else:
                raise Exception(f"FFmpeg error: {process.stderr}")
            
        except subprocess.TimeoutExpired:
            result['message'] = 'انتهت مهلة المعالجة. الملف كبير جداً'
        except Exception as e:
            result['message'] = f'خطأ في دمج الترجمة: {str(e)}'
            logger.error(f"Merge error: {e}")
        
        return result
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """التحقق من توفر ffmpeg"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except:
            return False
    
    @staticmethod
    def create_subtitle_filter(subtitle_path: str, settings: Dict = None) -> str:
        """إنشاء فلتر الترجمة لـ ffmpeg"""
        
        if not settings:
            settings = {}
        
        # Use ASS subtitle for advanced styling
        if subtitle_path.endswith('.ass'):
            # For ASS files, use ass filter
            return f"ass='{subtitle_path}'"
        
        # For SRT files, use subtitles filter with styling
        filter_parts = [f"subtitles='{subtitle_path}'"]
        
        # Add styling options
        style_options = []
        
        # Font settings - Use Arabic-compatible fonts
        font_name = settings.get('font_name', 'Arial')
        # Map to Arabic-compatible fonts
        arabic_fonts = {
            'Arial': 'Arial',
            'Tahoma': 'Tahoma', 
            'Helvetica': 'Arial',
            'Times New Roman': 'Traditional Arabic'
        }
        font_name = arabic_fonts.get(font_name, 'Arial')
        style_options.append(f"FontName={font_name}")
        
        if settings.get('font_size'):
            style_options.append(f"FontSize={settings['font_size']}")
        
        # Colors (need to be in ASS format)
        if settings.get('font_color'):
            color = settings['font_color'].replace('#', '')
            style_options.append(f"PrimaryColour=&H00{color[-2:]}{color[2:4]}{color[:2]}")
        
        if settings.get('bg_color'):
            bg_color = settings['bg_color'].replace('#', '')
            opacity = settings.get('bg_opacity', 128)
            style_options.append(f"BackColour=&H{hex(opacity)[2:]:0>2}{bg_color[-2:]}{bg_color[2:4]}{bg_color[:2]}")
        
        # Position
        alignment = {
            'top': '8',
            'center': '5',
            'bottom': '2'
        }.get(settings.get('position', 'bottom'), '2')
        style_options.append(f"Alignment={alignment}")
        
        if style_options:
            filter_parts.append(f"force_style='{','.join(style_options)}'")
        
        return ':'.join(filter_parts)
    
    @staticmethod
    def get_quality_settings(quality: str) -> List[str]:
        """الحصول على إعدادات الجودة لـ ffmpeg"""
        
        if quality == 'original':
            return ['-c:v', 'copy']  # No re-encoding
        elif quality == 'high':
            return [
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p'
            ]
        elif quality == 'medium':
            return [
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p'
            ]
        elif quality == 'low':
            return [
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '28',
                '-profile:v', 'baseline',
                '-level', '3.1',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=w=min(iw\\,1280):h=-2'  # Max width 1280
            ]
        else:
            return ['-c:v', 'libx264', '-crf', '23']

# Initialize downloader
downloader = EnhancedDownloader()

# Routes
@app.route('/')
def index():
    """الصفحة الرئيسية الموحدة - كل شيء في صفحة واحدة"""
    return render_template('index.html', 
                         whisper_available=WHISPER_AVAILABLE,
                         translator_available=TRANSLATOR_AVAILABLE)

@app.route('/api/get-video-thumbnail', methods=['POST'])
def api_get_video_thumbnail():
    """استخراج صورة مصغرة من الفيديو للمعاينة"""
    try:
        data = request.json
        # Get video file path
        video_file = data.get('video_file') or session.get('video_file')
        
        # Handle relative paths
        if video_file and not os.path.isabs(video_file):
            # Try different possible locations
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
        
        # Check if ffmpeg is available
        if not VideoProcessor.check_ffmpeg():
            return jsonify({'success': False, 'message': 'ffmpeg غير متوفر'}), 503
        
        # Generate thumbnail filename
        thumbnail_filename = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], thumbnail_filename)
        
        # Extract thumbnail using ffmpeg (at 1 second or middle of video)
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-ss', '00:00:01',  # At 1 second
            '-vframes', '1',
            '-vf', 'scale=640:-1',  # Scale to 640px width, maintain aspect ratio
            '-y',
            thumbnail_path
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if process.returncode == 0 and os.path.exists(thumbnail_path):
            return jsonify({
                'success': True,
                'thumbnail_url': f'/download/{thumbnail_filename}',
                'thumbnail_path': thumbnail_path
            })
        else:
            return jsonify({'success': False, 'message': 'فشل استخراج الصورة المصغرة'}), 500
            
    except Exception as e:
        logger.error(f"Thumbnail extraction error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/instant-translate', methods=['POST'])
def api_instant_translate():
    """API للترجمة الفورية - معالجة الخطوات المتعددة"""
    try:
        data = request.json
        step = data.get('step')
        
        if step == 'download':
            # Step 1: Download video
            url = data.get('url')
            quality = data.get('quality', '720p')
            
            if not url:
                return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
            
            # Map quality to yt-dlp format
            quality_map = {
                'best': 'best',
                '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]'
            }
            
            result = downloader.download_with_ytdlp(url, quality_map.get(quality, '720p'))
            
            if result['success']:
                # Store file path in session or temp storage
                session['video_file'] = result['file']
                # Return full path for thumbnail extraction
                full_path = result['file'] if os.path.isabs(result['file']) else os.path.join(app.config['DOWNLOAD_FOLDER'], result['file'])
                return jsonify({
                    'success': True,
                    'file': full_path,
                    'info': result['info']
                })
            else:
                return jsonify({'success': False, 'message': result['message']}), 400
                
        elif step == 'extract_audio':
            # Step 2: Extract audio from video
            video_file = data.get('video_file') or session.get('video_file')
            
            if not video_file or not os.path.exists(video_file):
                return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
            
            # Extract audio using ffmpeg
            audio_file = video_file.rsplit('.', 1)[0] + '_audio.wav'
            
            cmd = [
                'ffmpeg', '-i', video_file,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV format for Whisper
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                audio_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                session['audio_file'] = audio_file
                return jsonify({
                    'success': True,
                    'audio_file': audio_file
                })
            except subprocess.CalledProcessError as e:
                return jsonify({'success': False, 'message': f'خطأ في استخراج الصوت: {e}'}), 500
                
        elif step == 'transcribe':
            # Step 3: Transcribe audio
            if not WHISPER_AVAILABLE:
                return jsonify({'success': False, 'message': 'Whisper غير متوفر'}), 503
            
            audio_file = data.get('audio_file') or session.get('audio_file')
            model_size = data.get('model', 'base')
            language = data.get('language', 'auto')
            
            if not audio_file or not os.path.exists(audio_file):
                return jsonify({'success': False, 'message': 'ملف الصوت غير موجود'}), 400
            
            # Load Whisper model
            model = whisper.load_model(model_size)
            
            # Transcribe
            options = {
                'language': None if language == 'auto' else language,
                'task': 'transcribe',
                'fp16': False
            }
            
            result = model.transcribe(audio_file, **options)
            
            session['transcript'] = result['text']
            session['source_language'] = result.get('language', language)
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'language': result.get('language', language)
            })
            
        elif step == 'translate':
            # Step 4: Translate to Arabic
            if not TRANSLATOR_AVAILABLE:
                return jsonify({'success': False, 'message': 'المترجم غير متوفر'}), 503
            
            text = data.get('text') or session.get('transcript')
            source_lang = data.get('source_lang', 'auto')
            
            if not text:
                return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
            
            translator = GoogleTranslator(source=source_lang, target='ar')
            translated = translator.translate(text)
            
            session['translated_text'] = translated
            
            return jsonify({
                'success': True,
                'translated_text': translated
            })
            
        elif step == 'merge':
            # Step 5: Create subtitle and merge with video
            video_file = data.get('video_file') or session.get('video_file')
            subtitle_text = data.get('subtitle_text') or session.get('translated_text')
            settings = data.get('settings', {})
            
            if not video_file or not os.path.exists(video_file):
                return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
            
            if not subtitle_text:
                return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
            
            # Create SRT file
            srt_content = SubtitleProcessor.create_srt(subtitle_text)
            srt_filename = f"instant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # Create ASS file with styling
            ass_content = SubtitleProcessor.create_ass_subtitle(
                srt_content,
                font_size=int(settings.get('font_size', 22)),
                font_color=settings.get('font_color', '#FFFFFF').replace('#', ''),
                bg_color='000000',
                bg_opacity=180,
                position=settings.get('position', 'bottom'),
                font_name=settings.get('font_name', 'Arial')
            )
            
            ass_path = srt_path.replace('.srt', '.ass')
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            # Output filename
            output_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Merge subtitle with video
            result = VideoProcessor.merge_subtitles(
                video_file,
                ass_path,
                output_path,
                quality='medium',
                subtitle_settings=settings
            )
            
            if result['success']:
                # Clean up temporary files
                try:
                    if 'audio_file' in session and os.path.exists(session['audio_file']):
                        os.remove(session['audio_file'])
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
            return jsonify({'success': False, 'message': 'خطوة غير صحيحة'}), 400
            
    except Exception as e:
        logger.error(f"Instant translate error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/smart-translate', methods=['POST'])
def api_smart_translate():
    """الترجمة الذكية - تحميل وترجمة ودمج بضغطة واحدة"""
    try:
        data = request.json
        url = data.get('url')
        target_lang = data.get('target_lang', 'ar')
        whisper_model = data.get('whisper_model', 'base')
        subtitle_settings = data.get('subtitle_settings', {
            'font_size': 24,
            'font_color': '#FFFFFF',
            'bg_color': '#000000',
            'bg_opacity': 180,
            'position': 'bottom',
            'font_name': 'Arial'
        })
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        result = {
            'success': False,
            'steps': [],
            'files': {}
        }
        
        # Step 1: Download video
        logger.info("Step 1: Downloading video...")
        result['steps'].append({'step': 'download', 'status': 'processing'})
        
        download_result = downloader.download_with_ytdlp(url, 'best')
        if not download_result['success']:
            result['steps'][-1]['status'] = 'failed'
            result['message'] = f"فشل التحميل: {download_result['message']}"
            return jsonify(result), 500
        
        video_path = download_result['file']
        result['steps'][-1]['status'] = 'completed'
        result['files']['video'] = video_path
        
        # Step 2: Extract audio and transcribe
        if not WHISPER_AVAILABLE:
            result['message'] = "Whisper غير متوفر للتحويل إلى نص"
            return jsonify(result), 503
        
        logger.info("Step 2: Transcribing audio...")
        result['steps'].append({'step': 'transcribe', 'status': 'processing'})
        
        try:
            import whisper
            model = whisper.load_model(whisper_model)
            
            # Transcribe directly from video
            transcribe_result = model.transcribe(video_path, language=None, task='transcribe')
            original_text = transcribe_result['text']
            detected_language = transcribe_result.get('language', 'en')
            
            result['steps'][-1]['status'] = 'completed'
            result['files']['original_text'] = original_text
            
        except Exception as e:
            result['steps'][-1]['status'] = 'failed'
            result['message'] = f"فشل التحويل إلى نص: {str(e)}"
            return jsonify(result), 500
        
        # Step 3: Translate to target language
        if not TRANSLATOR_AVAILABLE:
            result['message'] = "المترجم غير متوفر"
            return jsonify(result), 503
        
        logger.info("Step 3: Translating text...")
        result['steps'].append({'step': 'translate', 'status': 'processing'})
        
        try:
            from deep_translator import GoogleTranslator
            
            # Split text into chunks if too long
            max_chars = 4500
            text_chunks = [original_text[i:i+max_chars] 
                          for i in range(0, len(original_text), max_chars)]
            
            translated_chunks = []
            for chunk in text_chunks:
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            
            translated_text = ' '.join(translated_chunks)
            
            result['steps'][-1]['status'] = 'completed'
            result['files']['translated_text'] = translated_text
            
        except Exception as e:
            result['steps'][-1]['status'] = 'failed'
            result['message'] = f"فشلت الترجمة: {str(e)}"
            return jsonify(result), 500
        
        # Step 4: Create subtitle file
        logger.info("Step 4: Creating subtitle file...")
        result['steps'].append({'step': 'subtitle', 'status': 'processing'})
        
        try:
            # Get video duration
            import subprocess
            duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                          'format=duration', '-of', 
                          'default=noprint_wrappers=1:nokey=1', video_path]
            
            try:
                duration = float(subprocess.check_output(duration_cmd).decode().strip())
            except:
                duration = 60  # Default duration
            
            # Create SRT content
            srt_content = SubtitleProcessor.create_srt(translated_text, duration)
            
            # Create ASS with styling
            ass_content = SubtitleProcessor.create_ass_subtitle(
                srt_content,
                font_size=subtitle_settings.get('font_size', 24),
                font_color=subtitle_settings.get('font_color', 'FFFFFF').replace('#', ''),
                bg_color=subtitle_settings.get('bg_color', '000000').replace('#', ''),
                bg_opacity=subtitle_settings.get('bg_opacity', 180),
                position=subtitle_settings.get('position', 'bottom'),
                font_name=subtitle_settings.get('font_name', 'Arial')
            )
            
            # Save subtitle file
            subtitle_filename = f"auto_subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ass"
            subtitle_path = os.path.join(app.config['SUBTITLE_FOLDER'], subtitle_filename)
            
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            result['steps'][-1]['status'] = 'completed'
            result['files']['subtitle'] = subtitle_path
            
        except Exception as e:
            result['steps'][-1]['status'] = 'failed'
            result['message'] = f"فشل إنشاء الترجمة: {str(e)}"
            return jsonify(result), 500
        
        # Step 5: Merge subtitle with video
        logger.info("Step 5: Merging subtitle with video...")
        result['steps'].append({'step': 'merge', 'status': 'processing'})
        
        try:
            output_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            merge_result = VideoProcessor.merge_subtitles(
                video_path,
                subtitle_path,
                output_path,
                quality='high',
                subtitle_settings=subtitle_settings
            )
            
            if merge_result['success']:
                result['steps'][-1]['status'] = 'completed'
                result['success'] = True
                result['files']['output'] = output_filename
                result['download_url'] = f'/download/{output_filename}'
                result['message'] = 'تمت الترجمة والدمج بنجاح!'
                
                # Clean up temporary files
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except:
                    pass
                
            else:
                result['steps'][-1]['status'] = 'failed'
                result['message'] = merge_result['message']
                
        except Exception as e:
            result['steps'][-1]['status'] = 'failed'
            result['message'] = f"فشل دمج الترجمة: {str(e)}"
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Smart translate error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def api_download():
    """API للتحميل من المنصات"""
    try:
        data = request.json
        url = data.get('url')
        quality = data.get('quality', 'best')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        # Download video
        result = downloader.download_with_ytdlp(url, quality)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Download API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/get-qualities', methods=['POST'])
def api_get_qualities():
    """الحصول على الجودات المتاحة للرابط"""
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        qualities = downloader.get_available_qualities(url)
        
        return jsonify({
            'success': True,
            'qualities': qualities
        })
        
    except Exception as e:
        logger.error(f"Get qualities error: {e}")
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
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load Whisper model
        model_size = request.form.get('model', 'base')
        model = whisper.load_model(model_size)
        
        # Transcribe
        options = {
            'language': None if language == 'auto' else language,
            'task': 'transcribe',
            'fp16': False
        }
        
        result = model.transcribe(filepath, **options)
        
        # Create SRT file
        srt_content = SubtitleProcessor.create_srt(result['text'])
        srt_filename = f"{os.path.splitext(filename)[0]}.srt"
        srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        # Clean up uploaded file
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
        # Get files
        if 'video' not in request.files or 'subtitle' not in request.files:
            return jsonify({'success': False, 'message': 'الرجاء رفع الفيديو والترجمة'}), 400
        
        video_file = request.files['video']
        subtitle_file = request.files['subtitle']
        
        # Get settings
        settings = {
            'font_size': int(request.form.get('font_size', 20)),
            'font_color': request.form.get('font_color', '#FFFFFF'),
            'bg_color': request.form.get('bg_color', '#000000'),
            'bg_opacity': int(request.form.get('bg_opacity', 128)),
            'position': request.form.get('position', 'bottom'),
            'font_name': request.form.get('font_name', 'Arial'),
            'quality': request.form.get('quality', 'medium')
        }
        
        # Save files
        video_filename = secure_filename(video_file.filename)
        subtitle_filename = secure_filename(subtitle_file.filename)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        subtitle_path = os.path.join(app.config['SUBTITLE_FOLDER'], subtitle_filename)
        
        video_file.save(video_path)
        subtitle_file.save(subtitle_path)
        
        # Convert SRT to ASS if needed for advanced styling
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
        
        # Output filename
        output_filename = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Merge subtitle with video
        result = VideoProcessor.merge_subtitles(
            video_path,
            subtitle_path,
            output_path,
            quality=settings['quality'],
            subtitle_settings=settings
        )
        
        # Clean up temporary files
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

@app.route('/api/create-subtitle', methods=['POST'])
def api_create_subtitle():
    """إنشاء ملف ترجمة من النص"""
    try:
        data = request.json
        text = data.get('text')
        format_type = data.get('format', 'srt')
        duration = data.get('duration', 60)
        
        if not text:
            return jsonify({'success': False, 'message': 'لا يوجد نص'}), 400
        
        # Create subtitle file
        if format_type == 'srt':
            subtitle_content = SubtitleProcessor.create_srt(text, duration)
            filename = f"subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        else:
            # Create ASS with default styling
            srt_content = SubtitleProcessor.create_srt(text, duration)
            subtitle_content = SubtitleProcessor.create_ass_subtitle(srt_content)
            filename = f"subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ass"
        
        filepath = os.path.join(app.config['SUBTITLE_FOLDER'], filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(subtitle_content)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/download/subtitle/{filename}'
        })
        
    except Exception as e:
        logger.error(f"Create subtitle error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """تحميل الملفات"""
    # Check in different folders
    folders = [
        app.config['OUTPUT_FOLDER'],
        app.config['DOWNLOAD_FOLDER'],
        app.config['SUBTITLE_FOLDER']
    ]
    
    for folder in folders:
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
    
    return "File not found", 404

@app.route('/download/subtitle/<filename>')
def download_subtitle(filename):
    """تحميل ملفات الترجمة"""
    filepath = os.path.join(app.config['SUBTITLE_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 التطبيق المتكامل المحسن v4.0")
    print("="*60)
    print(f"✅ Whisper: {'متوفر' if WHISPER_AVAILABLE else 'غير متوفر'}")
    print(f"✅ المترجم: {'متوفر' if TRANSLATOR_AVAILABLE else 'غير متوفر'}")
    print(f"✅ FFmpeg: {'متوفر' if VideoProcessor.check_ffmpeg() else 'غير متوفر'}")
    print("\n🌐 الخادم يعمل على: http://localhost:5000")
    print("📍 صفحة محرر الترجمة: http://localhost:5000/subtitle-editor")
    print("\n🛑 لإيقاف الخادم: CTRL+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
