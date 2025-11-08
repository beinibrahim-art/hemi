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
        
        # Base options - optimized for speed
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
            'retries': 3,  # Reduced retries for speed
            'fragment_retries': 3,
            'concurrent_fragment_downloads': 16,  # Increased for faster downloads
            'http_chunk_size': 10485760,  # 10MB chunks for faster downloads
            'no_check_certificate': True,
            'prefer_insecure': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],  # Faster clients
                    'skip': ['dash', 'hls']  # Skip some formats for speed
                }
            }
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
                
                # Check if file already exists
                expected_filename = ydl.prepare_filename(info)
                if quality == 'audio':
                    expected_filename = expected_filename.rsplit('.', 1)[0] + '.mp3'
                
                expected_basename = os.path.basename(expected_filename)
                existing_file = os.path.join(app.config['DOWNLOAD_FOLDER'], expected_basename)
                
                # If file exists, use it
                if os.path.exists(existing_file):
                    logger.info(f"File already exists: {existing_file}")
                    filename = existing_file
                else:
                    # Download the video
                    ydl.download([url])
                    
                    # Get the filename - yt-dlp might return relative or absolute path
                    filename = ydl.prepare_filename(info)
                    if quality == 'audio':
                        filename = filename.rsplit('.', 1)[0] + '.mp3'
                    
                    # Normalize the path
                    filename = os.path.normpath(filename)
                    
                    # If not absolute path, try to find it in download folder
                    if not os.path.isabs(filename):
                        # Try with basename in download folder
                        basename = os.path.basename(filename)
                        full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], basename)
                        if os.path.exists(full_path):
                            filename = full_path
                        else:
                            # Search for the file in download folder
                            download_folder = app.config['DOWNLOAD_FOLDER']
                            if os.path.exists(download_folder):
                                # Get most recently modified video file
                                video_files = []
                                for file in os.listdir(download_folder):
                                    file_path = os.path.join(download_folder, file)
                                    if os.path.isfile(file_path) and file.endswith(('.mp4', '.webm', '.mkv', '.mp3', '.m4a')):
                                        video_files.append((file_path, os.path.getmtime(file_path)))
                                
                                if video_files:
                                    # Sort by modification time, get most recent
                                    video_files.sort(key=lambda x: x[1], reverse=True)
                                    filename = video_files[0][0]
                                    logger.info(f"Found video file: {filename}")
                
                # Final verification
                if not os.path.exists(filename):
                    logger.warning(f"File not found at {filename}, searching download folder...")
                    download_folder = app.config['DOWNLOAD_FOLDER']
                    if os.path.exists(download_folder):
                        all_files = os.listdir(download_folder)
                        logger.info(f"Files in download folder: {all_files}")
                        # Try to find any video file
                        for file in all_files:
                            file_path = os.path.join(download_folder, file)
                            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in ['.mp4', '.webm', '.mkv', '.mp3', '.m4a']):
                                filename = file_path
                                logger.info(f"Using file: {filename}")
                                break
                
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
    def create_srt(text: str, duration: float = None, segments: List[Dict] = None) -> str:
        """إنشاء ملف SRT من النص مع توقيتات دقيقة"""
        srt_content = []
        
        # If segments with timing are provided, use them
        if segments:
            for i, segment in enumerate(segments):
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', start_time + 3))  # Use provided end_time, fallback to start+3
                text_segment = segment.get('text', '').strip()
                
                # Ensure end_time is valid
                if end_time <= start_time:
                    end_time = start_time + 3.0
                
                start_str = SubtitleProcessor.seconds_to_srt_time(start_time)
                end_str = SubtitleProcessor.seconds_to_srt_time(end_time)
                
                srt_content.append(f"{i + 1}")
                srt_content.append(f"{start_str} --> {end_str}")
                srt_content.append(text_segment)
                srt_content.append("")
            
            return '\n'.join(srt_content)
        
        # Otherwise, split text intelligently
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
                
                # Create new segment every ~35-45 characters or at punctuation
                if char_count >= 40 or word.endswith(('.', '!', '?', '،', '؛')):
                    segments_list.append(' '.join(current_segment))
                    current_segment = []
                    char_count = 0
        
        if current_segment:
            segments_list.append(' '.join(current_segment))
        
        if not segments_list:
            return ""
        
        # Calculate timing based on duration
        if duration and duration > 0:
            segment_duration = duration / len(segments_list)
            # Ensure minimum 2 seconds per segment
            segment_duration = max(segment_duration, 2.0)
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
        font_name: str = "Arial",
        vertical_offset: int = 0
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
        
        # Calculate MarginV based on position and vertical_offset
        # MarginV: distance from edge (top for top alignment, bottom for bottom alignment)
        if position == 'top':
            margin_v = max(10, 10 + vertical_offset)  # Distance from top
        elif position == 'center':
            margin_v = 10  # Center doesn't use MarginV, but we'll use it for offset
        else:  # bottom
            margin_v = max(10, 10 - vertical_offset)  # Distance from bottom
        
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
                        
                        # Add vertical offset effect if needed (for center position)
                        effect = ''
                        if position == 'center' and vertical_offset != 0:
                            # Use \pos for precise positioning
                            effect = f"\\pos(960,{540 + vertical_offset})"
                        
                        events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,{effect},{text}")
                    
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
            
            # Get video duration for better subtitle timing
            video_duration = VideoProcessor.get_video_duration(video_path)
            
            # Prepare subtitle filter
            subtitle_filter = VideoProcessor.create_subtitle_filter(subtitle_path, subtitle_settings)
            
            # Prepare quality settings
            # IMPORTANT: When using video filter (-vf), we cannot use -c:v copy
            # We must re-encode the video
            quality_settings = VideoProcessor.get_quality_settings(quality, use_filter=True)
            
            # Build ffmpeg command with optimized settings for speed
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', subtitle_filter,
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-threads', '0',  # Use all available CPU threads
                '-movflags', '+faststart',  # Optimize for web playback
            ]
            
            # Add quality settings (video encoding)
            cmd.extend(quality_settings)
            
            # Add output
            cmd.extend([
                '-y',  # Overwrite output
                output_path
            ])
            
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Execute ffmpeg with progress tracking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=600)  # 10 minutes timeout
                
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
            
        except subprocess.TimeoutExpired:
            result['message'] = 'انتهت مهلة المعالجة. الملف كبير جداً'
        except Exception as e:
            result['message'] = f'خطأ في دمج الترجمة: {str(e)}'
            logger.error(f"Merge error: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    @staticmethod
    def get_video_dimensions(video_path: str) -> Dict:
        """الحصول على مقاسات الفيديو الدقيقة"""
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
                import json
                data = json.loads(result.stdout)
                if 'streams' in data and len(data['streams']) > 0:
                    width = int(data['streams'][0].get('width', 0))
                    height = int(data['streams'][0].get('height', 0))
                    return {
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height if height > 0 else 1.0
                    }
        except Exception as e:
            logger.warning(f"Could not get video dimensions: {e}")
        return {'width': 1920, 'height': 1080, 'aspect_ratio': 16/9}
    
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
                duration = float(result.stdout.strip())
                return duration
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
        return None
    
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
        
        # Escape path properly for ffmpeg (handle spaces, special chars)
        # Use single quotes for paths with spaces
        subtitle_path_escaped = subtitle_path.replace("'", "'\\''")
        subtitle_path_escaped = f"'{subtitle_path_escaped}'"
        
        # Use ASS subtitle for advanced styling (preferred)
        if subtitle_path.endswith('.ass'):
            # For ASS files, use ass filter with proper escaping
            return f"ass={subtitle_path_escaped}"
        
        # For SRT files, convert to ASS first or use subtitles filter
        # Use subtitles filter with styling options
        filter_str = f"subtitles={subtitle_path_escaped}"
        
        # Add styling via force_style parameter
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
            style_options.append(f"FontSize={int(settings['font_size'])}")
        
        # Colors (need to be in ASS format: &HAABBGGRR)
        if settings.get('font_color'):
            color = settings['font_color'].replace('#', '')
            if len(color) == 6:
                # Convert RGB to BGR for ASS
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
                # Format: &HAABBGGRR where AA is opacity
                opacity_hex = format(opacity, '02X')
                style_options.append(f"BackColour=&H{opacity_hex}{b}{g}{r}")
        
        # Position
        alignment = {
            'top': '8',
            'center': '5',
            'bottom': '2'
        }.get(settings.get('position', 'bottom'), '2')
        style_options.append(f"Alignment={alignment}")
        
        if style_options:
            style_str = ','.join(style_options)
            filter_str = f"{filter_str}:force_style='{style_str}'"
        
        return filter_str
    
    @staticmethod
    def get_quality_settings(quality: str, use_filter: bool = False) -> List[str]:
        """الحصول على إعدادات الجودة لـ ffmpeg"""
        
        # If using filter (subtitle), we MUST re-encode, cannot use copy
        if quality == 'original' and not use_filter:
            return ['-c:v', 'copy']  # No re-encoding (only when no filter)
        elif quality == 'original' and use_filter:
            # When using filter, use fast preset for speed while maintaining quality
            return [
                '-c:v', 'libx264',
                '-preset', 'veryfast',  # Faster than 'slow' but still good quality
                '-crf', '20',  # Slightly higher than 18 for speed, still excellent quality
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'  # Use all CPU cores
            ]
        elif quality == 'high':
            return [
                '-c:v', 'libx264',
                '-preset', 'fast',  # Faster preset
                '-crf', '20',  # Good quality
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'
            ]
        elif quality == 'medium':
            return [
                '-c:v', 'libx264',
                '-preset', 'fast',  # Faster preset
                '-crf', '23',
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                '-threads', '0'
            ]
        elif quality == 'low':
            return [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Fastest preset
                '-crf', '28',
                '-profile:v', 'baseline',
                '-level', '3.1',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=w=min(iw\\,1280):h=-2',  # Max width 1280
                '-threads', '0'
            ]
        else:
            return ['-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-threads', '0']

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
    """استخراج صورة مصغرة من الفيديو للمعاينة مع معلومات المقاسات"""
    try:
        data = request.json
        # Get video file path
        video_file = data.get('video_file') or session.get('video_file')
        
        # Handle relative paths
        if video_file and not os.path.isabs(video_file):
            # Remove 'downloads/' prefix if present
            normalized = video_file.replace('downloads/', '').replace('downloads\\', '')
            possible_paths = [
                os.path.join(app.config['DOWNLOAD_FOLDER'], normalized),
                os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file)),
                os.path.join(app.config['UPLOAD_FOLDER'], normalized),
                os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_file)),
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
        
        # Check if ffmpeg is available
        if not VideoProcessor.check_ffmpeg():
            return jsonify({'success': False, 'message': 'ffmpeg غير متوفر'}), 503
        
        # Get video dimensions using ffprobe
        dimensions = VideoProcessor.get_video_dimensions(video_file)
        
        # Generate thumbnail filename
        thumbnail_filename = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], thumbnail_filename)
        
        # Extract thumbnail using ffmpeg (at 1 second or middle of video) with optimized settings
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-ss', '00:00:01',  # At 1 second
            '-vframes', '1',
            '-vf', 'scale=640:-1',  # Scale to 640px width, maintain aspect ratio
            '-threads', '0',  # Use all CPU threads
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
                'thumbnail_path': thumbnail_path,
                'dimensions': dimensions  # Return video dimensions
            })
        else:
            return jsonify({'success': False, 'message': 'فشل استخراج الصورة المصغرة'}), 500
            
    except Exception as e:
        logger.error(f"Thumbnail extraction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/instant-translate', methods=['POST'])
def api_instant_translate():
    """API للترجمة الفورية - معالجة الخطوات المتعددة"""
    try:
        data = request.json
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({'success': False, 'message': 'لا توجد بيانات في الطلب'}), 400
        
        step = data.get('step')
        
        if not step:
            logger.error(f"Missing 'step' parameter. Received data: {data}")
            return jsonify({'success': False, 'message': 'خطوة غير محددة. يرجى تحديد step'}), 400
        
        logger.info(f"Processing step: {step}, data keys: {list(data.keys())}")
        
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
                video_file = result['file']
                
                # Normalize path - handle cases where path might already include DOWNLOAD_FOLDER
                if os.path.isabs(video_file):
                    full_path = video_file
                else:
                    # Check if path already starts with downloads folder name
                    if video_file.startswith('downloads/') or video_file.startswith('downloads\\'):
                        # Remove the prefix and join properly
                        video_file = video_file.replace('downloads/', '').replace('downloads\\', '')
                    full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], video_file)
                
                # Normalize the path to handle any double slashes or issues
                full_path = os.path.normpath(full_path)
                
                # Verify file exists
                if not os.path.exists(full_path):
                    # Try to find the file
                    if os.path.exists(video_file):
                        full_path = os.path.abspath(video_file)
                    elif os.path.exists(os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file))):
                        full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file))
                
                session['video_file'] = full_path
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
            
            # Normalize and resolve path
            if video_file:
                # If it's already absolute, use it
                if os.path.isabs(video_file):
                    video_file = os.path.normpath(video_file)
                else:
                    # Remove any 'downloads/' prefix if present
                    normalized = video_file.replace('downloads/', '').replace('downloads\\', '')
                    
                    # Try different possible locations
                    possible_paths = [
                        os.path.join(app.config['DOWNLOAD_FOLDER'], normalized),
                        os.path.join(app.config['DOWNLOAD_FOLDER'], os.path.basename(video_file)),
                        os.path.join(app.config['UPLOAD_FOLDER'], normalized),
                        os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_file)),
                        video_file,
                        normalized
                    ]
                    
                    for path in possible_paths:
                        abs_path = os.path.abspath(path)
                        if os.path.exists(abs_path):
                            video_file = abs_path
                            break
            
            if not video_file or not os.path.exists(video_file):
                logger.error(f"Video file not found: {video_file}")
                logger.error(f"Download folder: {app.config['DOWNLOAD_FOLDER']}")
                logger.error(f"Files in download folder: {os.listdir(app.config['DOWNLOAD_FOLDER']) if os.path.exists(app.config['DOWNLOAD_FOLDER']) else 'Folder does not exist'}")
                return jsonify({'success': False, 'message': f'ملف الفيديو غير موجود: {video_file}'}), 400
            
            # Extract audio using ffmpeg with optimized settings for speed
            audio_file = video_file.rsplit('.', 1)[0] + '_audio.wav'
            
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV format for Whisper
                '-ar', '16000',  # 16kHz sample rate (Whisper requirement)
                '-ac', '1',  # Mono
                '-threads', '0',  # Use all available CPU threads
                '-y',  # Overwrite
                audio_file
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                session['audio_file'] = audio_file
                return jsonify({
                    'success': True,
                    'audio_file': audio_file
                })
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr}")
                return jsonify({'success': False, 'message': f'خطأ في استخراج الصوت: {e.stderr}'}), 500
            except subprocess.TimeoutExpired:
                return jsonify({'success': False, 'message': 'انتهت مهلة استخراج الصوت'}), 500
                
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
            
            # Detect if GPU is available for FP16
            import torch
            use_fp16 = torch.cuda.is_available()  # Only use FP16 if GPU is available
            
            # Transcribe with optimized settings for speed
            options = {
                'language': None if language == 'auto' else language,
                'task': 'transcribe',
                'fp16': use_fp16,  # Use FP16 only if GPU available, otherwise FP32
                'beam_size': 3,  # Reduced from default 5 for speed
                'best_of': 2,  # Reduced from default 5 for speed
                'temperature': 0.0,  # Deterministic output
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'condition_on_previous_text': True,
                'initial_prompt': None,
                'word_timestamps': True  # Keep word timestamps for accurate subtitles
            }
            
            result = model.transcribe(audio_file, **options)
            
            session['transcript'] = result['text']
            session['source_language'] = result.get('language', language)
            # Store segments for accurate subtitle timing
            if 'segments' in result:
                session['whisper_segments'] = result['segments']
            
            return jsonify({
                'success': True,
                'text': result['text'],
                'language': result.get('language', language),
                'segments': result.get('segments', [])  # Return segments to frontend
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
            quality = data.get('quality', 'original')  # Get quality from request
            
            # Handle relative paths
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
            
            # Get video duration for accurate subtitle timing
            video_duration = VideoProcessor.get_video_duration(video_file)
            
            # Get Whisper segments if available (for accurate timing)
            whisper_segments = session.get('whisper_segments', None)
            original_text = session.get('transcript', '')
            
            # Create SRT file with accurate timing
            if whisper_segments and len(whisper_segments) > 0:
                # Use Whisper segments for precise timing
                # IMPORTANT: Translate each segment individually for better accuracy
                segments_for_srt = []
                
                logger.info(f"Processing {len(whisper_segments)} Whisper segments")
                
                # Translate each segment individually to maintain timing accuracy
                translator = GoogleTranslator(source=session.get('source_language', 'auto'), target='ar')
                
                for i, segment in enumerate(whisper_segments):
                    # Use original timing from Whisper (these are accurate)
                    start_time = float(segment.get('start', 0))
                    end_time = float(segment.get('end', start_time + 3))
                    
                    # Get original segment text
                    original_segment_text = segment.get('text', '').strip()
                    
                    if not original_segment_text:
                        continue
                    
                    # Translate this segment individually
                    try:
                        translated_segment_text = translator.translate(original_segment_text)
                        logger.info(f"Segment {i+1}/{len(whisper_segments)}: {start_time:.2f}s-{end_time:.2f}s")
                    except Exception as e:
                        logger.warning(f"Translation failed for segment {i+1}, using original: {e}")
                        # Fallback: use ratio-based translation
                        if original_text:
                            original_words = original_text.split()
                            translated_words = subtitle_text.split()
                            if len(original_words) > 0:
                                word_ratio = len(translated_words) / len(original_words)
                                segment_words = original_segment_text.split()
                                translated_words_count = max(1, int(len(segment_words) * word_ratio))
                                # Estimate position in full text
                                words_before = len(original_text[:original_text.find(original_segment_text)].split())
                                start_idx = int(words_before * word_ratio)
                                end_idx = min(start_idx + translated_words_count, len(translated_words))
                                translated_segment_text = ' '.join(translated_words[start_idx:end_idx])
                            else:
                                translated_segment_text = original_segment_text
                        else:
                            translated_segment_text = original_segment_text
                    
                    if translated_segment_text.strip():
                        segments_for_srt.append({
                            'start': start_time,
                            'end': end_time,
                            'text': translated_segment_text.strip()
                        })
                
                # If we have segments, use them
                if segments_for_srt:
                    logger.info(f"Created {len(segments_for_srt)} subtitle segments with precise timing")
                    srt_content = SubtitleProcessor.create_srt(
                        subtitle_text, 
                        duration=video_duration,
                        segments=segments_for_srt
                    )
                else:
                    logger.warning("No segments created, falling back to calculated timing")
                    # Fallback: create SRT with calculated timing
                    srt_content = SubtitleProcessor.create_srt(
                        subtitle_text,
                        duration=video_duration
                    )
            else:
                logger.warning("No Whisper segments available, using calculated timing")
                # Fallback: create SRT with calculated timing
                srt_content = SubtitleProcessor.create_srt(
                    subtitle_text,
                    duration=video_duration
                )
            
            srt_filename = f"instant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # Create ASS file with styling (use settings from request)
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
            
            # Output filename
            output_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Prepare subtitle settings dict for merge
            subtitle_settings = {
                'font_name': font_name,
                'font_size': font_size,
                'font_color': font_color,
                'bg_color': bg_color,
                'bg_opacity': bg_opacity,
                'position': position
            }
            
            # Merge subtitle with video (use quality from request)
            result = VideoProcessor.merge_subtitles(
                video_file,
                ass_path,
                output_path,
                quality=quality,
                subtitle_settings=subtitle_settings
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
            logger.error(f"Unknown step: {step}")
            return jsonify({'success': False, 'message': f'خطوة غير صحيحة: {step}'}), 400
            
    except Exception as e:
        logger.error(f"Instant translate error: {e}")
        logger.error(traceback.format_exc())
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
            import torch
            model = whisper.load_model(whisper_model)
            
            # Detect if GPU is available for FP16
            use_fp16 = torch.cuda.is_available()
            
            # Transcribe directly from video with optimized settings
            transcribe_options = {
                'language': None,
                'task': 'transcribe',
                'fp16': use_fp16,  # Use FP16 only if GPU available
                'beam_size': 3,  # Reduced for speed
                'best_of': 2,  # Reduced for speed
                'temperature': 0.0,
                'word_timestamps': True
            }
            transcribe_result = model.transcribe(video_path, **transcribe_options)
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
        
        # Detect if GPU is available for FP16
        import torch
        use_fp16 = torch.cuda.is_available()
        
        # Transcribe with optimized settings for speed
        options = {
            'language': None if language == 'auto' else language,
            'task': 'transcribe',
            'fp16': use_fp16,  # Use FP16 only if GPU available
            'beam_size': 3,  # Reduced from default 5 for speed
            'best_of': 2,  # Reduced from default 5 for speed
            'temperature': 0.0,  # Deterministic output
            'word_timestamps': True
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

@app.route('/api/transcribe-from-url', methods=['POST'])
def api_transcribe_from_url():
    """تحميل الفيديو من رابط وتحويله إلى نص"""
    if not WHISPER_AVAILABLE:
        return jsonify({'success': False, 'message': 'Whisper غير متوفر'}), 503
    
    try:
        data = request.json
        url = data.get('url')
        quality = data.get('quality', '720p')
        language = data.get('language', 'auto')
        model_size = data.get('model', 'base')
        
        if not url:
            return jsonify({'success': False, 'message': 'يرجى إدخال رابط الفيديو'}), 400
        
        logger.info(f"Transcribing from URL: {url}")
        
        # Step 1: Download video
        quality_map = {
            'best': 'best',
            '720p': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            '480p': 'bestvideo[height<=480]+bestaudio/best[height<=480]'
        }
        
        download_result = downloader.download_with_ytdlp(url, quality_map.get(quality, '720p'))
        
        if not download_result['success']:
            return jsonify({'success': False, 'message': f'فشل تحميل الفيديو: {download_result["message"]}'}), 400
        
        video_file = download_result['file']
        
        # Normalize path
        if not os.path.isabs(video_file):
            if video_file.startswith('downloads/') or video_file.startswith('downloads\\'):
                video_file = video_file.replace('downloads/', '').replace('downloads\\', '')
            full_path = os.path.join(app.config['DOWNLOAD_FOLDER'], video_file)
        else:
            full_path = video_file
        
        full_path = os.path.normpath(full_path)
        
        if not os.path.exists(full_path):
            # Try to find the file
            basename = os.path.basename(video_file)
            possible_paths = [
                os.path.join(app.config['DOWNLOAD_FOLDER'], basename),
                full_path
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    full_path = path
                    break
        
        if not os.path.exists(full_path):
            return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود بعد التحميل'}), 400
        
        logger.info(f"Video downloaded to: {full_path}")
        
        # Step 2: Load Whisper model and transcribe
        import whisper
        import torch
        
        model = whisper.load_model(model_size)
        use_fp16 = torch.cuda.is_available()
        
        # Transcribe directly from video file
        transcribe_options = {
            'language': None if language == 'auto' else language,
            'task': 'transcribe',
            'fp16': use_fp16,
            'beam_size': 3,
            'best_of': 2,
            'temperature': 0.0,
            'word_timestamps': True
        }
        
        logger.info("Starting transcription...")
        result = model.transcribe(full_path, **transcribe_options)
        
        # Create SRT file
        video_basename = os.path.splitext(os.path.basename(full_path))[0]
        srt_content = SubtitleProcessor.create_srt(result['text'])
        srt_filename = f"{video_basename}_transcribed.srt"
        srt_path = os.path.join(app.config['SUBTITLE_FOLDER'], srt_filename)
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.info(f"Transcription completed. Text length: {len(result['text'])}")
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', language),
            'srt_file': srt_filename,
            'video_title': download_result.get('info', {}).get('title', 'Video')
        })
        
    except Exception as e:
        logger.error(f"Transcribe from URL error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'خطأ في التحويل: {str(e)}'}), 500

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

@app.route('/api/storage-info', methods=['GET'])
def api_storage_info():
    """الحصول على معلومات الحجم المستخدم"""
    try:
        total_size = 0
        
        # Calculate size for each folder
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
        
        # Convert to MB
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
    """حذف الملفات المحملة/المرفوعة/المترجمة"""
    try:
        data = request.json
        cleanup_type = data.get('type', 'all')  # 'downloads', 'uploads', 'outputs', 'subtitles', 'all'
        
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
