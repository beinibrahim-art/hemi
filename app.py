#!/usr/bin/env python3
"""
تطبيق متكامل: محمل الوسائط + تحويل الصوت/الفيديو إلى نص
يجمع جميع مميزات السكريبتين في تطبيق واحد
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import json
import subprocess
import threading
import time
import re
import secrets
import uuid
import zipfile
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from collections import defaultdict

# Whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available - transcription disabled")

# Video processing
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️  MoviePy not available - video extraction disabled")

# Speaker Diarization
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("⚠️  pyannote.audio not available - speaker diarization disabled")

# Translation
try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("⚠️  deep-translator not available - translation disabled")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
DOWNLOAD_DIR = Path("downloads")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

DOWNLOAD_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Store progress
download_progress = {}
transcription_status = {}

# File extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS

# Speaker colors
SPEAKER_COLORS = [
    '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', 
    '#10b981', '#06b6d4', '#ef4444', '#84cc16',
    '#f97316', '#14b8a6', '#a855f7', '#eab308'
]


# ============================================================================
# MEDIA DOWNLOADER FUNCTIONS - IMPROVED
# ============================================================================

class SmartMediaDownloader:
    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check if required tools are installed"""
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
        """Detect which platform the URL is from"""
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
        """Get all available formats for the video - IMPROVED"""
        try:
            # First try to get JSON info directly (more reliable)
            cmd_json = ['yt-dlp', '-J', '--no-warnings', '--no-playlist', url]
            result_json = subprocess.run(cmd_json, capture_output=True, text=True, timeout=60)
            
            if result_json.returncode == 0:
                try:
                    video_info = json.loads(result_json.stdout)
                    formats = self._parse_formats_from_json(video_info)
                    return {
                        'success': True,
                        'formats': formats,
                        'info': self._extract_info_from_json(video_info),
                        'platform': self.detect_platform(url)
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback to format list
            cmd = ['yt-dlp', '-F', '--no-warnings', '--no-playlist', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return {'success': False, 'error': f'Could not fetch formats: {result.stderr[:200]}'}
            
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
    
    def _parse_formats_from_json(self, video_info: dict) -> dict:
        """Parse formats from JSON info - MORE ACCURATE"""
        formats = {
            'video_audio': [],
            'video_only': [],
            'audio_only': [],
            'best': None
        }
        
        format_list = video_info.get('formats', [])
        
        for fmt in format_list:
            format_id = str(fmt.get('format_id', ''))
            height = fmt.get('height')
            width = fmt.get('width')
            resolution = None
            
            if height:
                resolution = f"{height}p"
            elif width:
                # Estimate height from width
                if width >= 3840:
                    resolution = "2160p"
                elif width >= 2560:
                    resolution = "1440p"
                elif width >= 1920:
                    resolution = "1080p"
                elif width >= 1280:
                    resolution = "720p"
                elif width >= 854:
                    resolution = "480p"
                else:
                    resolution = "360p"
            
            filesize = fmt.get('filesize') or fmt.get('filesize_approx', 0)
            filesize_str = self._format_filesize(filesize)
            
            format_info = {
                'id': format_id,
                'resolution': resolution,
                'note': '',
                'filesize': filesize_str,
                'vcodec': fmt.get('vcodec', 'none'),
                'acodec': fmt.get('acodec', 'none'),
                'fps': fmt.get('fps'),
                'tbr': fmt.get('tbr'),
                'vbr': fmt.get('vbr'),
                'abr': fmt.get('abr')
            }
            
            has_video = fmt.get('vcodec') and fmt.get('vcodec') != 'none'
            has_audio = fmt.get('acodec') and fmt.get('acodec') != 'none'
            
            if has_video and has_audio:
                format_info['note'] = f'{resolution or "Unknown"}'
                formats['video_audio'].append(format_info)
            elif has_video:
                format_info['note'] = f'{resolution or "Unknown"} (No Audio)'
                formats['video_only'].append(format_info)
            elif has_audio:
                bitrate = fmt.get('abr', 0)
                if bitrate:
                    format_info['bitrate'] = f'{int(bitrate)}kbps'
                format_info['note'] = 'Audio Only'
                formats['audio_only'].append(format_info)
        
        # Sort formats by resolution/quality
        formats['video_audio'].sort(key=lambda x: self._get_resolution_value(x.get('resolution', '0p')), reverse=True)
        formats['video_only'].sort(key=lambda x: self._get_resolution_value(x.get('resolution', '0p')), reverse=True)
        formats['audio_only'].sort(key=lambda x: int(x.get('bitrate', '0kbps').replace('kbps', '')), reverse=True)
        
        formats['presets'] = self._create_smart_presets(formats)
        
        return formats
    
    def _parse_formats(self, output: str) -> dict:
        """Parse yt-dlp format output"""
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
        
        # Sort formats
        formats['video_audio'].sort(key=lambda x: self._get_resolution_value(x.get('resolution', '0p')), reverse=True)
        formats['video_only'].sort(key=lambda x: self._get_resolution_value(x.get('resolution', '0p')), reverse=True)
        
        formats['presets'] = self._create_smart_presets(formats)
        
        return formats
    
    def _get_resolution_value(self, resolution: str) -> int:
        """Get numeric value for resolution sorting"""
        if not resolution:
            return 0
        try:
            return int(resolution.replace('p', ''))
        except:
            return 0
    
    def _format_filesize(self, size_bytes: int) -> str:
        """Format file size"""
        if not size_bytes:
            return 'Unknown'
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def _extract_filesize(self, line: str) -> str:
        size_match = re.search(r'(\d+\.?\d*\s*[KMG]i?B)', line, re.IGNORECASE)
        if size_match:
            return size_match.group(1)
        return 'Unknown'
    
    def _extract_info_from_json(self, video_info: dict) -> dict:
        """Extract video info from JSON"""
        return {
            'title': video_info.get('title', 'Unknown'),
            'uploader': video_info.get('uploader', 'Unknown'),
            'duration': video_info.get('duration', 0),
            'thumbnail': video_info.get('thumbnail', ''),
            'description': video_info.get('description', '')[:200],
            'view_count': video_info.get('view_count', 0),
            'upload_date': video_info.get('upload_date', '')
        }
    
    def _create_smart_presets(self, formats: dict) -> list:
        """Create smart download presets - IMPROVED"""
        presets = []
        
        # Best quality preset
        presets.append({
            'id': 'best',
            'name': 'أفضل جودة',
            'description': 'أعلى جودة متاحة (تلقائي)',
            'icon': 'crown',
            'command': 'bestvideo+bestaudio/best',
            'format_id': None
        })
        
        # Check available resolutions
        all_video_formats = formats['video_audio'] + formats['video_only']
        resolutions = set()
        for fmt in all_video_formats:
            res = fmt.get('resolution')
            if res:
                resolutions.add(res)
        
        resolutions_list = sorted([self._get_resolution_value(r) for r in resolutions], reverse=True)
        
        # 4K preset
        if any(r >= 2160 for r in resolutions_list):
            presets.append({
                'id': '4k',
                'name': '4K Ultra HD',
                'description': '2160p - جودة فائقة',
                'icon': 'sparkles',
                'command': 'bestvideo[height<=2160]+bestaudio/best[height<=2160]',
                'format_id': None
            })
        
        # 1440p preset
        if any(r >= 1440 for r in resolutions_list):
            presets.append({
                'id': '1440p',
                'name': '1440p QHD',
                'description': 'جودة عالية جداً',
                'icon': 'star',
                'command': 'bestvideo[height<=1440]+bestaudio/best[height<=1440]',
                'format_id': None
            })
        
        # 1080p preset
        if any(r >= 1080 for r in resolutions_list):
            presets.append({
                'id': '1080p',
                'name': '1080p Full HD',
                'description': 'جودة ممتازة',
                'icon': 'video',
                'command': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                'format_id': None
            })
        
        # 720p preset
        if any(r >= 720 for r in resolutions_list):
            presets.append({
                'id': '720p',
                'name': '720p HD',
                'description': 'جودة جيدة - حجم متوازن',
                'icon': 'film',
                'command': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                'format_id': None
            })
        
        # 480p preset
        if any(r >= 480 for r in resolutions_list):
            presets.append({
                'id': '480p',
                'name': '480p SD',
                'description': 'جودة متوسطة - حجم صغير',
                'icon': 'smartphone',
                'command': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
                'format_id': None
            })
        
        # Audio only preset
        if formats['audio_only']:
            presets.append({
                'id': 'audio',
                'name': 'صوت فقط',
                'description': 'MP3 بأفضل جودة',
                'icon': 'music',
                'command': 'bestaudio/best',
                'format_id': None
            })
        
        return presets
    
    def _get_video_info(self, url: str) -> dict:
        """Get video information"""
        try:
            cmd = ['yt-dlp', '--dump-json', '--no-warnings', '--no-playlist', '--skip-download', url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'description': info.get('description', '')[:200],
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        return {
            'title': 'Unknown',
            'uploader': 'Unknown',
            'duration': 0,
            'thumbnail': '',
            'description': '',
            'view_count': 0,
            'upload_date': ''
        }
    
    def download_with_format(self, url: str, format_command: str, 
                            download_id: str, is_audio: bool = False):
        """Download with specific format using improved strategies"""
        try:
            download_progress[download_id] = {
                'status': 'starting',
                'percent': '0%',
                'method': 'Preparing download...',
                'filename': None,
                'error': None
            }
            
            # Determine output filename
            output_template = str(self.output_dir / f'{download_id}_%(title)s.%(ext)s')
            
            # Try strategies in order
            strategies = [
                ('Direct download with format', self._download_strategy_direct),
                ('Download with cookies', self._download_strategy_cookies),
                ('Download compatible format', self._download_strategy_compatible),
                ('Download best available', self._download_strategy_fallback)
            ]
            
            for strategy_name, strategy_func in strategies:
                download_progress[download_id]['method'] = f'Trying: {strategy_name}...'
                
                result = strategy_func(url, format_command, download_id, output_template, is_audio)
                
                if result['success']:
                    download_progress[download_id] = {
                        'status': 'completed',
                        'percent': '100%',
                        'message': 'تم التحميل بنجاح!',
                        'filename': result.get('filename'),
                        'filepath': result.get('filepath')
                    }
                    return {'success': True, 'filename': result.get('filename')}
                else:
                    download_progress[download_id]['error'] = result.get('error', 'Unknown error')
            
            # All strategies failed
            download_progress[download_id] = {
                'status': 'error',
                'message': 'فشل التحميل بجميع الطرق',
                'error': 'All download strategies failed'
            }
            return {'success': False, 'error': 'All strategies failed'}
                
        except Exception as e:
            download_progress[download_id] = {
                'status': 'error',
                'message': f'خطأ: {str(e)}',
                'error': str(e)
            }
            return {'success': False, 'error': str(e)}
    
    def _download_strategy_direct(self, url: str, format_cmd: str, download_id: str, 
                                 output_template: str, is_audio: bool) -> dict:
        """Strategy 1: Direct download with specified format"""
        try:
            cmd = ['yt-dlp', '--no-warnings', '--no-playlist', '--newline']
            
            if is_audio or format_cmd == 'audio' or format_cmd == 'bestaudio/best':
                cmd.extend(['-x', '--audio-format', 'mp3', '--audio-quality', '0'])
            else:
                cmd.extend(['-f', format_cmd])
                # Try to merge to mp4 if separate video/audio
                if '+' in format_cmd or 'bestvideo' in format_cmd:
                    cmd.extend(['--merge-output-format', 'mp4'])
            
            cmd.extend(['-o', output_template, url])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            downloaded_file = self._monitor_download(process, download_id)
            
            if downloaded_file and os.path.exists(downloaded_file):
                return {'success': True, 'filename': os.path.basename(downloaded_file), 
                       'filepath': downloaded_file}
            
            return {'success': False, 'error': 'Download completed but file not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _download_strategy_cookies(self, url: str, format_cmd: str, download_id: str,
                                   output_template: str, is_audio: bool) -> dict:
        """Strategy 2: Download with browser cookies"""
        try:
            cmd = ['yt-dlp', '--no-warnings', '--no-playlist', '--newline']
            
            # Try different browsers
            browsers = ['chrome', 'firefox', 'edge', 'safari']
            for browser in browsers:
                try:
                    cmd_cookies = cmd + ['--cookies-from-browser', browser]
                    
                    if is_audio or format_cmd == 'audio':
                        cmd_cookies.extend(['-x', '--audio-format', 'mp3'])
                    else:
                        cmd_cookies.extend(['-f', format_cmd])
                        if '+' in format_cmd:
                            cmd_cookies.extend(['--merge-output-format', 'mp4'])
                    
                    cmd_cookies.extend(['-o', output_template, url])
                    
                    process = subprocess.Popen(cmd_cookies, stdout=subprocess.PIPE, 
                                              stderr=subprocess.STDOUT, text=True, bufsize=1)
                    
                    downloaded_file = self._monitor_download(process, download_id)
                    
                    if downloaded_file and os.path.exists(downloaded_file):
                        return {'success': True, 'filename': os.path.basename(downloaded_file),
                               'filepath': downloaded_file}
                except:
                    continue
            
            return {'success': False, 'error': 'Cookie strategy failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _download_strategy_compatible(self, url: str, format_cmd: str, download_id: str,
                                     output_template: str, is_audio: bool) -> dict:
        """Strategy 3: Download compatible format"""
        try:
            cmd = ['yt-dlp', '--no-warnings', '--no-playlist', '--newline']
            
            if is_audio:
                cmd.extend(['-x', '--audio-format', 'mp3', '--audio-quality', '0'])
            else:
                # Try best mp4 format
                cmd.extend(['-f', 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best'])
                cmd.extend(['--merge-output-format', 'mp4'])
            
            cmd.extend(['-o', output_template, url])
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            downloaded_file = self._monitor_download(process, download_id)
            
            if downloaded_file and os.path.exists(downloaded_file):
                return {'success': True, 'filename': os.path.basename(downloaded_file),
                       'filepath': downloaded_file}
            
            return {'success': False, 'error': 'Compatible format download failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _download_strategy_fallback(self, url: str, format_cmd: str, download_id: str,
                                   output_template: str, is_audio: bool) -> dict:
        """Strategy 4: Fallback to best available"""
        try:
            cmd = ['yt-dlp', '--no-warnings', '--no-playlist', '--newline', 
                  '-o', output_template, url]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            downloaded_file = self._monitor_download(process, download_id)
            
            if downloaded_file and os.path.exists(downloaded_file):
                return {'success': True, 'filename': os.path.basename(downloaded_file),
                       'filepath': downloaded_file}
            
            return {'success': False, 'error': 'Fallback download failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _monitor_download(self, process, download_id: str) -> str:
        """Monitor download progress and return downloaded file path"""
        downloaded_file = None
        last_percent = 0
        
        try:
            for line in process.stdout:
                line = line.strip()
                
                # Parse download progress
                if '[download]' in line:
                    # Extract percentage
                    percent_match = re.search(r'(\d+\.?\d*)%', line)
                    if percent_match:
                        percent = float(percent_match.group(1))
                        if percent > last_percent:
                            last_percent = percent
                            download_progress[download_id] = {
                                'status': 'downloading',
                                'percent': f'{int(percent)}%',
                                'method': download_progress.get(download_id, {}).get('method', 'Downloading...')
                            }
                    
                    # Extract filename
                    if 'Destination:' in line or 'has already been downloaded' in line:
                        filename_match = re.search(r'Destination:\s*(.+)|(.+)\s+has already been downloaded', line)
                        if filename_match:
                            downloaded_file = filename_match.group(1) or filename_match.group(2)
                            downloaded_file = downloaded_file.strip()
                    
                    # Check for completion
                    if '100%' in line or 'has already been downloaded' in line:
                        download_progress[download_id] = {
                            'status': 'processing',
                            'percent': '100%',
                            'method': 'Finalizing...'
                        }
                
                # Parse error messages
                elif 'ERROR' in line or 'WARNING' in line:
                    download_progress[download_id]['error'] = line[:200]
            
            process.wait()
            
            # If file not found in output, search for recently created files
            if not downloaded_file or not os.path.exists(downloaded_file):
                # Find the most recently created file in download directory
                files = list(self.output_dir.glob(f'{download_id}_*'))
                if files:
                    # Sort by modification time
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    downloaded_file = str(files[0])
            
            return downloaded_file if downloaded_file and os.path.exists(downloaded_file) else None
            
        except Exception as e:
            print(f"Monitor error: {e}")
            return None


# ============================================================================
# TRANSCRIPTION FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio_from_video(video_path, output_audio_path):
    """استخراج الصوت من الفيديو"""
    if not MOVIEPY_AVAILABLE:
        return False
    
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def perform_diarization(audio_path):
    """كشف المتحدثين في الملف الصوتي"""
    if not DIARIZATION_AVAILABLE:
        return None
    
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
        diarization = pipeline(audio_path)
        
        speakers_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_timeline.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        return speakers_timeline
    except Exception as e:
        print(f"Diarization error: {e}")
        return None


def match_speakers_to_segments(segments, speakers_timeline):
    """مطابقة المتحدثين مع النصوص"""
    if not speakers_timeline:
        return segments
    
    for segment in segments:
        segment_time = (segment['start'] + segment['end']) / 2
        
        for speaker_info in speakers_timeline:
            if speaker_info['start'] <= segment_time <= speaker_info['end']:
                segment['speaker'] = speaker_info['speaker']
                break
        
        if 'speaker' not in segment:
            segment['speaker'] = 'Unknown'
    
    return segments


def translate_text(text, target_lang='ar'):
    """ترجمة النص"""
    if not TRANSLATION_AVAILABLE:
        return text
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def format_timestamp(seconds, vtt=False):
    """تحويل الثواني إلى صيغة الوقت"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, output_file):
    """كتابة ملف SRT"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            speaker = segment.get('speaker', '')
            text = segment['text'].strip()
            
            if speaker:
                text = f"[{speaker}] {text}"
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def write_vtt(segments, output_file):
    """كتابة ملف VTT"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start_time = format_timestamp(segment['start'], vtt=True)
            end_time = format_timestamp(segment['end'], vtt=True)
            speaker = segment.get('speaker', '')
            text = segment['text'].strip()
            
            if speaker:
                text = f"[{speaker}] {text}"
            
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")


def process_transcription(job_id, file_path, model_size, language, output_format, 
                          enable_diarization=False, translation_lang=None):
    """معالجة ملف واحد للتحويل إلى نص"""
    try:
        transcription_status[job_id]['status'] = 'processing'
        transcription_status[job_id]['progress'] = 10
        
        if not WHISPER_AVAILABLE:
            raise Exception("Whisper is not installed")
        
        transcription_status[job_id]['message'] = 'تحميل نموذج Whisper...'
        model = whisper.load_model(model_size)
        transcription_status[job_id]['progress'] = 20
        
        file_ext = Path(file_path).suffix.lower()
        audio_file = file_path
        
        if file_ext.replace('.', '') in ALLOWED_VIDEO_EXTENSIONS:
            transcription_status[job_id]['message'] = 'استخراج الصوت من الفيديو...'
            temp_audio = os.path.join(UPLOAD_DIR, f"{job_id}_audio.wav")
            if extract_audio_from_video(file_path, temp_audio):
                audio_file = temp_audio
            transcription_status[job_id]['progress'] = 30
        
        speakers_timeline = None
        if enable_diarization:
            transcription_status[job_id]['message'] = 'كشف المتحدثين...'
            speakers_timeline = perform_diarization(audio_file)
            transcription_status[job_id]['progress'] = 50
        
        transcription_status[job_id]['message'] = 'تحويل الصوت إلى نص...'
        options = {}
        if language and language != 'auto':
            options['language'] = language
        
        result = model.transcribe(audio_file, **options)
        transcription_status[job_id]['progress'] = 70
        
        if speakers_timeline:
            transcription_status[job_id]['message'] = 'مطابقة المتحدثين مع النصوص...'
            result['segments'] = match_speakers_to_segments(result['segments'], speakers_timeline)
        
        if translation_lang and translation_lang != 'none':
            transcription_status[job_id]['message'] = 'ترجمة النص...'
            for segment in result['segments']:
                segment['translation'] = translate_text(segment['text'], translation_lang)
            transcription_status[job_id]['progress'] = 85
        
        transcription_status[job_id]['message'] = 'حفظ النتائج...'
        base_name = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_files = []
        
        if output_format in ['txt', 'all']:
            txt_file = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                for segment in result['segments']:
                    speaker = segment.get('speaker', '')
                    text = segment['text'].strip()
                    if speaker:
                        f.write(f"[{speaker}] {text}\n")
                    else:
                        f.write(f"{text}\n")
            output_files.append({
                'type': 'txt',
                'name': os.path.basename(txt_file),
                'path': txt_file
            })
        
        if output_format in ['srt', 'all']:
            srt_file = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.srt")
            write_srt(result['segments'], srt_file)
            output_files.append({
                'type': 'srt',
                'name': os.path.basename(srt_file),
                'path': srt_file
            })
        
        if output_format in ['vtt', 'all']:
            vtt_file = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.vtt")
            write_vtt(result['segments'], vtt_file)
            output_files.append({
                'type': 'vtt',
                'name': os.path.basename(vtt_file),
                'path': vtt_file
            })
        
        if output_format in ['json', 'all']:
            json_file = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            output_files.append({
                'type': 'json',
                'name': os.path.basename(json_file),
                'path': json_file
            })
        
        speakers_info = {}
        if speakers_timeline:
            for segment in result['segments']:
                speaker = segment.get('speaker', 'Unknown')
                if speaker not in speakers_info:
                    speaker_num = len(speakers_info)
                    speakers_info[speaker] = {
                        'name': speaker,
                        'color': SPEAKER_COLORS[speaker_num % len(SPEAKER_COLORS)],
                        'segments_count': 0
                    }
                speakers_info[speaker]['segments_count'] += 1
        
        transcription_status[job_id]['progress'] = 100
        transcription_status[job_id]['status'] = 'completed'
        transcription_status[job_id]['message'] = 'تم التحويل بنجاح!'
        transcription_status[job_id]['text'] = result['text']
        transcription_status[job_id]['segments'] = result['segments']
        transcription_status[job_id]['output_files'] = output_files
        transcription_status[job_id]['detected_language'] = result.get('language', 'Unknown')
        transcription_status[job_id]['timestamp'] = datetime.now().isoformat()
        transcription_status[job_id]['speakers_info'] = speakers_info
        
    except Exception as e:
        transcription_status[job_id]['status'] = 'error'
        transcription_status[job_id]['message'] = f'خطأ: {str(e)}'
        transcription_status[job_id]['progress'] = 0


# Initialize downloader
downloader = SmartMediaDownloader()


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('unified_index.html')


# Media Downloader Routes
@app.route('/api/media/analyze', methods=['POST'])
def analyze_url():
    """Analyze URL and get available formats"""
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})
    
    result = downloader.get_available_formats(url)
    return jsonify(result)


@app.route('/api/media/download', methods=['POST'])
def start_media_download():
    """Start download with selected format"""
    data = request.json
    url = data.get('url', '')
    format_command = data.get('format', 'best')
    is_audio = data.get('audio_only', False)
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})
    
    download_id = secrets.token_hex(8)
    
    thread = threading.Thread(
        target=downloader.download_with_format,
        args=(url, format_command, download_id, is_audio)
    )
    thread.start()
    
    return jsonify({'success': True, 'download_id': download_id})


@app.route('/api/media/progress/<download_id>')
def get_download_progress(download_id):
    """Get download progress"""
    progress = download_progress.get(download_id, {
        'status': 'unknown',
        'percent': '0%'
    })
    return jsonify(progress)


@app.route('/api/media/files')
def list_media_files():
    """List downloaded media files"""
    files = []
    for file in DOWNLOAD_DIR.glob('*'):
        if file.is_file():
            files.append({
                'name': file.name,
                'size': file.stat().st_size,
                'date': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    files.sort(key=lambda x: x['date'], reverse=True)
    return jsonify({'files': files})


@app.route('/api/media/download-file/<filename>')
def download_media_file(filename):
    """Download a media file"""
    file_path = DOWNLOAD_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


# Transcription Routes
@app.route('/api/transcribe/upload', methods=['POST'])
def upload_for_transcription():
    """رفع الملفات وبدء المعالجة"""
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'لم يتم رفع ملفات'}), 400
    
    job_ids = []
    
    for file in files:
        if not allowed_file(file.filename):
            continue
        
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
        file.save(file_path)
        
        model_size = request.form.get('model', 'base')
        language = request.form.get('language', 'auto')
        output_format = request.form.get('format', 'txt')
        enable_diarization = request.form.get('enable_diarization', 'false') == 'true'
        translation_lang = request.form.get('translation_lang', 'none')
        
        transcription_status[job_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'في الانتظار...',
            'filename': filename
        }
        
        thread = threading.Thread(
            target=process_transcription,
            args=(job_id, file_path, model_size, language, output_format, 
                  enable_diarization, translation_lang)
        )
        thread.start()
        
        job_ids.append(job_id)
    
    return jsonify({'job_ids': job_ids})


@app.route('/api/transcribe/status/<job_id>')
def get_transcription_status(job_id):
    """الحصول على حالة المعالجة"""
    if job_id not in transcription_status:
        return jsonify({'error': 'المهمة غير موجودة'}), 404
    
    return jsonify(transcription_status[job_id])


@app.route('/api/transcribe/download/<filename>')
def download_transcription_file(filename):
    """تحميل الملف الناتج"""
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/api/features')
def get_features():
    """الحصول على المميزات المتاحة"""
    return jsonify({
        'whisper_available': WHISPER_AVAILABLE,
        'moviepy_available': MOVIEPY_AVAILABLE,
        'diarization_available': DIARIZATION_AVAILABLE,
        'translation_available': TRANSLATION_AVAILABLE,
        'ytdlp_available': downloader.available_tools.get('yt-dlp', False),
        'ffmpeg_available': downloader.available_tools.get('ffmpeg', False)
    })


if __name__ == '__main__':
    print("=" * 80)
    print("🎬 التطبيق المتكامل: محمل الوسائط + تحويل الصوت إلى نص")
    print("=" * 80)
    print("\n📱 افتح المتصفح على: http://localhost:5000")
    print("\n✅ المميزات المتاحة:")
    print(f"  • تحميل الوسائط: {'✓' if downloader.available_tools.get('yt-dlp') else '✗'}")
    print(f"  • تحويل إلى نص (Whisper): {'✓' if WHISPER_AVAILABLE else '✗'}")
    print(f"  • كشف المتحدثين: {'✓' if DIARIZATION_AVAILABLE else '✗'}")
    print(f"  • الترجمة: {'✓' if TRANSLATION_AVAILABLE else '✗'}")
    print(f"  • معالجة الفيديو: {'✓' if MOVIEPY_AVAILABLE else '✗'}")
    print(f"  • ffmpeg: {'✓' if downloader.available_tools.get('ffmpeg') else '✗'}")
    print("\n⚠️  للإيقاف: اضغط Ctrl+C\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
