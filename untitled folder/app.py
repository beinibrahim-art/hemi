#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
التطبيق المتكامل للترجمة والتحميل v6.0 - النسخة المحسّنة
تم حل جميع المشاكل وإضافة تحسينات شاملة
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
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory
from werkzeug.utils import secure_filename
import yt_dlp

# استيراد المكتبات الاختيارية
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

# إعدادات التطبيق
app = Flask(__name__)
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024 * 1024,  # 5GB
    UPLOAD_FOLDER='uploads',
    DOWNLOAD_FOLDER='downloads',
    OUTPUT_FOLDER='outputs',
    SUBTITLE_FOLDER='subtitles',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=1800
)

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


# =============================================================================
# نظام إدارة المهام المحسّن
# =============================================================================

class TaskManager:
    """نظام إدارة المهام مع تتبع التقدم"""
    
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_task(self, task_id: str, total_steps: int = 5):
        """إنشاء مهمة جديدة"""
        with self.lock:
            self.tasks[task_id] = {
                'status': 'pending',
                'current_step': 0,
                'total_steps': total_steps,
                'progress': 0,
                'message': 'في الانتظار...',
                'result': None,
                'error': None,
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
    
    def update_task(self, task_id: str, **kwargs):
        """تحديث حالة المهمة"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
    
    def get_task(self, task_id: str) -> Dict:
        """الحصول على حالة المهمة"""
        with self.lock:
            return self.tasks.get(task_id, {})
    
    def complete_task(self, task_id: str, result: any):
        """إكمال المهمة"""
        self.update_task(
            task_id,
            status='completed',
            progress=100,
            result=result,
            completed_at=datetime.now().isoformat()
        )
    
    def fail_task(self, task_id: str, error: str):
        """فشل المهمة"""
        self.update_task(
            task_id,
            status='failed',
            error=error,
            failed_at=datetime.now().isoformat()
        )

task_manager = TaskManager()

# Store progress for downloads (للتوافق مع النظام القديم)
download_progress = {}


# =============================================================================
# محسّن Whisper مع دعم كامل
# =============================================================================

class WhisperTranscriber:
    """محسّن تحويل الصوت إلى نص"""
    
    def __init__(self):
        self.model_cache = {}
    
    def transcribe(self, audio_file: str, model_size: str = 'base', 
                   language: str = 'auto') -> Dict:
        """تحويل الصوت إلى نص مع دعم Faster Whisper"""
        
        # محاولة Faster Whisper أولاً
        if FASTER_WHISPER_AVAILABLE:
            try:
                return self._transcribe_faster(audio_file, model_size, language)
            except Exception as e:
                logger.warning(f"Faster Whisper failed: {e}, falling back to standard")
        
        # استخدام Whisper العادي
        if WHISPER_AVAILABLE:
            return self._transcribe_standard(audio_file, model_size, language)
        
        raise Exception("لا توجد مكتبة متاحة لتحويل الصوت إلى نص")
    
    def _transcribe_faster(self, audio_file: str, model_size: str, language: str) -> Dict:
        """استخدام Faster Whisper"""
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        language_code = None if language == 'auto' else language
        
        segments, info = model.transcribe(
            audio_file,
            language=language_code,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        full_text = ""
        segments_list = []
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                full_text += text + " "
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': text,
                    'words': [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end
                        }
                        for word in getattr(segment, 'words', [])
                    ]
                })
        
        return {
            'text': full_text.strip(),
            'language': getattr(info, 'language', language),
            'segments': segments_list
        }
    
    def _transcribe_standard(self, audio_file: str, model_size: str, language: str) -> Dict:
        """استخدام Whisper العادي"""
        if model_size not in self.model_cache:
            self.model_cache[model_size] = whisper.load_model(model_size)
        
        model = self.model_cache[model_size]
        
        result = model.transcribe(
            audio_file,
            language=None if language == 'auto' else language,
            word_timestamps=True
        )
        
        return {
            'text': result['text'],
            'language': result.get('language', language),
            'segments': result.get('segments', [])
        }

whisper_transcriber = WhisperTranscriber()


# =============================================================================
# محسّن معالجة الترجمة
# =============================================================================

class SubtitleProcessor:
    """معالج الترجمة المحسّن"""
    
    @staticmethod
    def improve_sync_with_words(segments: List[Dict], translated_segments: List[Dict]) -> List[Dict]:
        """
        تحسين المزامنة باستخدام word-level timestamps
        يحسب التوقيتات بناءً على طول النص المترجم مقارنة بالنص الأصلي
        """
        improved_segments = []
        
        for i, orig_seg in enumerate(segments):
            if i >= len(translated_segments):
                break
            
            orig_text = orig_seg.get('text', '').strip()
            orig_start = float(orig_seg.get('start', 0))
            orig_end = float(orig_seg.get('end', orig_start + 3))
            orig_duration = orig_end - orig_start
            
            trans_seg = translated_segments[i]
            trans_text = trans_seg.get('text', '').strip()
            
            if not orig_text or not trans_text:
                continue
            
            # استخدام word-level timestamps إذا كانت متوفرة
            orig_words = orig_seg.get('words', [])
            
            if orig_words and len(orig_words) > 0:
                # حساب التوقيتات بناءً على الكلمات
                orig_word_count = len(orig_text.split())
                trans_word_count = len(trans_text.split())
                
                # إذا كان النص المترجم أطول، نمدد المدة قليلاً
                # إذا كان أقصر، نحافظ على المدة الأصلية
                if trans_word_count > 0 and orig_word_count > 0:
                    word_ratio = trans_word_count / orig_word_count
                    # تعديل المدة بناءً على النسبة (مع حد أقصى 1.5x)
                    adjusted_duration = orig_duration * min(1.5, max(0.5, word_ratio))
                    adjusted_end = orig_start + adjusted_duration
                else:
                    adjusted_end = orig_end
                
                improved_segments.append({
                    'start': orig_start,
                    'end': adjusted_end,
                    'text': trans_text
                })
            else:
                # بدون word timestamps، استخدام النسبة بناءً على عدد الأحرف
                orig_char_count = len(orig_text)
                trans_char_count = len(trans_text)
                
                if orig_char_count > 0 and trans_char_count > 0:
                    char_ratio = trans_char_count / orig_char_count
                    # تعديل المدة بناءً على النسبة (مع حد أقصى 1.5x)
                    adjusted_duration = orig_duration * min(1.5, max(0.5, char_ratio))
                    adjusted_end = orig_start + adjusted_duration
                else:
                    adjusted_end = orig_end
                
                improved_segments.append({
                    'start': orig_start,
                    'end': adjusted_end,
                    'text': trans_text
                })
        
        return improved_segments
    
    @staticmethod
    def smart_split_translation(original_segments: List[Dict], translated_text: str) -> List[Dict]:
        """
        تقسيم ذكي للترجمة مع الحفاظ على المزامنة
        يقسم النص المترجم بناءً على segments الأصلية مع تحسين التوقيتات
        """
        translated_segments = []
        
        # تقسيم النص المترجم بناءً على عدد segments الأصلية
        orig_total_words = sum(len(seg.get('text', '').split()) for seg in original_segments)
        trans_words = translated_text.split()
        
        if orig_total_words == 0 or len(trans_words) == 0:
            return translated_segments
        
        # حساب عدد الكلمات لكل segment
        word_index = 0
        for i, orig_seg in enumerate(original_segments):
            orig_text = orig_seg.get('text', '').strip()
            orig_start = float(orig_seg.get('start', 0))
            orig_end = float(orig_seg.get('end', orig_start + 3))
            orig_duration = orig_end - orig_start
            
            if not orig_text:
                continue
            
            orig_word_count = len(orig_text.split())
            
            # حساب عدد الكلمات المترجمة المقابلة
            # بناءً على نسبة الكلمات الإجمالية
            if orig_total_words > 0:
                segment_ratio = orig_word_count / orig_total_words
                trans_word_count = max(1, int(len(trans_words) * segment_ratio))
            else:
                trans_word_count = orig_word_count
            
            # أخذ الكلمات المترجمة المقابلة
            trans_segment_words = trans_words[word_index:word_index + trans_word_count]
            trans_segment_text = ' '.join(trans_segment_words).strip()
            
            if trans_segment_text:
                # تحسين التوقيت بناءً على طول النص المترجم
                orig_char_count = len(orig_text)
                trans_char_count = len(trans_segment_text)
                
                if orig_char_count > 0:
                    char_ratio = trans_char_count / orig_char_count
                    adjusted_duration = orig_duration * min(1.5, max(0.5, char_ratio))
                    adjusted_end = orig_start + adjusted_duration
                else:
                    adjusted_end = orig_end
                
                translated_segments.append({
                    'start': orig_start,
                    'end': adjusted_end,
                    'text': trans_segment_text
                })
                
                word_index += trans_word_count
        
        return translated_segments
    
    @staticmethod
    def split_long_segments(segments: List[Dict], max_duration: float = 5.0, 
                           max_chars: int = 80) -> List[Dict]:
        """تقسيم segments الطويلة بذكاء"""
        split_segments = []
        
        for segment in segments:
            start = float(segment.get('start', 0))
            end = float(segment.get('end', start + 3))
            text = segment.get('text', '').strip()
            
            if not text:
                continue
            
            duration = end - start
            
            # إذا كان segment قصيراً، استخدمه كما هو
            if duration <= max_duration and len(text) <= max_chars:
                split_segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                continue
            
            # تقسيم النص بذكاء
            words = text.split()
            if len(words) <= 3:
                split_segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                continue
            
            # تقسيم النص حسب الجمل
            sentences = re.split(r'([.!?،؛]\s*)', text)
            
            current_text = ""
            current_start = start
            time_per_char = duration / len(text) if len(text) > 0 else 0.1
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                potential_text = current_text + sentence
                
                if len(potential_text) > max_chars and current_text:
                    # حفظ segment الحالي
                    estimated_end = current_start + len(current_text) * time_per_char
                    split_segments.append({
                        'start': current_start,
                        'end': min(estimated_end, end),
                        'text': current_text.strip()
                    })
                    current_start = estimated_end
                    current_text = sentence
                else:
                    current_text = potential_text
            
            # إضافة النص المتبقي
            if current_text.strip():
                split_segments.append({
                    'start': current_start,
                    'end': end,
                    'text': current_text.strip()
                })
        
        return split_segments
    
    @staticmethod
    def create_srt(segments: List[Dict]) -> str:
        """إنشاء ملف SRT من segments مع تنظيف وترتيب ودعم كامل للعربية"""
        if not segments:
            return ""
        
        # تنظيف وترتيب segments
        cleaned_segments = SubtitleProcessor.clean_and_merge_segments(segments)
        
        srt_lines = []
        
        for i, seg in enumerate(cleaned_segments, 1):
            start = float(seg.get('start', 0))
            end = float(seg.get('end', start + 3))
            text = seg.get('text', '').strip()
            
            # التأكد من أن التوقيتات منطقية
            if end <= start:
                end = start + 1.0  # مدة دنيا ثانية واحدة
            
            # التأكد من أن النص غير فارغ
            if not text:
                continue
            
            start_str = SubtitleProcessor._format_time(start)
            end_str = SubtitleProcessor._format_time(end)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")
        
        # إرجاع SRT مع دعم UTF-8
        return '\n'.join(srt_lines)
    
    @staticmethod
    def save_srt_file(content: str, file_path: Path) -> bool:
        """حفظ ملف SRT بترميز UTF-8 مع BOM لدعم أفضل للعربية"""
        try:
            # إضافة BOM UTF-8 لضمان قراءة صحيحة في جميع المشغلات
            bom = '\ufeff'
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                f.write(bom + content)
            return True
        except Exception as e:
            logger.error(f"Error saving SRT file: {e}")
            # Fallback: حفظ بدون BOM
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e2:
                logger.error(f"Error saving SRT file (fallback): {e2}")
                return False
    
    @staticmethod
    def clean_and_merge_segments(segments: List[Dict]) -> List[Dict]:
        """
        تنظيف ودمج segments المتداخلة والمكررة
        يزيل التكرار ويضمن التوقيتات المنطقية
        """
        if not segments:
            return []
        
        # ترتيب segments حسب وقت البداية
        sorted_segments = sorted(segments, key=lambda x: float(x.get('start', 0)))
        
        cleaned = []
        last_end = 0.0
        
        for seg in sorted_segments:
            start = float(seg.get('start', 0))
            end = float(seg.get('end', start + 3))
            text = seg.get('text', '').strip()
            
            # تخطي segments فارغة
            if not text:
                continue
            
            # التأكد من أن التوقيتات منطقية
            if end <= start:
                end = start + max(1.0, len(text) * 0.1)  # مدة بناءً على طول النص
            
            # إذا كان segment متداخل مع السابق، دمجهما
            if cleaned and start < last_end:
                # دمج مع segment السابق
                prev_seg = cleaned[-1]
                prev_text = prev_seg.get('text', '').strip()
                
                # إذا كان النص مختلف، أضفه
                if text != prev_text:
                    # تمديد نهاية segment السابق قليلاً
                    prev_seg['end'] = min(start + 0.5, end)
                    # بدء segment جديد بعد السابق مباشرة
                    start = prev_seg['end'] + 0.1
                    cleaned.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })
                    last_end = end
            else:
                # إضافة segment جديد
                # التأكد من وجود فجوة صغيرة بين segments
                if cleaned and start < last_end + 0.1:
                    start = last_end + 0.1
                
                cleaned.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                last_end = end
        
        return cleaned
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """تنسيق الوقت لـ SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millisecs = int((secs % 1) * 1000)
        secs = int(secs)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    @staticmethod
    def get_arabic_font(font_family: str) -> str:
        """الحصول على خط عربي مناسب"""
        # قائمة بالخطوط العربية المدعومة بشكل جيد
        arabic_fonts = [
            'Arial', 'Tahoma', 'DejaVu Sans', 'Segoe UI', 'Noto Sans Arabic',
            'Cairo', 'Amiri', 'Scheherazade', 'Lateef', 'IBM Plex Sans Arabic'
        ]
        
        # إذا كان الخط المطلوب في القائمة، استخدمه
        if font_family in arabic_fonts:
            return font_family
        
        # محاولة اكتشاف خطوط عربية متاحة
        # استخدام Arial أو Tahoma كافتراضي (متوفران في معظم الأنظمة)
        return 'Arial'  # Arial يدعم العربية بشكل جيد
    
    @staticmethod
    def create_ass(srt_content: str, settings: Dict) -> str:
        """إنشاء ملف ASS مع إعدادات مخصصة ودعم كامل للعربية"""
        # تحويل جميع القيم إلى الأنواع الصحيحة
        font_size = int(settings.get('fontSize', settings.get('font_size', 24)))
        font_color = str(settings.get('fontColor', settings.get('font_color', '#FFFFFF')))
        bg_color = str(settings.get('bgColor', settings.get('bg_color', '#000000')))
        
        # التأكد من تحويل bg_opacity إلى int بشكل آمن
        bg_opacity_raw = settings.get('bgOpacity', settings.get('bg_opacity', 180))
        if isinstance(bg_opacity_raw, str):
            try:
                bg_opacity = int(bg_opacity_raw)
            except (ValueError, TypeError):
                bg_opacity = 180
        else:
            bg_opacity = int(bg_opacity_raw)
        
        position = str(settings.get('position', 'bottom'))
        font_family = str(settings.get('fontFamily', settings.get('font_name', 'Arial')))
        
        # استخدام خط عربي مناسب
        arabic_font = SubtitleProcessor.get_arabic_font(font_family)
        
        # التأكد من أن bg_opacity في النطاق الصحيح (0-255)
        bg_opacity = max(0, min(255, int(bg_opacity)))
        
        # تحويل ألوان RGB إلى BGR
        def rgb_to_bgr(hex_color):
            hex_color = str(hex_color).lstrip('#')
            if len(hex_color) == 6:
                r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                return "&H00" + b + g + r
            return "&H00FFFFFF"
        
        primary_color = rgb_to_bgr(font_color)
        bg_color_bgr = rgb_to_bgr(bg_color)
        
        alignment = {'top': '8', 'center': '5', 'bottom': '2'}.get(position, '2')
        margin_v = 10
        
        # تحضير back_color بشكل منفصل - استخدام format على int فقط
        bg_opacity_hex = format(bg_opacity, '02X')
        bg_color_part = bg_color_bgr[3:] if len(bg_color_bgr) > 3 else bg_color_bgr
        back_colour = "&H" + bg_opacity_hex + bg_color_part
        
        # إنشاء ASS header مع دعم UTF-8 للعربية
        # Encoding: 1 = UTF-8 (مهم للعربية)
        ass_header = """[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,""" + str(arabic_font) + "," + str(font_size) + "," + str(primary_color) + ",&H000000FF,&H00000000," + str(back_colour) + ",0,0,0,0,100,100,0,0,3,2,1," + str(alignment) + ",10,10," + str(margin_v) + """,1

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
                        start_ass = start.replace(',', '.')
                        end_ass = end.replace(',', '.')
                        
                        # استخدام string concatenation لتجنب مشاكل f-string
                        dialogue_line = "Dialogue: 0," + start_ass + "," + end_ass + ",Default,,0,0,0,," + text
                        events.append(dialogue_line)
                    
                    i += 4
                else:
                    i += 1
            else:
                i += 1
        
        return ass_header + '\n'.join(events)


# =============================================================================
# محسّن معالجة الفيديو
# =============================================================================

class VideoProcessor:
    """معالج الفيديو المحسّن"""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """التحقق من توفر ffmpeg"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            return True
        except:
            return False
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str) -> bool:
        """استخراج الصوت من الفيديو"""
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-threads', '0',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    @staticmethod
    def merge_subtitles(video_path: str, subtitle_path: str, 
                       output_path: str, settings: Dict) -> bool:
        """دمج الترجمة مع الفيديو - مع تحويل إجباري إلى H.264"""
        try:
            # التأكد من أن المسارات مطلقة
            video_path = str(Path(video_path).resolve())
            subtitle_path = str(Path(subtitle_path).resolve())
            output_path = str(Path(output_path).resolve())
            
            # تحويل الفيديو إلى H.264 قبل الدمج إذا لم يكن كذلك
            video_file = Path(video_path)
            if video_file.exists():
                try:
                    # التحقق من codec
                    probe_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-select_streams', 'v:0',
                        '-show_entries', 'stream=codec_name',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        str(video_file)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, timeout=10, text=True)
                    codec = probe_result.stdout.strip().lower()
                    
                    # إذا لم يكن H.264، قم بالتحويل أولاً
                    if codec and codec != 'h264':
                        logger.info(f"Converting source video to H.264 before merge: {codec} -> h264")
                        temp_h264 = video_file.parent / f"{video_file.stem}_temp_h264_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        convert_cmd = [
                            'ffmpeg',
                            '-i', str(video_file),
                            '-c:v', 'libx264',
                            '-profile:v', 'high',
                            '-level', '4.0',
                            '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac',
                            '-b:a', '128k',
                            '-movflags', '+faststart',
                            '-y',
                            str(temp_h264)
                        ]
                        logger.info(f"Converting video: {' '.join(convert_cmd)}")
                        convert_result = subprocess.run(convert_cmd, capture_output=True, timeout=300, text=True)
                        if convert_result.returncode == 0 and temp_h264.exists():
                            video_path = str(temp_h264)
                            logger.info(f"Source video converted to H.264: {temp_h264}")
                        else:
                            logger.warning(f"Video conversion failed or incomplete: {convert_result.stderr}")
                            # المتابعة مع الفيديو الأصلي
                    else:
                        logger.info(f"Video already H.264: {codec}")
                except Exception as e:
                    logger.warning(f"Could not check/convert source video codec: {e}, proceeding with original video")
            
            # استخدام SRT مباشرة إذا كان متوفراً، أو تحويله إلى ASS
            # SRT أفضل للترجمة الفورية لأنه أبسط وأكثر دقة
            use_srt_directly = True  # استخدام SRT مباشرة
            
            if subtitle_path.endswith('.srt'):
                if use_srt_directly:
                    # استخدام SRT مباشرة مع subtitles filter (أفضل للترجمة الفورية)
                    subtitle_path = str(Path(subtitle_path).resolve())
                else:
                    # تحويل إلى ASS للتحكم الكامل في التنسيق
                    with open(subtitle_path, 'r', encoding='utf-8') as f:
                        srt_content = f.read()
                    
                    ass_content = SubtitleProcessor.create_ass(srt_content, settings)
                    ass_path = subtitle_path.replace('.srt', '.ass')
                    
                    # حفظ ASS بترميز UTF-8 مع BOM لدعم العربية
                    with open(ass_path, 'w', encoding='utf-8-sig') as f:
                        f.write(ass_content)
                    
                    subtitle_path = str(Path(ass_path).resolve())
            
            # التأكد من وجود ملف الترجمة
            if not os.path.exists(subtitle_path):
                logger.error(f"Subtitle file does not exist: {subtitle_path}")
                return False
            
            # Escape المسار للـ ffmpeg بشكل صحيح
            import platform
            
            # استخدام المسار المطلق مباشرة
            subtitle_path_abs = str(Path(subtitle_path).resolve())
            
            # لـ ffmpeg filters، نحتاج إلى تهريب خاص للأحرف: : , [ ] \
            def escape_ffmpeg_path(path):
                """تهريب مسار لاستخدامه في ffmpeg filter"""
                # على Windows، تحويل backslashes إلى forward slashes أولاً
                if platform.system() == 'Windows':
                    # تحويل C:\path\to\file إلى C:/path/to/file
                    escaped = path.replace('\\', '/')
                else:
                    escaped = path
                
                # تهريب الأحرف الخاصة في ffmpeg filter syntax
                # : يجب أن يكون \: (مهم جداً في Windows paths مثل C:)
                # , يجب أن يكون \,
                # [ يجب أن يكون \[
                # ] يجب أن يكون \]
                # \ يجب أن يكون \\ (إذا بقي أي backslash)
                escaped = escaped.replace('\\', '\\\\')  # تهريب أي backslash متبقي
                escaped = escaped.replace(':', '\\:')
                escaped = escaped.replace(',', '\\,')
                escaped = escaped.replace('[', '\\[')
                escaped = escaped.replace(']', '\\]')
                return escaped
            
            subtitle_path_escaped = escape_ffmpeg_path(subtitle_path_abs)
            
            # دمج مع الفيديو
            # استخدام subtitles filter لـ SRT (أفضل للترجمة الفورية)
            # أو ass filter لـ ASS (للتحكم الكامل في التنسيق)
            if subtitle_path.endswith('.srt'):
                # استخدام subtitles filter لـ SRT
                vf_filter = f"subtitles={subtitle_path_escaped}"
            else:
                # استخدام ass filter لـ ASS
                vf_filter = f"ass={subtitle_path_escaped}"
            
            # محاولة استخدام ass filter أو subtitles filter مع دعم UTF-8 للعربية
            # إعدادات متوافقة مع جميع الأجهزة (MP4 H.264)
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', vf_filter,
                '-c:v', 'libx264',  # H.264 codec
                '-preset', 'medium',  # توازن بين السرعة والجودة
                '-crf', '23',  # جودة جيدة
                '-profile:v', 'high',  # High profile متوافق مع جميع الأجهزة
                '-level', '4.0',  # Level 4.0 متوافق مع معظم الأجهزة
                '-pix_fmt', 'yuv420p',  # متوافق مع جميع الأجهزة
                '-c:a', 'aac',  # AAC audio متوافق
                '-b:a', '128k',  # bitrate صوت جيد
                '-movflags', '+faststart',  # لضمان التشغيل السريع
                '-threads', '0',
                '-sub_charenc', 'UTF-8',  # تحديد ترميز الترجمة كـ UTF-8
                '-f', 'mp4',  # إجبار صيغة MP4
                '-y',
                output_path
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            logger.info(f"Video file: {video_path}, exists: {os.path.exists(video_path)}")
            logger.info(f"Subtitle file: {subtitle_path}, exists: {os.path.exists(subtitle_path)}")
            logger.info(f"Output file: {output_path}")
            logger.info(f"Video filter: {vf_filter}")
            
            # التأكد من وجود مجلد الإخراج
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # التأكد من أن ملف الفيديو موجود
            if not os.path.exists(video_path):
                logger.error(f"Video file does not exist: {video_path}")
                return False
            
            # التأكد من أن ملف الترجمة موجود
            if not os.path.exists(subtitle_path):
                logger.error(f"Subtitle file does not exist: {subtitle_path}")
                return False
            
            # تشغيل ffmpeg مع logging أفضل
            logger.info(f"Executing ffmpeg command...")
            process = subprocess.run(cmd, capture_output=True, timeout=600, text=True)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg failed with return code: {process.returncode}")
                logger.error(f"Full command: {' '.join(cmd)}")
                logger.error(f"Video path (exists: {os.path.exists(video_path)}): {video_path}")
                logger.error(f"Subtitle path (original, exists: {os.path.exists(subtitle_path)}): {subtitle_path}")
                logger.error(f"Subtitle path (absolute): {subtitle_path_abs}")
                logger.error(f"Subtitle path (escaped): {subtitle_path_escaped}")
                logger.error(f"Video filter: {vf_filter}")
                logger.error(f"Output path: {output_path}")
                if process.stderr:
                    logger.error(f"STDERR (first 3000 chars): {process.stderr[:3000]}")
                if process.stdout:
                    logger.error(f"STDOUT (first 3000 chars): {process.stdout[:3000]}")
                
                # إذا كان يستخدم subtitles filter وفشل، جرب ass filter
                if subtitle_path.endswith('.srt'):
                    logger.info("Trying with ass filter as fallback...")
                    # تحويل SRT إلى ASS
                    with open(subtitle_path, 'r', encoding='utf-8') as f:
                        srt_content = f.read()
                    
                    ass_content = SubtitleProcessor.create_ass(srt_content, settings)
                    ass_path = subtitle_path.replace('.srt', '.ass')
                    
                    # حفظ ASS بترميز UTF-8 مع BOM لدعم العربية
                    with open(ass_path, 'w', encoding='utf-8-sig') as f:
                        f.write(ass_content)
                    
                    # Escape المسار بشكل صحيح للـ ASS
                    ass_path_abs = str(Path(ass_path).resolve())
                    
                    # استخدام نفس دالة التهريب
                    def escape_ffmpeg_path(path):
                        """تهريب مسار لاستخدامه في ffmpeg filter"""
                        # على Windows، تحويل backslashes إلى forward slashes أولاً
                        if platform.system() == 'Windows':
                            escaped = path.replace('\\', '/')
                        else:
                            escaped = path
                        
                        # تهريب الأحرف الخاصة
                        escaped = escaped.replace('\\', '\\\\')
                        escaped = escaped.replace(':', '\\:')
                        escaped = escaped.replace(',', '\\,')
                        escaped = escaped.replace('[', '\\[')
                        escaped = escaped.replace(']', '\\]')
                        return escaped
                    
                    ass_path_escaped = escape_ffmpeg_path(ass_path_abs)
                    
                    vf_filter_alt = f"ass={ass_path_escaped}"
                    
                    cmd_alt = [
                        'ffmpeg',
                        '-i', video_path,
                        '-vf', vf_filter_alt,
                        '-c:v', 'libx264',  # H.264 codec
                        '-preset', 'medium',  # توازن بين السرعة والجودة
                        '-crf', '23',  # جودة جيدة
                        '-profile:v', 'high',  # High profile متوافق مع جميع الأجهزة
                        '-level', '4.0',  # Level 4.0 متوافق مع معظم الأجهزة
                        '-pix_fmt', 'yuv420p',  # متوافق مع جميع الأجهزة
                        '-c:a', 'aac',  # AAC audio متوافق
                        '-b:a', '128k',  # bitrate صوت جيد
                        '-movflags', '+faststart',  # لضمان التشغيل السريع
                        '-threads', '0',
                        '-sub_charenc', 'UTF-8',  # تحديد ترميز الترجمة كـ UTF-8
                        '-f', 'mp4',  # إجبار صيغة MP4
                        '-y',
                        output_path
                    ]
                    
                    logger.info(f"Trying alternative command with ASS filter...")
                    logger.info(f"ASS command: {' '.join(cmd_alt)}")
                    process_alt = subprocess.run(cmd_alt, capture_output=True, timeout=600, text=True)
                    
                    if process_alt.returncode != 0:
                        logger.error(f"ASS filter also failed (return code: {process_alt.returncode})")
                        logger.error(f"ASS filter: {vf_filter_alt}")
                        logger.error(f"ASS path (original): {ass_path_abs}")
                        logger.error(f"ASS path (escaped): {ass_path_escaped}")
                        logger.error(f"ASS file exists: {os.path.exists(ass_path_abs)}")
                        if process_alt.stderr:
                            logger.error(f"ASS STDERR (first 3000 chars): {process_alt.stderr[:3000]}")
                        if process_alt.stdout:
                            logger.error(f"ASS STDOUT (first 3000 chars): {process_alt.stdout[:3000]}")
                        return False
                    else:
                        if output_path.exists() and output_path.stat().st_size > 0:
                            logger.info(f"Successfully merged with ASS filter. Output size: {output_path.stat().st_size} bytes")
                            return True
                        else:
                            logger.error(f"ASS filter succeeded but output file not found or empty: {output_path}")
                            return False
                else:
                    logger.error("Not an SRT file, cannot try ASS fallback")
                    return False
            
            # التحقق من نجاح العملية
            if output_path.exists():
                file_size = output_path.stat().st_size
                if file_size > 0:
                    logger.info(f"Successfully merged subtitles using {'subtitles' if subtitle_path.endswith('.srt') else 'ass'} filter. Output size: {file_size} bytes")
                    return True
                else:
                    logger.error(f"Output file exists but is empty (0 bytes): {output_path}")
                    return False
            else:
                logger.error(f"FFmpeg succeeded (return code 0) but output file not found: {output_path}")
                logger.error(f"Output directory exists: {output_dir.exists()}")
                logger.error(f"Output directory is writable: {os.access(str(output_dir), os.W_OK)}")
                return False
            
        except Exception as e:
            logger.error(f"Subtitle merge failed: {e}")
            logger.error(traceback.format_exc())
            return False


# =============================================================================
# محسّن التحميل مع دعم JSON API
# =============================================================================

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
        """
        الحصول على جميع التنسيقات المتاحة للفيديو باستخدام JSON
        هذه النسخة المحسّنة تكتشف 4K/8K وجميع الجودات بدقة
        """
        try:
            # استخدام JSON API من yt-dlp (أدق وأسرع)
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                '--skip-download',
                url
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': 'Could not fetch formats'}
            
            # تحليل JSON
            video_info = json.loads(result.stdout)
            
            # استخراج التنسيقات بذكاء
            formats = self._parse_formats_from_json(video_info)
            
            # إنشاء presets ذكية
            presets = self._create_smart_presets_from_json(formats, video_info)
            
            # معلومات الفيديو
            info = {
                'title': video_info.get('title', 'Unknown'),
                'uploader': video_info.get('uploader', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'thumbnail': video_info.get('thumbnail', ''),
                'description': video_info.get('description', '')[:200],
                'view_count': video_info.get('view_count', 0),
                'like_count': video_info.get('like_count', 0)
            }
            
            return {
                'success': True,
                'formats': formats,
                'presets': presets,
                'info': info,
                'platform': self.detect_platform(url)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Request timeout'}
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {'success': False, 'error': 'Failed to parse video info'}
        except Exception as e:
            logger.error(f"Error fetching formats: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_formats_from_json(self, video_info: Dict) -> Dict:
        """
        تحليل التنسيقات من JSON - نسخة محسّنة ودقيقة
        تكتشف 8K, 4K, 1440p, 1080p, 720p, 480p, 360p تلقائياً
        """
        formats = {
            'video_audio': [],
            'video_only': [],
            'audio_only': [],
            'all_heights': set(),
            'max_height': 0,
            'by_height': {}
        }
        
        raw_formats = video_info.get('formats', [])
        
        for fmt in raw_formats:
            format_id = fmt.get('format_id', '')
            ext = fmt.get('ext', '')
            height = fmt.get('height')
            width = fmt.get('width')
            fps = fmt.get('fps', 0)
            vcodec = fmt.get('vcodec', 'none')
            acodec = fmt.get('acodec', 'none')
            filesize = fmt.get('filesize') or fmt.get('filesize_approx', 0)
            tbr = fmt.get('tbr', 0)
            vbr = fmt.get('vbr', 0)
            abr = fmt.get('abr', 0)
            
            format_info = {
                'id': format_id,
                'ext': ext,
                'height': height,
                'width': width,
                'fps': int(fps) if fps else None,
                'vcodec': vcodec,
                'acodec': acodec,
                'filesize': filesize,
                'filesize_mb': round(filesize / (1024 * 1024), 2) if filesize else None,
                'tbr': round(tbr, 2) if tbr else None,
                'vbr': round(vbr, 2) if vbr else None,
                'abr': round(abr, 2) if abr else None,
                'resolution': f"{height}p" if height else None,
                'quality': fmt.get('quality', 0),
                'format_note': fmt.get('format_note', ''),
                'protocol': fmt.get('protocol', ''),
                'container': fmt.get('container', ''),
                'has_drm': fmt.get('has_drm', False)
            }
            
            # تصنيف التنسيق
            if vcodec != 'none' and acodec != 'none':
                format_info['type'] = 'video_audio'
                format_info['note'] = f"{height}p" if height else 'Video+Audio'
                formats['video_audio'].append(format_info)
                
                if height:
                    formats['all_heights'].add(height)
                    formats['max_height'] = max(formats['max_height'], height)
                    
                    if height not in formats['by_height']:
                        formats['by_height'][height] = []
                    formats['by_height'][height].append(format_info)
                    
            elif vcodec != 'none' and acodec == 'none':
                format_info['type'] = 'video_only'
                format_info['note'] = f"{height}p (No Audio)" if height else 'Video Only'
                formats['video_only'].append(format_info)
                
                if height:
                    formats['all_heights'].add(height)
                    formats['max_height'] = max(formats['max_height'], height)
                    
                    if height not in formats['by_height']:
                        formats['by_height'][height] = []
                    formats['by_height'][height].append(format_info)
                    
            elif vcodec == 'none' and acodec != 'none':
                format_info['type'] = 'audio_only'
                format_info['bitrate'] = f"{int(abr)}kbps" if abr else 'Unknown'
                format_info['note'] = f"Audio {int(abr)}kbps" if abr else 'Audio'
                formats['audio_only'].append(format_info)
        
        formats['all_heights'] = sorted(list(formats['all_heights']), reverse=True)
        
        formats['video_audio'].sort(
            key=lambda x: (
                x['height'] or 0, 
                x['fps'] or 0,
                x['tbr'] or 0
            ), 
            reverse=True
        )
        
        formats['video_only'].sort(
            key=lambda x: (
                x['height'] or 0,
                x['fps'] or 0,
                x['vbr'] or 0
            ), 
            reverse=True
        )
        
        formats['audio_only'].sort(
            key=lambda x: x['abr'] or 0, 
            reverse=True
        )
        
        return formats
    
    def _create_smart_presets_from_json(self, formats: Dict, video_info: Dict) -> List[Dict]:
        """
        إنشاء presets ذكية بناءً على التنسيقات المتاحة من JSON
        يكتشف تلقائياً 8K, 4K, 1440p, 1080p, وجميع الجودات المتاحة
        """
        presets = []
        all_heights = formats.get('all_heights', [])
        max_height = formats.get('max_height', 0)
        by_height = formats.get('by_height', {})
        
        # 1. أفضل جودة (دائماً موجود)
        best_description = f'أعلى جودة متاحة ({max_height}p)' if max_height else 'أعلى جودة متاحة'
        
        if max_height and max_height in by_height:
            best_formats = by_height[max_height]
            if best_formats:
                best_fmt = best_formats[0]
                fps_info = f" @ {best_fmt['fps']}fps" if best_fmt.get('fps') else ""
                size_info = f" • {best_fmt['filesize_mb']} MB" if best_fmt.get('filesize_mb') else ""
                best_description = f"{max_height}p{fps_info}{size_info}"
        
        presets.append({
            'id': 'best',
            'name': '⭐ أفضل جودة',
            'description': best_description,
            'icon': 'crown',
            'command': 'bestvideo+bestaudio/best',
            'height': max_height,
            'priority': 100
        })
        
        # 2. كشف الجودات المتاحة بذكاء
        quality_definitions = [
            {
                'height': 4320, 
                'id': '8k', 
                'name': '8K Ultra HD', 
                'description': '4320p - جودة خيالية 🔥', 
                'icon': 'sparkles',
                'emoji': '🎆'
            },
            {
                'height': 2160, 
                'id': '4k', 
                'name': '4K Ultra HD', 
                'description': '2160p - جودة فائقة', 
                'icon': 'gem',
                'emoji': '💎'
            },
            {
                'height': 1440, 
                'id': '1440p', 
                'name': '1440p QHD', 
                'description': 'جودة عالية جداً', 
                'icon': 'star',
                'emoji': '⭐'
            },
            {
                'height': 1080, 
                'id': '1080p', 
                'name': '1080p Full HD', 
                'description': 'جودة ممتازة', 
                'icon': 'video',
                'emoji': '📺'
            },
            {
                'height': 720, 
                'id': '720p', 
                'name': '720p HD', 
                'description': 'جودة جيدة - حجم متوازن', 
                'icon': 'film',
                'emoji': '📹'
            },
            {
                'height': 480, 
                'id': '480p', 
                'name': '480p SD', 
                'description': 'جودة متوسطة - حجم صغير', 
                'icon': 'smartphone',
                'emoji': '📱'
            },
            {
                'height': 360, 
                'id': '360p', 
                'name': '360p Low', 
                'description': 'جودة منخفضة - سريع', 
                'icon': 'phone',
                'emoji': '📵'
            }
        ]
        
        for quality_def in quality_definitions:
            height = quality_def['height']
            
            if height in all_heights:
                additional_info = ""
                if height in by_height and by_height[height]:
                    best_of_height = by_height[height][0]
                    
                    if best_of_height.get('fps'):
                        additional_info += f" @ {best_of_height['fps']}fps"
                    
                    if best_of_height.get('filesize_mb'):
                        additional_info += f" • ~{best_of_height['filesize_mb']} MB"
                    
                    if best_of_height.get('tbr'):
                        additional_info += f" • {best_of_height['tbr']} kbps"
                
                description = quality_def['description']
                if additional_info:
                    description += additional_info
                
                presets.append({
                    'id': quality_def['id'],
                    'name': f"{quality_def['emoji']} {quality_def['name']}",
                    'description': description,
                    'icon': quality_def['icon'],
                    'command': f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
                    'height': height,
                    'priority': 90 - (len(presets) * 5)
                })
        
        # 3. صوت فقط
        if formats.get('audio_only'):
            best_audio = formats['audio_only'][0]
            bitrate = best_audio.get('bitrate', 'Unknown')
            filesize_mb = best_audio.get('filesize_mb')
            
            audio_description = f'MP3 بأفضل جودة'
            if bitrate != 'Unknown':
                audio_description += f' ({bitrate})'
            if filesize_mb:
                audio_description += f' • ~{filesize_mb} MB'
            
            presets.append({
                'id': 'audio',
                'name': '🎵 صوت فقط',
                'description': audio_description,
                'icon': 'music',
                'command': 'bestaudio/bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
                'priority': 50
            })
        
        presets.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        logger.info(f"Created {len(presets)} smart presets. Heights available: {all_heights}")
        
        return presets
    
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
                # البحث عن الملف المحمّل
                download_folder = Path(self.output_dir)
                video_files = []
                for file in download_folder.iterdir():
                    if file.is_file():
                        ext = file.suffix.lower()
                        if ext in ['.mp4', '.webm', '.mkv', '.mp3', '.m4a', '.mov', '.avi', '.flv']:
                            video_files.append((file, file.stat().st_mtime))
                
                downloaded_file = None
                if video_files:
                    video_files.sort(key=lambda x: x[1], reverse=True)
                    downloaded_file = str(video_files[0][0])
                    
                    # تحويل إجباري إلى MP4 H.264 لضمان التوافق
                    if downloaded_file and not downloaded_file.endswith('.mp3') and not downloaded_file.endswith('.m4a'):
                        logger.info(f"Ensuring H.264 encoding for: {downloaded_file}")
                        downloaded_file = self._ensure_mp4_h264(downloaded_file)
                
                download_progress[download_id] = {
                    'status': 'completed',
                    'percent': '100%',
                    'message': 'تم التحميل بنجاح!',
                    'file': downloaded_file
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
                # إجبار MP4 H.264 متوافق مع جميع الأجهزة
                # استخدام format selector يفضل H.264
                # وإجبار إعادة الترميز إلى H.264
                cmd.extend([
                    '-f', format_cmd,
                    '--merge-output-format', 'mp4',
                    '--recode-video', 'mp4',  # إجبار إعادة الترميز إلى MP4
                    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -strict experimental'
                ])
            
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
                # إجبار MP4 H.264 متوافق مع جميع الأجهزة
                cmd.extend([
                    '-f', format_cmd,
                    '--merge-output-format', 'mp4',
                    '--recode-video', 'mp4',  # إجبار إعادة الترميز إلى MP4
                    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -strict experimental'
                ])
            
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
                # إجبار MP4 H.264 متوافق مع جميع الأجهزة
                cmd.extend([
                    '-f', 'best[ext=mp4]/best',
                    '--merge-output-format', 'mp4',
                    '--recode-video', 'mp4',  # إجبار إعادة الترميز إلى MP4
                    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -strict experimental'
                ])
            
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
            
            cmd = ['yt-dlp']
            
            if is_audio:
                cmd.extend(['-x', '--audio-format', 'mp3'])
            else:
                # إجبار MP4 H.264 متوافق مع جميع الأجهزة
                cmd.extend([
                    '--merge-output-format', 'mp4',
                    '--recode-video', 'mp4',  # إجبار إعادة الترميز إلى MP4
                    '--postprocessor-args', 'ffmpeg:-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart -strict experimental'
                ])
            
            cmd.extend(['-o', output_template, '--no-warnings', url])
            
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
                                'percent': part,
                                'method': download_progress.get(download_id, {}).get('method', '')
                            }
                            break
            
            process.wait()
            return process.returncode == 0
            
        except Exception as e:
            logger.error(f"Monitor failed: {e}")
            return False
    
    def _ensure_mp4_h264(self, video_path: str) -> str:
        """تحويل الفيديو إلى MP4 H.264 متوافق مع جميع الأجهزة - إجباري دائماً"""
        try:
            video_file = Path(video_path)
            
            # إنشاء اسم ملف جديد
            output_file = video_file.parent / f"{video_file.stem}_h264.mp4"
            
            # التحقق من codec الحالي
            needs_conversion = True
            try:
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,codec_type',
                    '-of', 'json',
                    str(video_file)
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, timeout=10, text=True)
                if probe_result.returncode == 0:
                    import json
                    probe_data = json.loads(probe_result.stdout)
                    streams = probe_data.get('streams', [])
                    for stream in streams:
                        if stream.get('codec_type') == 'video':
                            codec = stream.get('codec_name', '').lower()
                            # إذا كان H.264 بالفعل و MP4، استخدم الملف الأصلي
                            if codec == 'h264' and video_file.suffix.lower() == '.mp4':
                                logger.info(f"Video already H.264 MP4, skipping conversion: {video_file}")
                                return str(video_file)
                            break
            except Exception as e:
                logger.debug(f"Could not probe codec: {e}, will convert anyway")
            
            # تحويل إجباري إلى MP4 H.264
            logger.info(f"Converting video to H.264 MP4: {video_file} -> {output_file}")
            cmd = [
                'ffmpeg',
                '-i', str(video_file),
                '-c:v', 'libx264',  # H.264 codec إجباري
                '-profile:v', 'high',  # High profile متوافق
                '-level', '4.0',  # Level 4.0 متوافق
                '-pix_fmt', 'yuv420p',  # متوافق مع جميع الأجهزة
                '-c:a', 'aac',  # AAC audio
                '-b:a', '128k',  # bitrate صوت
                '-movflags', '+faststart',  # لضمان التشغيل السريع
                '-y',
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=600, text=True)
            
            if result.returncode == 0 and output_file.exists():
                # حذف الملف الأصلي دائماً بعد التحويل الناجح
                if output_file != video_file:
                    try:
                        video_file.unlink()
                        logger.info(f"Deleted original file: {video_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete original: {e}")
                logger.info(f"Successfully converted to MP4 H.264: {output_file}")
                return str(output_file)
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                # إذا فشل التحويل، حاول استخدام الملف الأصلي
                return str(video_file)
                
        except Exception as e:
            logger.error(f"Error converting video: {e}")
            logger.error(traceback.format_exc())
            return str(video_path)  # إرجاع الملف الأصلي في حالة الفشل

# Initialize downloader
downloader = SmartMediaDownloader()


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')


@app.route('/api/instant-translate', methods=['POST'])
def api_instant_translate():
    """API الترجمة الفورية المحسّنة"""
    try:
        data = request.json
        step = data.get('step')
        
        if step == 'download':
            url = data.get('url')
            quality = data.get('quality', '720p')
            
            if not url:
                return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
            
            # استخدام yt-dlp مباشرة
            download_folder = Path(app.config['DOWNLOAD_FOLDER'])
            
            # تحديد format command من quality
            if quality == 'best':
                format_cmd = 'bestvideo+bestaudio/best'
            elif quality == '720p':
                format_cmd = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
            elif quality == '480p':
                format_cmd = 'bestvideo[height<=480]+bestaudio/best[height<=480]'
            elif quality == '1080p':
                format_cmd = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
            elif quality == '4k' or quality == '2160p':
                format_cmd = 'bestvideo[height<=2160]+bestaudio/best[height<=2160]'
            else:
                format_cmd = quality
            
            # إعدادات yt-dlp مع تحويل إجباري إلى MP4 H.264 متوافق
            ydl_opts = {
                'format': format_cmd,
                'outtmpl': str(download_folder / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'merge_output_format': 'mp4',
                'recode_video': 'mp4',  # إجبار إعادة الترميز إلى MP4
                'postprocessor_args': {
                    'ffmpeg': [
                        '-c:v', 'libx264',
                        '-profile:v', 'high',
                        '-level', '4.0',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-movflags', '+faststart',
                        '-strict', 'experimental'
                    ]
                }
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                
                # التأكد من أن الملف موجود
                if not os.path.exists(filename):
                    # البحث عن أحدث ملف
                    video_files = []
                    for file in download_folder.iterdir():
                        if file.is_file():
                            ext = file.suffix.lower()
                            if ext in ['.mp4', '.webm', '.mkv', '.mov', '.avi', '.flv']:
                                video_files.append((file, file.stat().st_mtime))
                    
                    if video_files:
                        video_files.sort(key=lambda x: x[1], reverse=True)
                        filename = str(video_files[0][0])
                
                if not os.path.exists(filename):
                    return jsonify({
                        'success': False,
                        'message': 'تم التحميل لكن الملف غير موجود'
                    }), 400
                
                # تحويل إجباري إلى MP4 H.264
                if not filename.endswith('.mp3') and not filename.endswith('.m4a'):
                    logger.info(f"Converting downloaded video to H.264: {filename}")
                    converted_file = downloader._ensure_mp4_h264(filename)
                    if converted_file != filename:
                        filename = converted_file
                
                # استخدام ملف مؤقت
                temp_file = download_folder / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(temp_file, 'w') as f:
                    f.write(filename)
                
                return jsonify({
                    'success': True,
                    'file': filename,
                    'temp_file': temp_file.name
                })
                
            except Exception as e:
                logger.error(f"Download error: {e}")
                return jsonify({
                    'success': False,
                    'message': f'خطأ في التحميل: {str(e)}'
                }), 500
        
        elif step == 'extract_audio':
            video_file = data.get('video_file')
            
            # قراءة من الملف المؤقت إذا لزم الأمر
            if not video_file and data.get('temp_video_file'):
                temp_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_video_file']
                if temp_path.exists():
                    with open(temp_path, 'r') as f:
                        video_file = f.read().strip()
            
            if not video_file or not os.path.exists(video_file):
                return jsonify({
                    'success': False,
                    'message': 'ملف الفيديو غير موجود'
                }), 400
            
            audio_file = video_file.rsplit('.', 1)[0] + '_audio.wav'
            
            success = VideoProcessor.extract_audio(video_file, audio_file)
            
            if success:
                temp_file = Path(app.config['DOWNLOAD_FOLDER']) / f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(temp_file, 'w') as f:
                    f.write(audio_file)
                
                return jsonify({
                    'success': True,
                    'audio_file': audio_file,
                    'temp_file': temp_file.name
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'فشل استخراج الصوت'
                }), 500
        
        elif step == 'transcribe':
            audio_file = data.get('audio_file')
            model = data.get('model', 'base')
            language = data.get('language', 'auto')
            
            # قراءة من الملف المؤقت إذا لزم الأمر
            if not audio_file and data.get('temp_audio_file'):
                temp_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_audio_file']
                if temp_path.exists():
                    with open(temp_path, 'r') as f:
                        audio_file = f.read().strip()
            
            if not audio_file or not os.path.exists(audio_file):
                return jsonify({
                    'success': False,
                    'message': 'ملف الصوت غير موجود'
                }), 400
            
            result = whisper_transcriber.transcribe(audio_file, model, language)
            
            # حفظ في ملف مؤقت (JSON للحفاظ على segments مع التوقيتات)
            temp_file = Path(app.config['DOWNLOAD_FOLDER']) / f"temp_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # إرجاع النتيجة مع segments للتأكد من استخدام التوقيتات
            return jsonify({
                'success': True,
                'temp_file': temp_file.name,
                'text': result.get('text', ''),
                'language': result.get('language', language),
                'segments': result.get('segments', []),
                **result
            })
        
        elif step == 'translate':
            text = data.get('text')
            source_lang = data.get('source_lang', 'auto')
            segments = data.get('segments')  # دعم ترجمة segments مع التوقيتات
            
            if not text:
                return jsonify({
                    'success': False,
                    'message': 'لا يوجد نص للترجمة'
                }), 400
            
            if not TRANSLATOR_AVAILABLE:
                return jsonify({
                    'success': False,
                    'message': 'المترجم غير متوفر'
                }), 503
            
            translator = GoogleTranslator(source=source_lang, target='ar')
            
            # إذا كانت هناك segments مع توقيتات، ترجم كل segment بشكل منفصل
            if segments and isinstance(segments, list):
                translated_segments = []
                for seg in segments:
                    seg_text = seg.get('text', '').strip()
                    if seg_text:
                        try:
                            translated_text = translator.translate(seg_text)
                            translated_segments.append({
                                'start': seg.get('start', 0),
                                'end': seg.get('end', 0),
                                'text': translated_text
                            })
                        except Exception as e:
                            logger.warning(f"Translation failed for segment: {e}")
                            translated_segments.append({
                                'start': seg.get('start', 0),
                                'end': seg.get('end', 0),
                                'text': seg_text
                            })
                
                # حفظ segments المترجمة
                temp_file = Path(app.config['DOWNLOAD_FOLDER']) / f"temp_translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump({'translated_segments': translated_segments, 'text': ' '.join([s['text'] for s in translated_segments])}, f, ensure_ascii=False, indent=2)
                
                return jsonify({
                    'success': True,
                    'translated_text': ' '.join([s['text'] for s in translated_segments]),
                    'translated_segments': translated_segments,
                    'temp_file': temp_file.name
                })
            else:
                # ترجمة نص عادي
                translated = translator.translate(text)
                
                # حفظ في ملف مؤقت
                temp_file = Path(app.config['DOWNLOAD_FOLDER']) / f"temp_translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(translated)
                
                return jsonify({
                    'success': True,
                    'translated_text': translated,
                    'temp_file': temp_file.name
                })
        
        elif step == 'merge':
            logger.info(f"Merge step called with data keys: {list(data.keys())}")
            
            video_file = data.get('video_file')
            # دعم كلا الاسمين: subtitle_text و translated_text
            subtitle_text = data.get('subtitle_text') or data.get('translated_text')
            
            logger.info(f"Initial values: video_file={bool(video_file)}, subtitle_text={bool(subtitle_text)}")
            
            # قراءة من الملف المؤقت إذا لزم الأمر
            if not video_file and data.get('temp_video_file'):
                temp_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_video_file']
                logger.info(f"Trying to read video from temp file: {temp_path}")
                if temp_path.exists():
                    with open(temp_path, 'r') as f:
                        video_file = f.read().strip()
                        logger.info(f"Read video_file from temp: {video_file}")
                else:
                    logger.warning(f"Temp video file not found: {temp_path}")
            
            if not subtitle_text and data.get('temp_translated_file'):
                temp_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_translated_file']
                logger.info(f"Trying to read subtitle from temp file: {temp_path}")
                if temp_path.exists():
                    try:
                        # محاولة قراءة كـ JSON أولاً
                        if temp_path.suffix == '.json':
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                temp_data = json.load(f)
                                # محاولة الحصول على النص المترجم
                                subtitle_text = temp_data.get('translated_text') or temp_data.get('text', '')
                                if not subtitle_text and temp_data.get('translated_segments'):
                                    # بناء النص من segments
                                    subtitle_text = ' '.join([s.get('text', '') for s in temp_data['translated_segments']])
                                logger.info(f"Read subtitle_text from JSON temp, length: {len(subtitle_text)}")
                        else:
                            # قراءة كـ نص عادي
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                subtitle_text = f.read().strip()
                                logger.info(f"Read subtitle_text from temp, length: {len(subtitle_text)}")
                    except Exception as e:
                        logger.error(f"Error reading temp translated file: {e}")
                        subtitle_text = None
                else:
                    logger.warning(f"Temp translated file not found: {temp_path}")
            
            if not video_file or not subtitle_text:
                logger.error(f"Merge error: video_file={video_file}, subtitle_text={'exists' if subtitle_text else 'missing'}")
                logger.error(f"Data received: {json.dumps({k: str(v)[:100] if isinstance(v, str) else v for k, v in data.items()}, ensure_ascii=False)}")
                return jsonify({
                    'success': False,
                    'message': 'ملف الفيديو أو النص غير موجود',
                    'debug': {
                        'has_video_file': bool(video_file),
                        'has_subtitle_text': bool(subtitle_text),
                        'video_file': video_file if video_file else None,
                        'received_keys': list(data.keys()),
                        'temp_video_file': data.get('temp_video_file'),
                        'temp_translated_file': data.get('temp_translated_file')
                    }
                }), 400
            
            # التأكد من أن ملف الفيديو موجود
            if not os.path.exists(video_file):
                # البحث في مجلد التحميل
                download_folder = Path(app.config['DOWNLOAD_FOLDER'])
                basename = os.path.basename(video_file)
                possible_paths = [
                    download_folder / basename,
                    Path(video_file)
                ]
                
                for path in possible_paths:
                    if path.exists():
                        video_file = str(path)
                        break
                else:
                    return jsonify({
                        'success': False,
                        'message': f'ملف الفيديو غير موجود: {video_file}'
                    }), 400
            
            # التأكد من تحويل جميع القيم إلى الأنواع الصحيحة
            settings = {
                'fontSize': int(data.get('fontSize', data.get('font_size', 24))),
                'fontColor': str(data.get('fontColor', data.get('font_color', '#FFFFFF'))),
                'bgColor': str(data.get('bgColor', data.get('bg_color', '#000000'))),
                'bgOpacity': int(data.get('bgOpacity', data.get('bg_opacity', 180))),
                'position': str(data.get('position', 'bottom')),
                'fontFamily': str(data.get('fontFamily', data.get('font_name', 'Arial')))
            }
            
            # إنشاء SRT من النص المترجم مع التوقيتات الصحيحة
            # إذا كان النص ليس بصيغة SRT، نحوله إلى SRT
            if not subtitle_text.strip().startswith('1\n') and not subtitle_text.strip().startswith('WEBVTT'):
                # محاولة استخدام segments المترجمة من ملف JSON إذا كانت متوفرة
                translated_segments_data = None
                if data.get('temp_translated_file'):
                    temp_translated_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_translated_file']
                    if temp_translated_path.exists() and temp_translated_path.suffix == '.json':
                        try:
                            with open(temp_translated_path, 'r', encoding='utf-8') as f:
                                translated_segments_data = json.load(f)
                        except:
                            pass
                
                # محاولة استخدام segments من transcript إذا كانت متوفرة
                transcript_data = None
                if data.get('temp_transcript_file'):
                    temp_transcript_path = Path(app.config['DOWNLOAD_FOLDER']) / data['temp_transcript_file']
                    if temp_transcript_path.exists():
                        try:
                            with open(temp_transcript_path, 'r', encoding='utf-8') as f:
                                transcript_data = json.load(f)
                        except:
                            pass
                
                # الحالة المثلى: استخدام segments المترجمة مع التوقيتات
                if translated_segments_data and translated_segments_data.get('translated_segments'):
                    translated_segments = translated_segments_data['translated_segments']
                    
                    # إذا كانت هناك segments أصلية، نحسن المزامنة
                    if transcript_data and transcript_data.get('segments'):
                        original_segments = transcript_data['segments']
                        # تحسين المزامنة باستخدام word-level timestamps
                        improved_segments = SubtitleProcessor.improve_sync_with_words(
                            original_segments, 
                            translated_segments
                        )
                    else:
                        improved_segments = translated_segments
                    
                    # تنظيف وترتيب segments قبل إنشاء SRT
                    cleaned_segments = SubtitleProcessor.clean_and_merge_segments(improved_segments)
                    
                    # إنشاء SRT من segments المنظفة
                    if cleaned_segments:
                        subtitle_text = SubtitleProcessor.create_srt(cleaned_segments)
                    else:
                        subtitle_text = ""
                
                # الحالة الثانية: استخدام segments من transcript مع النص المترجم
                elif transcript_data and transcript_data.get('segments'):
                    segments = transcript_data['segments']
                    translated_text = subtitle_text.strip()
                    
                    # استخدام التقسيم الذكي للترجمة مع تحسين المزامنة
                    translated_segments = SubtitleProcessor.smart_split_translation(segments, translated_text)
                    
                    if translated_segments:
                        # تحسين المزامنة باستخدام word-level timestamps إذا كانت متوفرة
                        improved_segments = SubtitleProcessor.improve_sync_with_words(segments, translated_segments)
                        
                        # تنظيف وترتيب segments قبل إنشاء SRT
                        cleaned_segments = SubtitleProcessor.clean_and_merge_segments(improved_segments)
                        
                        # إنشاء SRT من segments المنظفة
                        if cleaned_segments:
                            subtitle_text = SubtitleProcessor.create_srt(cleaned_segments)
                        else:
                            subtitle_text = ""
                    else:
                        # Fallback: تقسيم بسيط
                        if '\n' in translated_text:
                            translated_lines = [line.strip() for line in translated_text.split('\n') if line.strip()]
                        else:
                            words = translated_text.split()
                            if len(words) > 0 and len(segments) > 0:
                                words_per_segment = max(1, len(words) // len(segments))
                                translated_lines = []
                                for i in range(0, len(words), words_per_segment):
                                    translated_lines.append(' '.join(words[i:i+words_per_segment]))
                            else:
                                translated_lines = [translated_text]
                        
                        srt_lines = []
                        for i, seg in enumerate(segments):
                            start = float(seg.get('start', 0))
                            end = float(seg.get('end', start + 3))
                            
                            if i < len(translated_lines):
                                text = translated_lines[i]
                            else:
                                text = seg.get('text', '').strip()
                            
                            if text:
                                start_str = SubtitleProcessor._format_time(start)
                                end_str = SubtitleProcessor._format_time(end)
                                
                                srt_lines.append(f"{len(srt_lines) // 4 + 1}")
                                srt_lines.append(f"{start_str} --> {end_str}")
                                srt_lines.append(text)
                                srt_lines.append("")
                        
                        if srt_lines:
                            subtitle_text = '\n'.join(srt_lines)
                else:
                    # Fallback: تحويل النص العادي إلى SRT بسيط
                    lines = subtitle_text.split('\n')
                    srt_lines = []
                    for i, line in enumerate(lines, 1):
                        if line.strip():
                            start_time = (i - 1) * 3
                            end_time = i * 3
                            start_str = SubtitleProcessor._format_time(start_time)
                            end_str = SubtitleProcessor._format_time(end_time)
                            
                            srt_lines.append(f"{i}")
                            srt_lines.append(f"{start_str} --> {end_str}")
                            srt_lines.append(line.strip())
                            srt_lines.append("")
                    
                    subtitle_text = '\n'.join(srt_lines)
            
            # التأكد من أن subtitle_text غير فارغ
            if not subtitle_text or not subtitle_text.strip():
                logger.error("Subtitle text is empty!")
                return jsonify({
                    'success': False,
                    'message': 'نص الترجمة فارغ. يرجى المحاولة مرة أخرى.'
                }), 400
            
            # إنشاء ملف SRT نظيف ومنظم
            # التأكد من أن subtitle_text هو SRT صحيح
            if not subtitle_text.strip().startswith('1\n') and not subtitle_text.strip().startswith('WEBVTT'):
                # إذا لم يكن SRT، تحويله
                if subtitle_text.strip():
                    # تقسيم إلى أسطر وإنشاء SRT بسيط
                    lines = [line.strip() for line in subtitle_text.split('\n') if line.strip()]
                    if lines:
                        srt_lines = []
                        for i, line in enumerate(lines, 1):
                            start_time = (i - 1) * 3
                            end_time = i * 3
                            start_str = SubtitleProcessor._format_time(start_time)
                            end_str = SubtitleProcessor._format_time(end_time)
                            
                            srt_lines.append(f"{i}")
                            srt_lines.append(f"{start_str} --> {end_str}")
                            srt_lines.append(line)
                            srt_lines.append("")
                        subtitle_text = '\n'.join(srt_lines)
            
            # التأكد مرة أخرى من أن subtitle_text غير فارغ بعد المعالجة
            if not subtitle_text or not subtitle_text.strip():
                logger.error("Subtitle text is empty after processing!")
                return jsonify({
                    'success': False,
                    'message': 'فشل إنشاء ملف الترجمة. يرجى المحاولة مرة أخرى.'
                }), 400
            
            # إنشاء ملف SRT بترميز UTF-8 مع BOM لدعم أفضل للعربية
            srt_path = Path(app.config['SUBTITLE_FOLDER']) / f"subtitle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            save_success = SubtitleProcessor.save_srt_file(subtitle_text, srt_path)
            
            if not save_success:
                logger.error(f"Failed to save SRT file: {srt_path}")
                return jsonify({
                    'success': False,
                    'message': 'فشل حفظ ملف الترجمة'
                }), 500
            
            if not srt_path.exists():
                logger.error(f"SRT file was not created: {srt_path}")
                return jsonify({
                    'success': False,
                    'message': 'فشل إنشاء ملف الترجمة'
                }), 500
            
            segment_count = len([s for s in subtitle_text.split('\n\n') if s.strip()])
            logger.info(f"Created SRT file: {srt_path} with {segment_count} segments (UTF-8 with BOM)")
            
            # التأكد من أن ملف الفيديو موجود وقابل للقراءة
            if not os.path.exists(video_file):
                logger.error(f"Video file does not exist: {video_file}")
                return jsonify({
                    'success': False,
                    'message': f'ملف الفيديو غير موجود: {video_file}'
                }), 400
            
            # دمج
            output_file = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path = Path(app.config['OUTPUT_FOLDER']) / output_file
            
            logger.info(f"Merging subtitles: video={video_file}, srt={srt_path}, output={output_path}")
            logger.info(f"Video file exists: {os.path.exists(video_file)}, SRT file exists: {srt_path.exists()}")
            
            try:
                success = VideoProcessor.merge_subtitles(str(video_file), str(srt_path), str(output_path), settings)
                
                if success:
                    if not output_path.exists():
                        logger.error(f"Output file was not created: {output_path}")
                        return jsonify({
                            'success': False,
                            'message': 'فشل إنشاء الفيديو النهائي'
                        }), 500
                    
                    logger.info(f"Successfully merged subtitles. Output: {output_path}")
                    return jsonify({
                        'success': True,
                        'download_url': f'/download/{output_file}'
                    })
                else:
                    logger.error("merge_subtitles returned False")
                    return jsonify({
                        'success': False,
                        'message': 'فشل دمج الترجمة مع الفيديو'
                    }), 500
            except Exception as e:
                logger.error(f"Error in merge_subtitles: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'message': f'خطأ في دمج الترجمة: {str(e)}'
                }), 500
        
        return jsonify({'success': False, 'message': 'خطوة غير صحيحة'})
    
    except Exception as e:
        logger.error(f"API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get-qualities', methods=['POST'])
def api_get_qualities():
    """الحصول على الجودات المتاحة للفيديو"""
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        result = downloader.get_available_formats(url)
        
        if result.get('success'):
            formats_data = result.get('formats', {})
            presets = result.get('presets', [])
            
            return jsonify({
                'success': True,
                'formats': {
                    'video_audio': formats_data.get('video_audio', [])[:10],
                    'video_only': formats_data.get('video_only', [])[:10],
                    'audio_only': formats_data.get('audio_only', [])[:5],
                    'all_heights': formats_data.get('all_heights', []),
                    'max_height': formats_data.get('max_height', 0),
                    'presets': presets
                },
                'info': result.get('info', {}),
                'platform': result.get('platform', 'unknown')
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
            'message': f'خطأ في فحص الجودات: {str(e)}'
        }), 500


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
    """بدء التحميل مع تنسيق محدد"""
    data = request.json
    url = data.get('url', '')
    format_command = data.get('format', data.get('quality', 'best'))
    is_audio = data.get('audio_only', False)
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    
    download_id = secrets.token_hex(8)
    
    thread = threading.Thread(
        target=downloader.download_with_format,
        args=(url, format_command, download_id, is_audio)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'download_id': download_id})


@app.route('/api/media/progress/<download_id>')
def api_get_download_progress(download_id):
    """الحصول على حالة التحميل"""
    progress = download_progress.get(download_id, {
        'status': 'unknown',
        'percent': '0%'
    })
    return jsonify(progress)


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """API للتحويل إلى نص من ملف"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'لم يتم اختيار ملف'}), 400
        
        # حفظ الملف
        filename = secure_filename(file.filename)
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(file_path))
        
        # استخراج الصوت إذا كان فيديو
        audio_file = str(file_path)
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')):
            audio_file = str(file_path).rsplit('.', 1)[0] + '_audio.wav'
            if not VideoProcessor.extract_audio(str(file_path), audio_file):
                return jsonify({'success': False, 'message': 'فشل استخراج الصوت'}), 500
        
        # التحويل إلى نص
        model = request.form.get('model', 'base')
        language = request.form.get('language', 'auto')
        
        result = whisper_transcriber.transcribe(audio_file, model, language)
        
        # إنشاء ملف SRT من segments
        srt_file = None
        if result.get('segments'):
            segments = result.get('segments', [])
            srt_content = SubtitleProcessor.create_srt(segments)
            
            if srt_content:
                srt_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
                srt_path = Path(app.config['SUBTITLE_FOLDER']) / srt_filename
                SubtitleProcessor.save_srt_file(srt_content, srt_path)
                srt_file = srt_filename
                logger.info(f"Created SRT file: {srt_path}")
        
        # حذف الملفات المؤقتة
        try:
            if audio_file != str(file_path):
                os.remove(audio_file)
            os.remove(str(file_path))
        except:
            pass
        
        return jsonify({
            'success': True,
            'text': result.get('text', ''),
            'language': result.get('language', language),
            'segments': result.get('segments', []),
            'srt_file': srt_file,
            'srt_url': f'/download/{srt_file}' if srt_file else None
        })
    
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/transcribe-from-url', methods=['POST'])
def api_transcribe_from_url():
    """API للتحويل إلى نص من رابط"""
    try:
        data = request.json
        url = data.get('url')
        quality = data.get('quality', '720p')
        model = data.get('model', 'base')
        language = data.get('language', 'auto')
        
        if not url:
            return jsonify({'success': False, 'message': 'الرجاء إدخال رابط'}), 400
        
        # تحميل الفيديو
        download_folder = Path(app.config['DOWNLOAD_FOLDER'])
        
        # تحديد format command
        if quality == 'best':
            format_cmd = 'bestvideo+bestaudio/best'
        elif quality == '720p':
            format_cmd = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
        elif quality == '480p':
            format_cmd = 'bestvideo[height<=480]+bestaudio/best[height<=480]'
        elif quality == '1080p':
            format_cmd = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
        else:
            format_cmd = quality
        
        # إعدادات yt-dlp مع تحويل إجباري إلى MP4 H.264 متوافق
        ydl_opts = {
            'format': format_cmd,
            'outtmpl': str(download_folder / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'recode_video': 'mp4',  # إجبار إعادة الترميز إلى MP4
            'postprocessor_args': {
                'ffmpeg': [
                    '-c:v', 'libx264',
                    '-profile:v', 'high',
                    '-level', '4.0',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-movflags', '+faststart',
                    '-strict', 'experimental'
                ]
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        
        # التأكد من أن الملف موجود
        if not os.path.exists(filename):
            video_files = []
            for file in download_folder.iterdir():
                if file.is_file():
                    ext = file.suffix.lower()
                    if ext in ['.mp4', '.webm', '.mkv', '.mov', '.avi', '.flv']:
                        video_files.append((file, file.stat().st_mtime))
            
            if video_files:
                video_files.sort(key=lambda x: x[1], reverse=True)
                filename = str(video_files[0][0])
        
        if not os.path.exists(filename):
            return jsonify({'success': False, 'message': 'فشل تحميل الفيديو'}), 400
        
        # تحويل إجباري إلى MP4 H.264 قبل المعالجة
        if not filename.endswith('.mp3') and not filename.endswith('.m4a'):
            logger.info(f"Converting video to H.264 before processing: {filename}")
            converted_file = downloader._ensure_mp4_h264(filename)
            if converted_file != filename:
                filename = converted_file
        
        # استخراج الصوت
        audio_file = filename.rsplit('.', 1)[0] + '_audio.wav'
        if not VideoProcessor.extract_audio(filename, audio_file):
            return jsonify({'success': False, 'message': 'فشل استخراج الصوت'}), 500
        
        # التحويل إلى نص
        result = whisper_transcriber.transcribe(audio_file, model, language)
        
        # إنشاء ملف SRT من segments
        srt_file = None
        if result.get('segments'):
            segments = result.get('segments', [])
            srt_content = SubtitleProcessor.create_srt(segments)
            
            if srt_content:
                srt_filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
                srt_path = Path(app.config['SUBTITLE_FOLDER']) / srt_filename
                SubtitleProcessor.save_srt_file(srt_content, srt_path)
                srt_file = srt_filename
                logger.info(f"Created SRT file: {srt_path}")
        
        # حذف الملفات المؤقتة
        try:
            os.remove(audio_file)
        except:
            pass
        
        return jsonify({
            'success': True,
            'text': result.get('text', ''),
            'language': result.get('language', language),
            'segments': result.get('segments', []),
            'srt_file': srt_file,
            'srt_url': f'/download/{srt_file}' if srt_file else None
        })
    
    except Exception as e:
        logger.error(f"Transcribe from URL error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def api_translate():
    """API للترجمة - يدعم النص وملفات SRT"""
    try:
        # التحقق من نوع الطلب (JSON أو FormData)
        if request.content_type and 'multipart/form-data' in request.content_type:
            # رفع ملف SRT
            if 'srt_file' not in request.files:
                return jsonify({'success': False, 'message': 'لم يتم رفع ملف SRT'}), 400
            
            srt_file = request.files['srt_file']
            if srt_file.filename == '':
                return jsonify({'success': False, 'message': 'لم يتم اختيار ملف'}), 400
            
            source_lang = request.form.get('source_lang', 'auto')
            target_lang = request.form.get('target_lang', 'ar')
            
            # حفظ الملف مؤقتاً
            filename = secure_filename(srt_file.filename)
            file_path = Path(app.config['UPLOAD_FOLDER']) / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            srt_file.save(str(file_path))
            
            # قراءة ملف SRT
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                srt_content = f.read()
            
            # تحليل ملف SRT
            segments = []
            lines = srt_content.strip().split('\n')
            i = 0
            
            while i < len(lines):
                if lines[i].strip().isdigit():
                    if i + 2 < len(lines):
                        timing = lines[i + 1].strip()
                        text = lines[i + 2].strip()
                        
                        if ' --> ' in timing and text:
                            start, end = timing.split(' --> ')
                            # تحويل التوقيت إلى ثواني
                            def time_to_seconds(time_str):
                                time_str = time_str.replace(',', '.')
                                parts = time_str.split(':')
                                if len(parts) == 3:
                                    hours = int(parts[0])
                                    minutes = int(parts[1])
                                    secs = float(parts[2])
                                    return hours * 3600 + minutes * 60 + secs
                                return 0.0
                            
                            start_sec = time_to_seconds(start)
                            end_sec = time_to_seconds(end)
                            
                            segments.append({
                                'start': start_sec,
                                'end': end_sec,
                                'text': text
                            })
                        i += 4
                    else:
                        i += 1
                else:
                    i += 1
            
            if not segments:
                return jsonify({'success': False, 'message': 'ملف SRT فارغ أو غير صحيح'}), 400
            
            # ترجمة كل segment
            if not TRANSLATOR_AVAILABLE:
                return jsonify({'success': False, 'message': 'المترجم غير متوفر'}), 503
            
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_segments = []
            
            for seg in segments:
                try:
                    translated_text = translator.translate(seg['text'])
                    translated_segments.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': translated_text
                    })
                except Exception as e:
                    logger.warning(f"Translation failed for segment: {e}")
                    translated_segments.append(seg)  # استخدام النص الأصلي في حالة الفشل
            
            # إنشاء ملف SRT مترجم
            cleaned_segments = SubtitleProcessor.clean_and_merge_segments(translated_segments)
            translated_srt_content = SubtitleProcessor.create_srt(cleaned_segments)
            
            if translated_srt_content:
                srt_filename = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
                srt_path = Path(app.config['SUBTITLE_FOLDER']) / srt_filename
                SubtitleProcessor.save_srt_file(translated_srt_content, srt_path)
                
                # حذف الملف المؤقت
                try:
                    os.remove(str(file_path))
                except:
                    pass
                
                return jsonify({
                    'success': True,
                    'translated_text': ' '.join([s['text'] for s in translated_segments]),
                    'translated_segments': translated_segments,
                    'srt_file': srt_filename,
                    'srt_url': f'/download/{srt_filename}',
                    'source_lang': source_lang,
                    'target_lang': target_lang
                })
            else:
                return jsonify({'success': False, 'message': 'فشل إنشاء ملف SRT مترجم'}), 500
        
        else:
            # ترجمة نص عادي
            data = request.json
            text = data.get('text')
            source_lang = data.get('source_lang', 'auto')
            target_lang = data.get('target_lang', 'ar')
            
            if not text:
                return jsonify({'success': False, 'message': 'لا يوجد نص للترجمة'}), 400
            
            if not TRANSLATOR_AVAILABLE:
                return jsonify({'success': False, 'message': 'المترجم غير متوفر'}), 503
            
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            
            return jsonify({
                'success': True,
                'translated_text': translated,
                'source_lang': source_lang,
                'target_lang': target_lang
            })
    
    except Exception as e:
        logger.error(f"Translate error: {e}")
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
        
        download_id = secrets.token_hex(8)
        
        # تحديد format command
        if quality == 'best':
            format_cmd = 'bestvideo+bestaudio/best'
        elif quality == '720p' or quality == 'medium':
            format_cmd = 'bestvideo[height<=720]+bestaudio/best[height<=720]'
        elif quality == '480p' or quality == 'low':
            format_cmd = 'bestvideo[height<=480]+bestaudio/best[height<=480]'
        elif quality == '1080p':
            format_cmd = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]'
        elif quality == '4k' or quality == '2160p':
            format_cmd = 'bestvideo[height<=2160]+bestaudio/best[height<=2160]'
        elif quality == 'audio':
            format_cmd = 'audio'
        else:
            format_cmd = quality
        
        is_audio = (quality == 'audio')
        
        thread = threading.Thread(
            target=downloader.download_with_format,
            args=(url, format_cmd, download_id, is_audio)
        )
        thread.daemon = True
        thread.start()
        
        # الانتظار حتى يكتمل التحميل
        max_wait = 120
        waited = 0
        
        while waited < max_wait:
            progress = download_progress.get(download_id, {})
            
            if progress.get('status') == 'completed':
                # البحث عن الملف المحمّل
                download_folder = Path(app.config['DOWNLOAD_FOLDER'])
                video_files = []
                for file in download_folder.iterdir():
                    if file.is_file():
                        ext = file.suffix.lower()
                        if ext in ['.mp4', '.webm', '.mkv', '.mp3', '.m4a', '.mov', '.avi', '.flv']:
                            video_files.append((file, file.stat().st_mtime))
                
                if video_files:
                    video_files.sort(key=lambda x: x[1], reverse=True)
                    downloaded_file = str(video_files[0][0])
                    
                    return jsonify({
                        'success': True,
                        'file': downloaded_file,
                        'message': 'تم التحميل بنجاح'
                    })
            
            elif progress.get('status') == 'error':
                return jsonify({
                    'success': False,
                    'message': progress.get('message', 'فشل التحميل')
                }), 400
            
            time.sleep(2)
            waited += 2
        
        return jsonify({
            'success': False,
            'message': 'انتهت مهلة التحميل'
        }), 408
    
    except Exception as e:
        logger.error(f"Download API error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    """تحميل الملفات"""
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


@app.route('/api/storage-info', methods=['GET'])
def api_storage_info():
    """معلومات التخزين"""
    try:
        folders = {
            'downloads': app.config['DOWNLOAD_FOLDER'],
            'outputs': app.config['OUTPUT_FOLDER'],
            'subtitles': app.config['SUBTITLE_FOLDER'],
            'uploads': app.config['UPLOAD_FOLDER']
        }
        
        info = {}
        for name, folder in folders.items():
            folder_path = Path(folder)
            if folder_path.exists():
                total_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                file_count = len([f for f in folder_path.rglob('*') if f.is_file()])
                info[name] = {
                    'size': total_size,
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'files': file_count
                }
            else:
                info[name] = {'size': 0, 'size_mb': 0, 'files': 0}
        
        return jsonify({'success': True, 'folders': info})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/cleanup', methods=['POST'])
def api_cleanup():
    """تنظيف الملفات المؤقتة والقديمة"""
    try:
        data = request.json or {}
        cleanup_type = data.get('type', 'all')  # all, temp, old
        
        folders_to_clean = []
        if cleanup_type == 'all':
            folders_to_clean = [
                app.config['DOWNLOAD_FOLDER'],
                app.config['OUTPUT_FOLDER'],
                app.config['SUBTITLE_FOLDER'],
                app.config['UPLOAD_FOLDER']
            ]
        elif cleanup_type == 'temp':
            folders_to_clean = [app.config['UPLOAD_FOLDER']]
        elif cleanup_type == 'old':
            # حذف الملفات الأقدم من 7 أيام
            folders_to_clean = [
                app.config['DOWNLOAD_FOLDER'],
                app.config['OUTPUT_FOLDER'],
                app.config['SUBTITLE_FOLDER']
            ]
        
        deleted_count = 0
        deleted_size = 0
        current_time = time.time()
        days_old = 7  # حذف الملفات الأقدم من 7 أيام
        
        for folder_path in folders_to_clean:
            folder = Path(folder_path)
            if not folder.exists():
                continue
            
            for file_path in folder.rglob('*'):
                if file_path.is_file():
                    try:
                        # حذف الملفات المؤقتة أو القديمة
                        should_delete = False
                        
                        if cleanup_type == 'temp':
                            # حذف الملفات المؤقتة فقط
                            should_delete = file_path.name.startswith('temp_')
                        elif cleanup_type == 'old':
                            # حذف الملفات الأقدم من 7 أيام
                            file_age = current_time - file_path.stat().st_mtime
                            should_delete = file_age > (days_old * 24 * 3600)
                        else:
                            # حذف جميع الملفات
                            should_delete = True
                        
                        if should_delete:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            deleted_count += 1
                            deleted_size += file_size
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
                        continue
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'deleted_size': deleted_size,
            'deleted_size_mb': round(deleted_size / (1024 * 1024), 2),
            'message': f'تم حذف {deleted_count} ملف ({round(deleted_size / (1024 * 1024), 2)} MB)'
        })
    
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get-video-thumbnail', methods=['POST'])
def api_get_video_thumbnail():
    """استخراج صورة مصغرة من الفيديو"""
    try:
        data = request.json or {}
        video_file = data.get('video_file')
        
        logger.info(f"Thumbnail request: video_file={video_file}")
        
        if not video_file:
            logger.warning("No video_file provided in thumbnail request")
            return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 400
        
        # البحث عن الملف
        if not os.path.exists(video_file):
            download_folder = Path(app.config['DOWNLOAD_FOLDER'])
            basename = os.path.basename(video_file)
            possible_paths = [
                download_folder / basename,
                Path(video_file),
                download_folder / video_file.replace('downloads/', '').replace('downloads\\', '')
            ]
            
            logger.info(f"Video file not found at {video_file}, trying paths: {possible_paths}")
            
            found = False
            for path in possible_paths:
                if path.exists():
                    video_file = str(path)
                    found = True
                    logger.info(f"Found video file at: {video_file}")
                    break
            
            if not found:
                logger.error(f"Video file not found in any location: {video_file}")
                return jsonify({'success': False, 'message': 'ملف الفيديو غير موجود'}), 404
        
        # استخراج صورة مصغرة باستخدام ffmpeg
        thumbnail_path = Path(app.config['OUTPUT_FOLDER']) / f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-ss', '00:00:01',
            '-vframes', '1',
            '-q:v', '2',
            '-y',
            str(thumbnail_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        
        if result.returncode == 0 and thumbnail_path.exists():
            return jsonify({
                'success': True,
                'thumbnail_url': f'/download/{thumbnail_path.name}'
            })
        else:
            # إذا فشل، استخدم صورة افتراضية
            return jsonify({
                'success': False,
                'message': 'فشل استخراج الصورة المصغرة'
            }), 500
            
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/merge-subtitle', methods=['POST'])
def api_merge_subtitle():
    """دمج الترجمة مع الفيديو (من تبويب محرر الترجمة)"""
    try:
        video_file = request.files.get('video')
        subtitle_file = request.files.get('subtitle')
        
        if not video_file or not subtitle_file:
            return jsonify({
                'success': False,
                'message': 'يرجى رفع الفيديو وملف الترجمة'
            }), 400
        
        # حفظ الملفات
        video_filename = secure_filename(video_file.filename)
        subtitle_filename = secure_filename(subtitle_file.filename)
        
        temp_video_path = Path(app.config['UPLOAD_FOLDER']) / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_filename}"
        temp_subtitle_path = Path(app.config['UPLOAD_FOLDER']) / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{subtitle_filename}"
        
        # التأكد من وجود المجلد
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_file.save(str(temp_video_path))
        subtitle_file.save(str(temp_subtitle_path))
        
        # التأكد من وجود الملفات
        if not temp_video_path.exists():
            return jsonify({
                'success': False,
                'message': 'فشل حفظ ملف الفيديو'
            }), 500
        
        if not temp_subtitle_path.exists():
            return jsonify({
                'success': False,
                'message': 'فشل حفظ ملف الترجمة'
            }), 500
        
        # قراءة إعدادات الترجمة مع معالجة آمنة
        try:
            fontSize = request.form.get('fontSize', request.form.get('font_size', 24))
            fontSize = int(fontSize) if fontSize else 24
        except:
            fontSize = 24
        
        try:
            bgOpacity = request.form.get('bgOpacity', request.form.get('bg_opacity', 180))
            bgOpacity = int(bgOpacity) if bgOpacity else 180
        except:
            bgOpacity = 180
        
        settings = {
            'fontSize': fontSize,
            'fontColor': str(request.form.get('fontColor', request.form.get('font_color', '#FFFFFF'))),
            'bgColor': str(request.form.get('bgColor', request.form.get('bg_color', '#000000'))),
            'bgOpacity': bgOpacity,
            'position': str(request.form.get('position', 'bottom')),
            'fontFamily': str(request.form.get('fontFamily', request.form.get('font_name', 'Arial')))
        }
        
        # دمج
        output_file = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = Path(app.config['OUTPUT_FOLDER']) / output_file
        
        # التأكد من وجود مجلد الإخراج
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Merging subtitle: video={temp_video_path}, subtitle={temp_subtitle_path}, output={output_path}")
        
        success = VideoProcessor.merge_subtitles(
            str(temp_video_path),
            str(temp_subtitle_path),
            str(output_path),
            settings
        )
        
        # حذف الملفات المؤقتة
        try:
            if temp_video_path.exists():
                temp_video_path.unlink()
            if temp_subtitle_path.exists():
                temp_subtitle_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp files: {e}")
        
        if success:
            if output_path.exists():
                return jsonify({
                    'success': True,
                    'download_url': f'/download/{output_file}'
                })
            else:
                logger.error(f"Merge succeeded but output file not found: {output_path}")
                return jsonify({
                    'success': False,
                    'message': 'فشل إنشاء الفيديو النهائي'
                }), 500
        else:
            logger.error("merge_subtitles returned False")
            return jsonify({
                'success': False,
                'message': 'فشل دمج الترجمة مع الفيديو'
            }), 500
            
    except Exception as e:
        logger.error(f"Merge subtitle error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'خطأ في دمج الترجمة: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 التطبيق المتكامل للترجمة والتحميل v6.0")
    print("="*60)
    print(f"✅ Whisper: {'متوفر' if WHISPER_AVAILABLE else 'غير متوفر'}")
    print(f"✅ Faster Whisper: {'متوفر' if FASTER_WHISPER_AVAILABLE else 'غير متوفر'}")
    print(f"✅ المترجم: {'متوفر' if TRANSLATOR_AVAILABLE else 'غير متوفر'}")
    print(f"✅ FFmpeg: {'متوفر' if VideoProcessor.check_ffmpeg() else 'غير متوفر'}")
    print("\n🌐 الخادم يعمل على: http://localhost:5000")
    print("\n🛑 لإيقاف الخادم: CTRL+C")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
