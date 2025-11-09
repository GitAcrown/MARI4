"""### LLM > Attachments
Pipeline de traitement des pièces jointes (audio, vidéo, fichiers texte)."""

import io
import os
import base64
import logging
from pathlib import Path
from typing import Optional

import discord
from moviepy import VideoFileClip
import imageio

from .context import ContentComponent, TextComponent, ImageComponent, MetadataComponent
from .client import MariaLLMClient

logger = logging.getLogger(f'MARI4.llm.attachments')

# CONSTANTES ------------------------------------------------------

# Répertoire temporaire
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Limites de taille
MAX_VIDEO_SIZE = 20 * 1024 * 1024  # 20 Mo
MAX_TEXT_FILE_SIZE = 1024 * 1024  # 1 Mo
MAX_TEXT_CONTENT_LENGTH = 100000  # 100k caractères

# Configuration analyse vidéo
VIDEO_ANALYSIS_MODEL = 'gpt-4.1-mini'
VIDEO_ANALYSIS_AUDIO_MODEL = 'gpt-4o-mini-transcribe'
VIDEO_ANALYSIS_TEMPERATURE = 0.15
VIDEO_ANALYSIS_MAX_TOKENS = 1000
VIDEO_ANALYSIS_NB_FRAMES = 10
VIDEO_ANALYSIS_PROMPT = """A partir des éléments fournis (images et transcription audio) qui ont été extraits d'une vidéo, réalise une description EXTREMEMENT DÉTAILLÉE (sujets, actions, scène, apparences etc.). Ne répond qu'avec cette description sans aucun autre texte. Les images sont fournies dans l'ordre chronologique et sont des frames extraites à intervalles égaux de la vidéo."""

# CACHE -----------------------------------------------------------

class AttachmentCache:
    """Cache pour les transcriptions et analyses vidéo."""
    
    def __init__(self, max_size: int = 25):
        self.max_size = max_size
        self._transcript_cache: dict[str, str] = {}
        self._video_cache: dict[str, MetadataComponent] = {}
    
    def get_transcript(self, url: str) -> Optional[str]:
        """Récupère une transcription depuis le cache."""
        return self._transcript_cache.get(url)
    
    def set_transcript(self, url: str, transcript: str) -> None:
        """Enregistre une transcription dans le cache."""
        self._transcript_cache[url] = transcript
        self._cleanup_cache(self._transcript_cache)
    
    def get_video_analysis(self, filename: str) -> Optional[MetadataComponent]:
        """Récupère une analyse vidéo depuis le cache."""
        return self._video_cache.get(filename)
    
    def set_video_analysis(self, filename: str, analysis: MetadataComponent) -> None:
        """Enregistre une analyse vidéo dans le cache."""
        self._video_cache[filename] = analysis
        self._cleanup_cache(self._video_cache)
    
    def _cleanup_cache(self, cache: dict) -> None:
        """Nettoie un cache s'il dépasse la taille max."""
        if len(cache) > self.max_size:
            items = list(cache.items())
            cache.clear()
            cache.update(dict(items[-self.max_size:]))
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du cache."""
        return {
            'transcript_cache_size': len(self._transcript_cache),
            'video_cache_size': len(self._video_cache)
        }

# PROCESSEURS -----------------------------------------------------

async def process_audio_attachment(attachment: discord.Attachment,
                                   client: MariaLLMClient,
                                   cache: AttachmentCache) -> list[ContentComponent]:
    """Traite une pièce jointe audio.
    
    Returns:
        Liste de composants (métadonnées + transcription)
    """
    # Vérifier le cache
    cached = cache.get_transcript(attachment.url)
    if cached:
        return [MetadataComponent('AUDIO', 
                                 filename=attachment.filename,
                                 transcript=cached,
                                 url=attachment.url)]
    
    # Transcription
    audio_file = None
    try:
        audio_file = io.BytesIO()
        audio_file.name = attachment.filename
        await attachment.save(audio_file, seek_begin=True)
        
        transcript = await client.create_transcription(audio_file)
        cache.set_transcript(attachment.url, transcript)
        
        return [MetadataComponent('AUDIO',
                                 filename=attachment.filename,
                                 transcript=transcript,
                                 url=attachment.url)]
        
    except Exception as e:
        logger.error(f"Erreur transcription audio '{attachment.filename}': {e}")
        return [MetadataComponent('AUDIO',
                                 filename=attachment.filename,
                                 error='TRANSCRIPTION_FAILED',
                                 url=attachment.url)]
    finally:
        if audio_file:
            audio_file.close()

async def process_video_attachment(attachment: discord.Attachment,
                                   client: MariaLLMClient,
                                   cache: AttachmentCache) -> list[ContentComponent]:
    """Traite une pièce jointe vidéo.
    
    Returns:
        Liste de composants (métadonnées + analyse)
    """
    # Vérifier le cache
    cached = cache.get_video_analysis(attachment.filename)
    if cached:
        return [cached]
    
    # Vérification taille
    if attachment.size > MAX_VIDEO_SIZE:
        logger.warning(f"Vidéo trop volumineuse: {attachment.filename}")
        return [MetadataComponent('VIDEO',
                                 filename=attachment.filename,
                                 size=attachment.size,
                                 error='FILE_TOO_LARGE')]
    
    # Téléchargement
    path = TEMP_DIR / attachment.filename
    await attachment.save(path, seek_begin=True, use_cached=True)
    
    if not path.exists():
        return [MetadataComponent('VIDEO',
                                 filename=attachment.filename,
                                 error='FILE_NOT_FOUND')]
    
    # Analyse
    try:
        analysis = await _analyze_video_file(path, attachment.filename, client)
        cache.set_video_analysis(attachment.filename, analysis)
        return [analysis]
    except Exception as e:
        logger.error(f"Erreur analyse vidéo '{attachment.filename}': {e}")
        return [MetadataComponent('VIDEO',
                                 filename=attachment.filename,
                                 error='ANALYSIS_FAILED')]

async def _analyze_video_file(path: Path, filename: str, client: MariaLLMClient) -> MetadataComponent:
    """Analyse un fichier vidéo (extraction audio + frames + analyse IA)."""
    audio_transcript = ''
    images = []
    duration = 0
    
    clip = None
    audio = None
    audio_path = None
    
    try:
        clip = VideoFileClip(str(path))
        duration = getattr(clip, 'duration', 0) or 0
        audio = getattr(clip, 'audio', None)
        
        # Extraction audio
        if audio:
            audio_path = path.with_suffix('.wav')
            try:
                audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                audio_transcript = await client.create_transcription(
                    audio_path, 
                    model=VIDEO_ANALYSIS_AUDIO_MODEL
                )
            except TypeError:
                # Fallback anciennes versions moviepy
                audio.write_audiofile(str(audio_path))
                audio_transcript = await client.create_transcription(
                    audio_path,
                    model=VIDEO_ANALYSIS_AUDIO_MODEL
                )
            except Exception as e:
                logger.warning(f"Erreur extraction audio: {e}")
                audio_transcript = "AUDIO_EXTRACTION_FAILED"
        
        # Extraction frames
        if duration and duration > 0:
            time_points = [duration * i / VIDEO_ANALYSIS_NB_FRAMES 
                          for i in range(VIDEO_ANALYSIS_NB_FRAMES)]
            
            for t, time_point in enumerate(time_points):
                try:
                    frame = clip.get_frame(time_point)
                    frame_path = path.with_stem(f"frame_{t}").with_suffix('.jpg')
                    imageio.imwrite(str(frame_path), frame)
                    images.append(frame_path)
                except Exception as e:
                    logger.warning(f"Erreur extraction frame {t}: {e}")
                    continue
        
        # Analyse IA
        if images:
            description = await _analyze_video_content(images, filename, duration, audio_transcript, client)
        else:
            description = "NO_FRAMES_EXTRACTED"
        
        return MetadataComponent('VIDEO',
                                filename=filename,
                                duration=duration,
                                audio_transcript=audio_transcript,
                                images_extracted=len(images),
                                description=description)
        
    finally:
        # Nettoyage
        await _cleanup_video_resources(clip, audio, images, audio_path, path)

async def _analyze_video_content(images: list[Path], 
                                 filename: str,
                                 duration: float,
                                 audio_transcript: str,
                                 client: MariaLLMClient) -> str:
    """Analyse le contenu vidéo avec l'IA."""
    try:
        from .context import MessageRecord
        from datetime import datetime, timezone
        
        # Construire les messages
        messages = []
        
        # Prompt développeur
        messages.append({
            'role': 'developer',
            'content': VIDEO_ANALYSIS_PROMPT
        })
        
        # Composants utilisateur
        user_content = []
        
        # Ajouter les images
        for image_path in images:
            if not image_path.exists():
                continue
            try:
                with open(image_path, 'rb') as img_file:
                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                    data_url = f"data:image/jpeg;base64,{encoded}"
                    user_content.append({
                        'type': 'image_url',
                        'image_url': {'url': data_url, 'detail': 'low'}
                    })
            except Exception as e:
                logger.warning(f"Erreur encodage image {image_path}: {e}")
                continue
        
        if not user_content:
            return "NO_VALID_IMAGES"
        
        # Ajouter métadonnées
        metadata_text = f'<VIDEO filename={filename} duration={duration} audio_transcript={audio_transcript} images_extracted={len(images)}>'
        user_content.append({'type': 'text', 'text': metadata_text})
        
        messages.append({
            'role': 'user',
            'content': user_content
        })
        
        # Appel API
        completion = await client.create_completion(
            messages=messages,
            model=VIDEO_ANALYSIS_MODEL,
            max_tokens=VIDEO_ANALYSIS_MAX_TOKENS
        )
        
        return completion.choices[0].message.content or "ANALYSIS_FAILED"
        
    except Exception as e:
        logger.error(f"Erreur analyse IA vidéo: {e}")
        return f"ANALYSIS_FAILED: {str(e)}"

async def _cleanup_video_resources(clip, audio, images: list[Path], audio_path: Path | None, video_path: Path):
    """Nettoie les ressources vidéo."""
    try:
        if clip:
            clip.close()
        if audio:
            audio.close()
    except Exception as e:
        logger.error(f"Erreur fermeture ressources vidéo: {e}")
    
    # Supprimer fichiers temporaires
    for image_path in images:
        try:
            if image_path.exists():
                os.unlink(image_path)
        except OSError:
            pass
    
    if audio_path and audio_path.exists():
        try:
            os.unlink(audio_path)
        except OSError:
            pass
    
    if video_path.exists():
        try:
            os.unlink(video_path)
        except OSError:
            pass

async def process_text_file_attachment(attachment: discord.Attachment) -> list[ContentComponent]:
    """Traite une pièce jointe fichier texte.
    
    Returns:
        Liste de composants (métadonnées + contenu)
    """
    # Vérification taille
    if attachment.size > MAX_TEXT_FILE_SIZE:
        logger.warning(f"Fichier texte trop volumineux: {attachment.filename}")
        return [MetadataComponent('TEXT_FILE',
                                 filename=attachment.filename,
                                 size=attachment.size,
                                 error='FILE_TOO_LARGE')]
    
    try:
        # Téléchargement
        content_bytes = await attachment.read()
        
        # Décodage avec plusieurs encodages
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                content = content_bytes.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            return [MetadataComponent('TEXT_FILE',
                                     filename=attachment.filename,
                                     error='ENCODING_ERROR')]
        
        # Troncature si nécessaire
        truncated = False
        if len(content) > MAX_TEXT_CONTENT_LENGTH:
            content = content[:MAX_TEXT_CONTENT_LENGTH] + "\n... [CONTENU TRONQUÉ]"
            truncated = True
        
        # Extension
        file_extension = attachment.filename.split('.')[-1].lower() if '.' in attachment.filename else 'txt'
        
        # Composants
        components = [
            MetadataComponent('TEXT_FILE',
                            filename=attachment.filename,
                            size=attachment.size,
                            encoding=used_encoding,
                            extension=file_extension,
                            truncated=truncated),
            TextComponent(f"```{file_extension}\n{content}\n```")
        ]
        
        logger.info(f"Fichier texte traité: {attachment.filename}")
        return components
        
    except Exception as e:
        logger.error(f"Erreur traitement fichier texte '{attachment.filename}': {e}")
        return [MetadataComponent('TEXT_FILE',
                                 filename=attachment.filename,
                                 error='PROCESSING_ERROR')]

# DISPATCHER ------------------------------------------------------

async def process_attachment(attachment: discord.Attachment,
                            client: MariaLLMClient,
                            cache: AttachmentCache) -> list[ContentComponent]:
    """Dispatcher principal pour traiter une pièce jointe.
    
    Détecte le type et appelle le processeur approprié.
    """
    content_type = attachment.content_type or ''
    filename_lower = attachment.filename.lower()
    
    # Audio
    if content_type.startswith('audio/') or filename_lower.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.flac')):
        return await process_audio_attachment(attachment, client, cache)
    
    # Vidéo
    elif content_type.startswith('video/') or filename_lower.endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
        return await process_video_attachment(attachment, client, cache)
    
    # Fichier texte
    elif (content_type.startswith('text/') or 
          filename_lower.endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log'))):
        return await process_text_file_attachment(attachment)
    
    # Type non supporté
    else:
        logger.debug(f"Type de pièce jointe non supporté: {attachment.filename} ({content_type})")
        return []

