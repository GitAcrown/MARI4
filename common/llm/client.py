"""### LLM > Client
Wrapper central pour l'API OpenAI avec gestion des erreurs et configuration."""

import logging
from typing import Any, Callable, Optional

from openai import AsyncOpenAI
import openai

logger = logging.getLogger(f'MARI4.llm.client')

# CONSTANTES ------------------------------------------------------

# Modèles par défaut
DEFAULT_COMPLETION_MODEL = 'gpt-5-mini'
DEFAULT_TRANSCRIPTION_MODEL = 'gpt-4o-transcribe'
DEFAULT_MAX_COMPLETION_TOKENS = 1000

# EXCEPTIONS ------------------------------------------------------

class MariaLLMError(Exception):
    """Erreur de base pour l'API LLM."""
    pass

class MariaOpenAIError(MariaLLMError):
    """Erreur provenant de l'API OpenAI."""
    pass

# CLIENT ----------------------------------------------------------

class MariaLLMClient:
    """Client central pour les interactions avec l'API OpenAI.
    
    Encapsule AsyncOpenAI et fournit une interface simplifiée avec gestion d'erreurs,
    logs et métriques.
    """
    
    def __init__(self, 
                 api_key: str,
                 *,
                 completion_model: str = DEFAULT_COMPLETION_MODEL,
                 transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
                 max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
                 on_completion: Optional[Callable] = None):
        """Initialise le client LLM.
        
        Args:
            api_key: Clé API OpenAI
            completion_model: Modèle par défaut pour les complétions
            transcription_model: Modèle par défaut pour les transcriptions
            max_completion_tokens: Nombre max de tokens par complétion
            on_completion: Callback optionnel appelé après chaque complétion (monitoring)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Configuration
        self.completion_model = completion_model
        self.transcription_model = transcription_model
        self.max_completion_tokens = max_completion_tokens
        self.on_completion = on_completion
        
        # Statistiques
        self._stats = {
            'completions': 0,
            'transcriptions': 0,
            'errors': 0
        }
        
        logger.info(f"MariaLLMClient initialisé avec modèle {completion_model}")
    
    async def create_completion(self,
                               messages: list[dict],
                               *,
                               model: Optional[str] = None,
                               max_tokens: Optional[int] = None,
                               tools: Optional[list[dict]] = None,
                               parallel_tool_calls: bool = True,
                               **kwargs) -> Any:
        """Crée une complétion de chat.
        
        Args:
            messages: Liste des messages au format OpenAI
            model: Modèle à utiliser (défaut: self.completion_model)
            max_tokens: Nombre max de tokens (défaut: self.max_completion_tokens)
            tools: Liste des outils disponibles
            parallel_tool_calls: Autoriser les appels d'outils parallèles
            **kwargs: Arguments additionnels pour l'API OpenAI
            
        Returns:
            ChatCompletion object
            
        Raises:
            MariaOpenAIError: En cas d'erreur API
        """
        try:
            completion = await self.client.chat.completions.create(
                model=model or self.completion_model,
                messages=messages,
                max_completion_tokens=max_tokens or self.max_completion_tokens,
                tools=tools or [],
                parallel_tool_calls=parallel_tool_calls,
                reasoning_effort='low',
                verbosity='low',
                **kwargs
            )
            
            self._stats['completions'] += 1
            
            # Callback monitoring
            if self.on_completion:
                try:
                    await self.on_completion(completion)
                except Exception as e:
                    logger.warning(f"Erreur dans callback on_completion: {e}")
            
            return completion
            
        except openai.BadRequestError as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur BadRequest OpenAI: {e}")
            raise MariaOpenAIError(f"Requête invalide: {e}") from e
        except openai.OpenAIError as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur OpenAI: {e}")
            raise MariaOpenAIError(f"Erreur API OpenAI: {e}") from e
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur inattendue lors de la complétion: {e}")
            raise MariaLLMError(f"Erreur inattendue: {e}") from e
    
    async def create_transcription(self,
                                   audio_file,
                                   *,
                                   model: Optional[str] = None,
                                   prompt: str = '') -> str:
        """Crée une transcription audio.
        
        Args:
            audio_file: Fichier audio (BytesIO ou Path)
            model: Modèle à utiliser (défaut: self.transcription_model)
            prompt: Guide de transcription optionnel
            
        Returns:
            Texte transcrit
            
        Raises:
            MariaOpenAIError: En cas d'erreur API
        """
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=model or self.transcription_model,
                file=audio_file,
                prompt=prompt
            )
            
            self._stats['transcriptions'] += 1
            return transcript.text
            
        except openai.BadRequestError as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur BadRequest OpenAI (transcription): {e}")
            raise MariaOpenAIError(f"Requête de transcription invalide: {e}") from e
        except openai.OpenAIError as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur OpenAI (transcription): {e}")
            raise MariaOpenAIError(f"Erreur API OpenAI (transcription): {e}") from e
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Erreur inattendue lors de la transcription: {e}")
            raise MariaLLMError(f"Erreur inattendue (transcription): {e}") from e
    
    def get_stats(self) -> dict:
        """Retourne les statistiques d'utilisation."""
        return self._stats.copy()
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()
        logger.info("MariaLLMClient fermé")

