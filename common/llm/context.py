"""### LLM > Context
Gestion du contexte de conversation et des messages."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal, Optional, Any
from dataclasses import dataclass, field

import discord
import tiktoken

logger = logging.getLogger(f'MARI4.llm.context')

# CONSTANTES ------------------------------------------------------

GPT_TOKENIZER = tiktoken.get_encoding('cl100k_base')

# Configuration par défaut du contexte
DEFAULT_CONTEXT_WINDOW = 512 * 64  # 32k tokens
DEFAULT_CONTEXT_AGE = timedelta(hours=2)  # 2h


# COMPOSANTS DE CONTENU -------------------------------------------

@dataclass
class ContentComponent:
    """Composant de contenu d'un message."""
    type: Literal['text', 'image_url']
    data: dict
    token_count: int = 0
    
    def to_payload(self) -> dict:
        """Convertit en format OpenAI."""
        return self.data

@dataclass
class TextComponent(ContentComponent):
    """Composant texte."""
    def __init__(self, text: str):
        data = {'type': 'text', 'text': text}
        token_count = len(GPT_TOKENIZER.encode(text))
        super().__init__(type='text', data=data, token_count=token_count)

@dataclass
class ImageComponent(ContentComponent):
    """Composant image."""
    def __init__(self, url: str, detail: Literal['low', 'high', 'auto'] = 'auto'):
        data = {
            'type': 'image_url',
            'image_url': {'url': url, 'detail': detail}
        }
        super().__init__(type='image_url', data=data, token_count=250)  # Estimation

@dataclass
class MetadataComponent(ContentComponent):
    """Composant de métadonnées (affiché comme texte)."""
    def __init__(self, title: str, **metadata):
        text = f'<{title.upper()}'
        if metadata:
            text += ' ' + ' '.join([f'{k.lower()}={v}' for k, v in metadata.items()])
        text += '>'
        data = {'type': 'text', 'text': text}
        token_count = len(GPT_TOKENIZER.encode(text))
        super().__init__(type='text', data=data, token_count=token_count)

# RECORDS DE MESSAGES ---------------------------------------------

@dataclass
class MessageRecord:
    """Enregistrement d'un message dans l'historique."""
    role: Literal['user', 'assistant', 'developer', 'tool']
    components: list[ContentComponent]
    created_at: datetime
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    # Référence Discord optionnelle
    discord_message: Optional[discord.Message] = None
    
    @property
    def token_count(self) -> int:
        """Nombre total de tokens du message."""
        return sum(c.token_count for c in self.components)
    
    @property
    def full_text(self) -> str:
        """Texte complet du message (composants texte uniquement)."""
        return ''.join(
            c.data['text'] for c in self.components 
            if c.type == 'text' and 'text' in c.data
        )
    
    @property
    def contains_image(self) -> bool:
        """True si le message contient au moins une image."""
        return any(c.type == 'image_url' for c in self.components)
    
    def to_payload(self) -> dict:
        """Convertit en format OpenAI."""
        payload = {
            'role': self.role,
            'content': [c.to_payload() for c in self.components]
        }
        if self.name:
            payload['name'] = self.name
        return payload

@dataclass
class ToolCallRecord:
    """Enregistrement d'un appel d'outil."""
    id: str
    function_name: str
    arguments: dict
    
    def to_payload(self) -> dict:
        """Convertit en format OpenAI."""
        return {
            'id': self.id,
            'type': 'function',
            'function': {
                'name': self.function_name,
                'arguments': json.dumps(self.arguments)
            }
        }

class AssistantRecord(MessageRecord):
    """Enregistrement d'un message assistant avec possibles tool calls."""
    
    def __init__(self, components: list[ContentComponent], 
                 created_at: datetime,
                 tool_calls: list[ToolCallRecord] = None,
                 finish_reason: str = None,
                 **kwargs):
        super().__init__(
            role='assistant',
            components=components,
            created_at=created_at,
            **kwargs
        )
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
    
    def to_payload(self) -> dict:
        """Convertit en format OpenAI."""
        if self.tool_calls:
            return {
                'role': 'assistant',
                'tool_calls': [tc.to_payload() for tc in self.tool_calls]
            }
        return super().to_payload()

class ToolResponseRecord(MessageRecord):
    """Enregistrement d'une réponse d'outil."""
    
    def __init__(self, tool_call_id: str, response_data: dict, created_at: datetime, **kwargs):
        components = [TextComponent(json.dumps(response_data))]
        super().__init__(
            role='tool',
            components=components,
            created_at=created_at,
            **kwargs
        )
        self.tool_call_id = tool_call_id
        self.response_data = response_data
    
    def to_payload(self) -> dict:
        """Convertit en format OpenAI."""
        payload = super().to_payload()
        payload['tool_call_id'] = self.tool_call_id
        return payload

# CONTEXTE DE CONVERSATION ----------------------------------------

class ConversationContext:
    """Gestion du contexte de conversation pour un salon.
    
    Maintient l'historique des messages avec nettoyage automatique
    basé sur une fenêtre de tokens et d'âge.
    """
    
    def __init__(self,
                 developer_prompt: str,
                 *,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 context_age: timedelta = DEFAULT_CONTEXT_AGE):
        """Initialise le contexte.
        
        Args:
            developer_prompt: Prompt système/développeur
            context_window: Taille max de la fenêtre en tokens
            context_age: Âge maximum des messages
        """
        self.developer_prompt = developer_prompt
        self.context_window = context_window
        self.context_age = context_age
        
        # Historique des messages
        self._messages: list[MessageRecord] = []
        
        logger.debug(f"ConversationContext créé (window={context_window}, age={context_age})")
    
    def add_message(self, message: MessageRecord) -> None:
        """Ajoute un message à l'historique."""
        self._messages.append(message)
        logger.debug(f"Message ajouté: role={message.role}, tokens={message.token_count}")
    
    def add_user_message(self, components: list[ContentComponent], 
                        name: str = 'user',
                        discord_message: Optional[discord.Message] = None,
                        **metadata) -> MessageRecord:
        """Ajoute un message utilisateur."""
        record = MessageRecord(
            role='user',
            components=components,
            created_at=datetime.now(timezone.utc),
            name=name,
            discord_message=discord_message,
            metadata=metadata
        )
        self.add_message(record)
        return record
    
    def add_assistant_message(self, components: list[ContentComponent],
                             tool_calls: list[ToolCallRecord] = None,
                             finish_reason: str = None,
                             discord_message: Optional[discord.Message] = None,
                             **metadata) -> AssistantRecord:
        """Ajoute un message assistant."""
        record = AssistantRecord(
            components=components,
            created_at=datetime.now(timezone.utc),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            discord_message=discord_message,
            metadata=metadata
        )
        self.add_message(record)
        return record
    
    def add_tool_response(self, tool_call_id: str, response_data: dict, **metadata) -> ToolResponseRecord:
        """Ajoute une réponse d'outil."""
        record = ToolResponseRecord(
            tool_call_id=tool_call_id,
            response_data=response_data,
            created_at=datetime.now(timezone.utc),
            metadata=metadata
        )
        self.add_message(record)
        return record
    
    def get_messages(self, 
                     filter_func: Optional[Callable[[MessageRecord], bool]] = None) -> list[MessageRecord]:
        """Récupère les messages avec filtre optionnel."""
        if filter_func is None:
            return self._messages.copy()
        return [m for m in self._messages if filter_func(m)]
    
    def get_recent_messages(self, count: int = 10) -> list[MessageRecord]:
        """Récupère les N derniers messages."""
        return self._messages[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """Vide l'historique."""
        self._messages.clear()
        logger.info("Historique vidé")
    
    def trim(self) -> None:
        """Nettoie l'historique selon les limites de tokens et d'âge."""
        now = datetime.now(timezone.utc)
        
        # 1. Supprimer les messages trop vieux
        self._messages = [
            m for m in self._messages 
            if now - m.created_at < self.context_age
        ]
        
        # 2. Supprimer les messages dépassant la fenêtre de tokens
        # On garde les plus récents en priorité
        total_tokens = 0
        valid_messages = []
        
        for message in reversed(self._messages):
            msg_tokens = message.token_count
            if total_tokens + msg_tokens <= self.context_window:
                valid_messages.insert(0, message)
                total_tokens += msg_tokens
            else:
                break
        
        removed_count = len(self._messages) - len(valid_messages)
        self._messages = valid_messages
        
        if removed_count > 0:
            logger.debug(f"Nettoyage: {removed_count} message(s) supprimé(s), {total_tokens} tokens restants")
    
    def prepare_payload(self) -> list[dict]:
        """Prépare le payload pour l'API OpenAI.
        
        Inclut le prompt développeur + tous les messages de l'historique.
        """
        # Nettoyage avant préparation
        self.trim()
        
        # Prompt développeur
        dev_message = MessageRecord(
            role='developer',
            components=[TextComponent(self.developer_prompt)],
            created_at=datetime.now(timezone.utc)
        )
        
        # Construction du payload
        messages = [dev_message] + self._messages
        return [m.to_payload() for m in messages]
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur le contexte."""
        total_tokens = sum(m.token_count for m in self._messages)
        user_count = sum(1 for m in self._messages if m.role == 'user')
        assistant_count = sum(1 for m in self._messages if m.role == 'assistant')
        
        return {
            'total_messages': len(self._messages),
            'total_tokens': total_tokens,
            'user_messages': user_count,
            'assistant_messages': assistant_count,
            'window_usage_pct': (total_tokens / self.context_window * 100) if self.context_window > 0 else 0
        }
    
    def filter_images(self) -> None:
        """Supprime tous les composants image de l'historique.
        
        Utile en cas d'erreur 'invalid_image_url' pour retry sans images.
        """
        for message in self._messages:
            message.components = [c for c in message.components if c.type != 'image_url']
        logger.info("Toutes les images ont été retirées du contexte")

