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
DEFAULT_CONTEXT_WINDOW = 24_576  # 24k tokens
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
        # Format spécial pour REFERENCE pour plus de clarté
        if title.upper() == 'REFERENCE':
            if metadata.get('yourself'):
                # Référence à un message du bot
                content = metadata.get('starting_with', '')
                text = f"[RÉFÉRENCE à ton message précédent: {content}]"
            else:
                # Référence à un message d'un autre utilisateur
                author = metadata.get('author', 'utilisateur')
                content = metadata.get('content', '')
                text = f"[RÉFÉRENCE au message de {author}: {content}]"
        else:
            # Format générique pour les autres métadonnées
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
            payload = {
                'role': 'assistant',
                'tool_calls': [tc.to_payload() for tc in self.tool_calls],
                'content': None  # OpenAI requiert content: null pour les messages avec tool_calls
            }
            return payload
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
        """Convertit en format OpenAI.
        
        Pour les messages de type 'tool', OpenAI attend 'content' comme une chaîne JSON,
        pas une liste de composants.
        """
        # Pour les tool responses, content doit être une chaîne JSON directement
        # OpenAI attend que content soit une chaîne, même si c'est du JSON
        content = json.dumps(self.response_data, ensure_ascii=False)
        payload = {
            'role': 'tool',
            'content': content,
            'tool_call_id': self.tool_call_id
        }
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
        
    
    def add_message(self, message: MessageRecord) -> None:
        """Ajoute un message à l'historique."""
        self._messages.append(message)
    
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
        valid_messages: list[MessageRecord] = []
        
        for message in reversed(self._messages):
            msg_tokens = message.token_count
            
            # Cas limite : conserver au moins le message le plus récent,
            # même s'il dépasse la fenêtre de tokens.
            if not valid_messages and (self.context_window <= 0 or msg_tokens >= self.context_window):
                valid_messages.insert(0, message)
                total_tokens = min(msg_tokens, self.context_window) if self.context_window > 0 else msg_tokens
                continue
            
            if self.context_window <= 0 or total_tokens + msg_tokens <= self.context_window:
                valid_messages.insert(0, message)
                total_tokens += msg_tokens
            else:
                # on continue pour supprimer les messages les plus anciens
                continue
        
        # Nettoyage : supprimer les réponses d'outil orphelines si l'assistant associé n'est plus présent
        cleaned_messages: list[MessageRecord] = []
        assistant_tool_map: list[tuple[AssistantRecord, set[str]]] = []
        resolved_tool_ids: set[str] = set()
        
        for message in valid_messages:
            if message.role == 'assistant' and isinstance(message, AssistantRecord):
                if message.tool_calls:
                    assistant_tool_map.append((message, {tc.id for tc in message.tool_calls}))
                cleaned_messages.append(message)
                continue
            
            if message.role == 'tool' and isinstance(message, ToolResponseRecord):
                tool_id = message.tool_call_id
                if any(tool_id in ids for _, ids in assistant_tool_map):
                    cleaned_messages.append(message)
                    resolved_tool_ids.add(tool_id)
                continue
            
            cleaned_messages.append(message)
        
        if assistant_tool_map:
            skip_tool_ids: set[str] = set()
            final_messages: list[MessageRecord] = []
            
            for message in cleaned_messages:
                if message.role == 'assistant':
                    match = next((ids for assistant, ids in assistant_tool_map if assistant is message), None)
                    if match:
                        if match.issubset(resolved_tool_ids):
                            final_messages.append(message)
                        else:
                            skip_tool_ids.update(match)
                        continue
                
                if message.role == 'tool' and isinstance(message, ToolResponseRecord):
                    if message.tool_call_id in skip_tool_ids:
                        continue
                
                final_messages.append(message)
            
            valid_messages = final_messages
        else:
            valid_messages = cleaned_messages
        
        removed_count = len(self._messages) - len(valid_messages)
        self._messages = valid_messages
    
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
        payload = [m.to_payload() for m in messages]
        
        # Validation : vérifier que les tool responses suivent bien les tool_calls
        tool_responses = [i for i, m in enumerate(payload) if m.get('role') == 'tool']
        assistants_with_tools = [i for i, m in enumerate(payload) if m.get('role') == 'assistant' and m.get('tool_calls')]
        
        # Log minimal si problème détecté
        if tool_responses and not assistants_with_tools:
            logger.warning(f"Tool responses présentes ({len(tool_responses)}) mais aucun assistant avec tool_calls trouvé")
        
        if tool_responses:
            # Vérifier que chaque tool response a un assistant avec tool_calls avant
            for idx in tool_responses:
                tool_call_id = payload[idx].get('tool_call_id')
                # Chercher un assistant avec ce tool_call_id dans les messages précédents
                found = False
                for i in range(idx - 1, -1, -1):
                    if payload[i].get('role') == 'assistant' and payload[i].get('tool_calls'):
                        tool_call_ids = [tc.get('id') for tc in payload[i].get('tool_calls', [])]
                        if tool_call_id in tool_call_ids:
                            found = True
                            break
                if not found:
                    logger.warning(f"Tool response avec tool_call_id={tool_call_id} n'a pas d'assistant correspondant avant")
        
        return payload
    
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
            'window_usage_pct': (total_tokens / self.context_window * 100) if self.context_window > 0 else 0,
            'context_window': self.context_window
        }
    
    def filter_images(self) -> None:
        """Supprime tous les composants image de l'historique.
        
        Utile en cas d'erreur 'invalid_image_url' pour retry sans images.
        """
        for message in self._messages:
            message.components = [c for c in message.components if c.type != 'image_url']
        logger.info("Toutes les images ont été retirées du contexte")

