"""### LLM > Session
Gestion des sessions de conversation par salon avec contrôle de concurrence."""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Callable

import discord

from .client import MariaLLMClient
from .context import (
    ConversationContext, MessageRecord, AssistantRecord, 
    TextComponent, ImageComponent, MetadataComponent,
    ToolCallRecord, ContentComponent
)
from .tools import ToolRegistry
from .attachments import AttachmentCache, process_attachment

logger = logging.getLogger(f'MARI4.llm.session')

# CONSTANTES ------------------------------------------------------

# Format par défaut pour les messages utilisateur
DEFAULT_USER_FORMAT = '[{message.id}] {message.author.name} ({message.author.id})'

# CHANNEL SESSION -------------------------------------------------

class ChannelSession:
    """Session de conversation pour un salon Discord.
    
    Gère le contexte, les locks de concurrence et l'orchestration des complétions.
    """
    
    def __init__(self,
                 channel_id: int,
                 client: MariaLLMClient,
                 tool_registry: ToolRegistry,
                 attachment_cache: AttachmentCache,
                 developer_prompt_template: Callable[[], str],
                 **context_kwargs):
        """Initialise une session de salon.
        
        Args:
            channel_id: ID du salon Discord
            client: Client LLM
            tool_registry: Registre des outils
            attachment_cache: Cache des pièces jointes
            developer_prompt_template: Fonction retournant le prompt développeur
            **context_kwargs: Arguments pour ConversationContext
        """
        self.channel_id = channel_id
        self.client = client
        self.tool_registry = tool_registry
        self.attachment_cache = attachment_cache
        self.developer_prompt_template = developer_prompt_template
        
        # Contexte de conversation
        self.context = ConversationContext(
            developer_prompt=developer_prompt_template(),
            **context_kwargs
        )
        
        # Lock pour éviter les race conditions
        self._lock = asyncio.Lock()
        
        # Statistiques
        self._stats = {
            'completions': 0,
            'messages_ingested': 0,
            'last_completion': None
        }
        
        # Message ayant déclenché la complétion en cours
        self.trigger_message: Optional[discord.Message] = None
        
        logger.info(f"ChannelSession créée pour salon {channel_id}")
    
    async def ingest_message(self, message: discord.Message, is_context_only: bool = False) -> MessageRecord:
        """Ingère un message Discord dans le contexte.
        
        Traite les pièces jointes, références, etc.
        Thread-safe via lock.
        
        Args:
            message: Message Discord à ingérer
            is_context_only: Si True, marque le message comme contexte uniquement
            
        Returns:
            MessageRecord créé
        """
        async with self._lock:
            return await self._ingest_message_unsafe(message, is_context_only)
    
    async def _ingest_message_unsafe(self, message: discord.Message, is_context_only: bool = False) -> MessageRecord:
        """Version non-thread-safe de l'ingestion (utilisée en interne)."""
        components = []
        
        # Contenu texte
        if message.content:
            user_format = DEFAULT_USER_FORMAT.format(message=message)
            context_marker = "[CONTEXTE] " if is_context_only else ""
            components.append(TextComponent(f"{context_marker}{user_format}: {message.clean_content}"))
            
            # Extraction URLs d'images dans le texte
            for match in re.finditer(r'(https?://[^\s]+)', message.content):
                url = match.group(0)
                clean_url = re.sub(r'\?.*$', '', url)
                if clean_url.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    components.append(ImageComponent(clean_url, detail='auto'))
                # Pour les GIFs, utiliser le proxy Discord pour avoir une frame statique
                elif clean_url.lower().endswith('.gif'):
                    # Ajouter ?format=png pour forcer une image statique
                    static_url = f"{url}?format=png" if '?' not in url else f"{url}&format=png"
                    components.append(ImageComponent(static_url, detail='auto'))
        
        # Embeds
        for embed in message.embeds:
            if embed.title or embed.description or embed.url:
                components.append(MetadataComponent('EMBED',
                                                   embed_title=embed.title,
                                                   embed_description=embed.description,
                                                   embed_url=embed.url))
            if embed.image and embed.image.url:
                url = embed.image.url
                # Pour les GIFs, utiliser le proxy Discord pour avoir une frame statique
                if url.lower().endswith('.gif'):
                    url = f"{url}?format=png" if '?' not in url else f"{url}&format=png"
                components.append(ImageComponent(url, detail='high'))
            if embed.thumbnail and embed.thumbnail.url:
                url = embed.thumbnail.url
                # Pour les GIFs, utiliser le proxy Discord pour avoir une frame statique
                if url.lower().endswith('.gif'):
                    url = f"{url}?format=png" if '?' not in url else f"{url}&format=png"
                components.append(ImageComponent(url, detail='low'))
        
        # Stickers
        for sticker in message.stickers:
            if sticker.url:
                components.append(ImageComponent(sticker.url, detail='auto'))
        
        # Attachments images directes
        for attachment in message.attachments:
            content_type = attachment.content_type or ''
            filename_lower = attachment.filename.lower()
            
            if (content_type.startswith('image/') or 
                filename_lower.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))):
                url = attachment.url
                # Pour les GIFs, utiliser le proxy Discord pour avoir une frame statique
                if filename_lower.endswith('.gif'):
                    url = f"{url}?format=png" if '?' not in url else f"{url}&format=png"
                components.append(ImageComponent(url, detail='auto'))
        
        # Références
        if message.reference and message.reference.resolved:
            ref_msg = message.reference.resolved
            # Si référence au bot
            if ref_msg.author.bot:  # Approximation, devrait vérifier l'ID exact
                start = ref_msg.content[:200].replace('\n', ' ')
                if len(ref_msg.content) > 200:
                    start += '...'
                components.append(MetadataComponent('REFERENCE', yourself=True, starting_with=start))
            else:
                components.append(MetadataComponent('REFERENCE', message_id=ref_msg.id))
        
        # Créer le record
        record = self.context.add_user_message(
            components=components,
            name=message.author.name,
            discord_message=message
        )
        
        self._stats['messages_ingested'] += 1
        logger.debug(f"Message {message.id} ingéré dans salon {self.channel_id}")
        
        return record
    
    async def process_attachments(self, message: discord.Message) -> list[ContentComponent]:
        """Traite les pièces jointes d'un message et retourne les composants.
        
        Args:
            message: Message Discord
            
        Returns:
            Liste de composants générés par le traitement
        """
        components = []
        
        for attachment in message.attachments:
            processed = await process_attachment(attachment, self.client, self.attachment_cache)
            components.extend(processed)
        
        return components
    
    async def run_completion(self, trigger_message: Optional[discord.Message] = None) -> AssistantRecord:
        """Exécute une complétion GPT.
        
        Thread-safe via lock - une seule complétion à la fois.
        
        Args:
            trigger_message: Message Discord ayant déclenché la complétion (optionnel)
            
        Returns:
            AssistantRecord avec la réponse
        """
        async with self._lock:
            return await self._run_completion_unsafe(trigger_message)
    
    async def _run_completion_unsafe(self, trigger_message: Optional[discord.Message]) -> AssistantRecord:
        """Version non-thread-safe de la complétion."""
        # Stocker le trigger_message pour les outils
        self.trigger_message = trigger_message
        
        # Traiter les pièces jointes du dernier message si nécessaire
        if trigger_message:
            attachment_components = await self.process_attachments(trigger_message)
            if attachment_components:
                # Ajouter au dernier message utilisateur
                recent = self.context.get_recent_messages(1)
                if recent and recent[0].role == 'user':
                    recent[0].components.extend(attachment_components)
        
        # Mettre à jour le prompt développeur
        self.context.developer_prompt = self.developer_prompt_template()
        
        # Préparer le payload
        try:
            messages = self.context.prepare_payload()
        except Exception as e:
            logger.error(f"Erreur préparation payload: {e}")
            raise
        
        # Outils compilés
        tools = self.tool_registry.get_compiled() if len(self.tool_registry) > 0 else []
        
        # Appel API
        try:
            completion = await self.client.create_completion(
                messages=messages,
                tools=tools if tools else None
            )
        except Exception as e:
            # Retry sans images en cas d'erreur invalid_image_url
            if 'invalid_image_url' in str(e):
                logger.warning("Erreur invalid_image_url, retry sans images")
                self.context.filter_images()
                messages = self.context.prepare_payload()
                completion = await self.client.create_completion(
                    messages=messages,
                    tools=tools if tools else None
                )
            else:
                raise
        
        # Conversion en AssistantRecord
        choice = completion.choices[0]
        message_obj = choice.message
        
        # Composants
        components = []
        if message_obj.content:
            components.append(TextComponent(message_obj.content))
        else:
            components.append(MetadataComponent('EMPTY'))
        
        # Tool calls
        tool_calls = []
        if message_obj.tool_calls:
            for tc in message_obj.tool_calls:
                tool_calls.append(ToolCallRecord(
                    id=tc.id,
                    function_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
        
        # Créer l'assistant record
        assistant_record = self.context.add_assistant_message(
            components=components,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason
        )
        
        # Exécuter les tools si nécessaire
        if tool_calls:
            await self._execute_tools(tool_calls)
            # Récursion pour obtenir la réponse finale
            return await self._run_completion_unsafe(None)
        
        # Si la réponse est vide (pas de contenu), retry une fois
        if not message_obj.content or not message_obj.content.strip():
            logger.warning("Réponse vide reçue, retry...")
            # Supprimer le message vide du contexte
            if self.context._messages and self.context._messages[-1] == assistant_record:
                self.context._messages.pop()
            # Ajouter un message système pour forcer une réponse
            self.context.add_user_message(
                components=[TextComponent("[SYSTEM] Reponds maintenant au dernier message.")],
                name="system"
            )
            return await self._run_completion_unsafe(None)
        
        self._stats['completions'] += 1
        self._stats['last_completion'] = datetime.now(timezone.utc)
        
        logger.info(f"Complétion réussie pour salon {self.channel_id}")
        return assistant_record
    
    async def _execute_tools(self, tool_calls: list[ToolCallRecord]) -> None:
        """Exécute les appels d'outils et ajoute les réponses au contexte."""
        for tool_call in tool_calls:
            tool = self.tool_registry.get(tool_call.function_name)
            if not tool:
                logger.warning(f"Outil '{tool_call.function_name}' non trouvé")
                continue
            
            try:
                response = await tool.execute(tool_call, context_data=self)
                self.context.add_message(response)
                logger.debug(f"Outil '{tool_call.function_name}' exécuté")
            except Exception as e:
                logger.error(f"Erreur exécution outil '{tool_call.function_name}': {e}")
    
    def forget(self) -> None:
        """Vide l'historique de conversation."""
        self.context.clear()
        logger.info(f"Historique vidé pour salon {self.channel_id}")
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la session."""
        return {
            **self._stats,
            'context_stats': self.context.get_stats()
        }

# SESSION MANAGER -------------------------------------------------

class ChannelSessionManager:
    """Gestionnaire de sessions par salon."""
    
    def __init__(self,
                 client: MariaLLMClient,
                 tool_registry: ToolRegistry,
                 developer_prompt_template: Callable[[], str],
                 **default_context_kwargs):
        """Initialise le gestionnaire.
        
        Args:
            client: Client LLM
            tool_registry: Registre des outils
            developer_prompt_template: Template du prompt développeur
            **default_context_kwargs: Arguments par défaut pour les contextes
        """
        self.client = client
        self.tool_registry = tool_registry
        self.developer_prompt_template = developer_prompt_template
        self.default_context_kwargs = default_context_kwargs
        
        # Cache partagé des attachments
        self.attachment_cache = AttachmentCache()
        
        # Sessions actives
        self._sessions: dict[int, ChannelSession] = {}
        
        logger.info("ChannelSessionManager initialisé")
    
    def get_or_create_session(self, channel: discord.abc.Messageable) -> ChannelSession:
        """Récupère ou crée une session pour un salon.
        
        Args:
            channel: Salon Discord
            
        Returns:
            ChannelSession
        """
        if channel.id not in self._sessions:
            self._sessions[channel.id] = ChannelSession(
                channel_id=channel.id,
                client=self.client,
                tool_registry=self.tool_registry,
                attachment_cache=self.attachment_cache,
                developer_prompt_template=self.developer_prompt_template,
                **self.default_context_kwargs
            )
            logger.info(f"Nouvelle session créée pour salon {channel.id}")
        
        return self._sessions[channel.id]
    
    def get_session(self, channel_id: int) -> Optional[ChannelSession]:
        """Récupère une session existante."""
        return self._sessions.get(channel_id)
    
    def remove_session(self, channel_id: int) -> None:
        """Supprime une session."""
        if channel_id in self._sessions:
            del self._sessions[channel_id]
            logger.info(f"Session supprimée pour salon {channel_id}")
    
    def get_all_sessions(self) -> list[ChannelSession]:
        """Retourne toutes les sessions actives."""
        return list(self._sessions.values())
    
    def get_stats(self) -> dict:
        """Retourne les statistiques globales."""
        return {
            'active_sessions': len(self._sessions),
            'cache_stats': self.attachment_cache.get_stats()
        }

