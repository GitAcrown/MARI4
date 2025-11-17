"""### LLM > Session
Gestion des sessions de conversation par salon avec contrôle de concurrence."""

import asyncio
import copy
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable

import discord

from .client import MariaLLMClient
from .context import (
    ConversationContext, MessageRecord, AssistantRecord, 
    TextComponent, ImageComponent, MetadataComponent,
    ToolCallRecord, ContentComponent, ToolResponseRecord
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
        # Compteur des appels d'outils dans la complétion en cours
        self._tool_call_counter: Optional[defaultdict[str, int]] = None
        
        # Compteur global de tool calls pour éviter les boucles infinies
        self._total_tool_calls_in_session: int = 0
        self._max_tool_calls_per_completion = 20  # Limite globale
        
        # Message ayant déclenché la complétion en cours
        self.trigger_message: Optional[discord.Message] = None
        
    
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
            # Extraire le contenu du message référencé
            ref_content = ref_msg.content or ""
            if ref_msg.embeds:
                # Si le message n'a pas de contenu mais a des embeds, utiliser la description
                for embed in ref_msg.embeds:
                    if embed.description:
                        ref_content = embed.description
                        break
            
            # Limiter la longueur pour éviter de surcharger le contexte
            if len(ref_content) > 300:
                ref_preview = ref_content[:300].replace('\n', ' ') + '...'
            else:
                ref_preview = ref_content.replace('\n', ' ') if ref_content else "(message sans texte)"
            
            # Si référence au bot
            if ref_msg.author.bot:
                components.append(MetadataComponent('REFERENCE', yourself=True, starting_with=ref_preview))
            else:
                # Référence à un autre utilisateur : inclure auteur et contenu
                author_name = ref_msg.author.name
                components.append(MetadataComponent('REFERENCE', 
                                                   author=author_name,
                                                   message_id=ref_msg.id,
                                                   content=ref_preview))
        
        # Créer le record
        record = self.context.add_user_message(
            components=components,
            name=message.author.name,
            discord_message=message
        )
        
        self._stats['messages_ingested'] += 1
        
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
    
    async def run_completion(self, 
                            trigger_message: Optional[discord.Message] = None,
                            status_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> AssistantRecord:
        """Exécute une complétion GPT.
        
        Thread-safe via lock - une seule complétion à la fois.
        
        Args:
            trigger_message: Message Discord ayant déclenché la complétion (optionnel)
            status_callback: Callback appelé avec les statuts pendant l'exécution (optionnel)
            
        Returns:
            AssistantRecord avec la réponse
        """
        async with self._lock:
            return await self._run_completion_unsafe(trigger_message, status_callback=status_callback)
    
    async def run_autonomous_task(self, user_name: str, user_id: int, task_prompt: str) -> AssistantRecord:
        """Exécute une tâche autonome sans message Discord.
        
        Exécute la tâche de manière ISOLÉE du contexte récent pour éviter toute contamination,
        puis ajoute le résultat au contexte partagé.
        
        Thread-safe via lock - une seule complétion à la fois.
        
        Args:
            user_name: Nom de l'utilisateur concerné
            user_id: ID de l'utilisateur
            task_prompt: Prompt de la tâche à exécuter
            
        Returns:
            AssistantRecord avec la réponse
        """
        async with self._lock:
            # Contexte isolé pour l'exécution de la tâche
            isolated_context = ConversationContext(
                developer_prompt=self.developer_prompt_template(),
                context_window=self.context.context_window,
                context_age=self.context.context_age
            )

            # Message utilisateur synthétique (non réinjecté ensuite)
            isolated_context.add_user_message(
                components=[TextComponent(task_prompt)],
                name=user_name,
                discord_message=None,
                autonomous_task=True,
                task_owner_id=user_id
            )

            # Sauvegarder l'état courant
            original_context = self.context
            original_tool_counter = self._tool_call_counter
            self.context = isolated_context
            self._tool_call_counter = None

            try:
                # Indice à partir duquel copier les messages (on ignore le user synthétique)
                start_index = len(isolated_context._messages)

                # Exécuter la complétion dans le contexte isolé
                final_assistant = await self._run_completion_unsafe(None)

                # Collecter les nouveaux messages (assistant + outils)
                new_messages = isolated_context._messages[start_index:]

            finally:
                # Restaurer le contexte original
                self.context = original_context
                self._tool_call_counter = original_tool_counter

            def _clone_component(component: ContentComponent) -> ContentComponent:
                """Clone léger d'un composant pour le réinjecter dans le contexte principal."""
                if component.type == 'text':
                    return TextComponent(component.data.get('text', ''))
                if component.type == 'image_url':
                    image_data = component.data.get('image_url', {})
                    return ImageComponent(
                        image_data.get('url', ''),
                        detail=image_data.get('detail', 'auto')
                    )
                # Fallback: transformer en simple texte
                return TextComponent(component.data.get('text', ''))

            assistant_record: Optional[AssistantRecord] = None

            for message in new_messages:
                if message.role == 'assistant':
                    cloned_components = [_clone_component(comp) for comp in message.components]
                    cloned_tool_calls = []
                    if isinstance(message, AssistantRecord) and message.tool_calls:
                        for call in message.tool_calls:
                            cloned_tool_calls.append(
                                ToolCallRecord(
                                    id=call.id,
                                    function_name=call.function_name,
                                    arguments=copy.deepcopy(call.arguments)
                                )
                            )

                    metadata = dict(message.metadata)
                    metadata['autonomous_task'] = True
                    metadata['task_owner_id'] = user_id

                    assistant_record = self.context.add_assistant_message(
                        components=cloned_components,
                        tool_calls=cloned_tool_calls or None,
                        finish_reason=getattr(message, 'finish_reason', None),
                        metadata=metadata
                    )

                elif message.role == 'tool' and isinstance(message, ToolResponseRecord):
                    # Ne pas réinjecter les réponses d'outils des tâches autonomes
                    # pour éviter de polluer le contexte principal avec du JSON.
                    continue

            if assistant_record is None:
                # Sécurité : ne jamais retourner None
                assistant_record = self.context.add_assistant_message(
                    components=[TextComponent("Échec de la tâche autonome (réponse vide).")],
                    metadata={
                        'autonomous_task': True,
                        'task_owner_id': user_id,
                        'error': True
                    }
                )

            return assistant_record
    
    async def _run_completion_unsafe(self, 
                                    trigger_message: Optional[discord.Message],
                                    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
                                    recursion_depth: int = 0) -> AssistantRecord:
        """Version non-thread-safe de la complétion.
        
        Args:
            trigger_message: Message Discord déclencheur
            status_callback: Callback pour les statuts
            recursion_depth: Profondeur de récursion (pour éviter les boucles infinies)
        """
        # Limite de récursion pour éviter les boucles infinies
        MAX_RECURSION_DEPTH = 10
        if recursion_depth >= MAX_RECURSION_DEPTH:
            logger.warning(f"Limite de récursion atteinte ({MAX_RECURSION_DEPTH}), arrêt des tool calls")
            # Retourner une réponse d'erreur
            return self.context.add_assistant_message(
                components=[TextComponent("Désolée, j'ai atteint la limite d'appels d'outils. Peux-tu reformuler ta demande de manière plus simple ?")],
                created_at=datetime.now(timezone.utc)
            )
        
        is_root_call = self._tool_call_counter is None
        if is_root_call:
            self._tool_call_counter = defaultdict(int)
            self._total_tool_calls_in_session = 0  # Réinitialiser au début d'une nouvelle complétion
        
        # Vérifier la limite globale de tool calls
        if self._total_tool_calls_in_session >= self._max_tool_calls_per_completion:
            logger.warning(f"Limite globale de tool calls atteinte ({self._max_tool_calls_per_completion}), arrêt")
            return self.context.add_assistant_message(
                components=[TextComponent("Désolée, j'ai fait trop d'appels d'outils. Peux-tu reformuler ta demande de manière plus précise ?")],
                created_at=datetime.now(timezone.utc)
            )
        
        try:
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
                    self._total_tool_calls_in_session += 1
            
            # Créer l'assistant record
            assistant_record = self.context.add_assistant_message(
                components=components,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason
            )
            
            # Exécuter les tools si nécessaire
            if tool_calls:
                await self._execute_tools(tool_calls, status_callback=status_callback)
                # Vérifier que les tool responses sont bien dans le contexte
                recent_tool_responses = [m for m in self.context.get_recent_messages(10) if m.role == 'tool']
                if not recent_tool_responses:
                    logger.warning("Aucune tool response trouvée dans le contexte après exécution des outils")
                # Récursion pour obtenir la réponse finale (incrémenter la profondeur)
                return await self._run_completion_unsafe(None, status_callback=status_callback, recursion_depth=recursion_depth + 1)
            
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
                return await self._run_completion_unsafe(None, status_callback=status_callback, recursion_depth=recursion_depth + 1)
            
            self._stats['completions'] += 1
            self._stats['last_completion'] = datetime.now(timezone.utc)
            
            return assistant_record
        finally:
            if is_root_call:
                self._tool_call_counter = None
    
    async def _execute_tools(self, 
                           tool_calls: list[ToolCallRecord],
                           status_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> None:
        """Exécute les appels d'outils et ajoute les réponses au contexte."""
        for tool_call in tool_calls:
            if self._tool_call_counter is None:
                self._tool_call_counter = defaultdict(int)
            
            if tool_call.function_name == 'schedule_task':
                self._tool_call_counter[tool_call.function_name] += 1
                if self._tool_call_counter[tool_call.function_name] > 5:
                    logger.warning("Limite de planification en chaîne atteinte")
                    response = ToolResponseRecord(
                        tool_call_id=tool_call.id,
                        response_data={
                            'error': "Limite atteinte : maximum 5 tâches programmées dans la même exécution."
                        },
                        created_at=datetime.now(timezone.utc),
                        metadata={'header': "Limite de planification atteinte"}
                    )
                    self.context.add_message(response)
                    continue
            else:
                self._tool_call_counter[tool_call.function_name] += 1
            
            tool = self.tool_registry.get(tool_call.function_name)
            if not tool:
                logger.warning(f"Outil '{tool_call.function_name}' non trouvé")
                continue
            
            # Générer un message de statut pour le callback
            if status_callback:
                status_msg = self._get_tool_status_message(tool_call)
                if status_msg:
                    try:
                        await status_callback(status_msg)
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'appel du callback de statut: {e}")
            
            try:
                response = await tool.execute(tool_call, context_data=self)
                self.context.add_message(response)
            except Exception as e:
                logger.error(f"Erreur exécution outil '{tool_call.function_name}': {e}")
    
    def _get_tool_status_message(self, tool_call: ToolCallRecord) -> Optional[str]:
        """Génère un message de statut pour un appel d'outil.
        
        Utilise le même formatage que les headers finaux pour la cohérence.
        """
        args = tool_call.arguments
        
        if tool_call.function_name == 'search_web':
            query = args.get('query', '')
            if query:
                # Utiliser le même format que le header final
                return f"-# Recherche de \"{query}\" •••"
        
        elif tool_call.function_name == 'read_web_page':
            url = args.get('url', '')
            if url:
                # Utiliser le même formatage que le header final (lien cliquable)
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc or url.split("//")[-1].split("/")[0]
                    return f"-# Lecture de [{domain}](<{url}>) •••"
                except:
                    return "-# Lecture d'une page web •••"
            return "-# Lecture d'une page web •••"
        
        elif tool_call.function_name == 'schedule_task':
            task_desc = args.get('task_description', '')
            if task_desc:
                if len(task_desc) > 40:
                    task_desc = task_desc[:37] + '...'
                return f"-# Planification de tâche : {task_desc} •••"
            return "-# Planification d'une tâche •••"
        
        elif tool_call.function_name == 'cancel_scheduled_task':
            return "-# Annulation d'une tâche •••"
        
        elif tool_call.function_name == 'update_user_profile':
            return "-# Mise à jour du profil •••"
        
        # Pour les autres outils, pas de message de statut
        return None
    
    def forget(self) -> None:
        """Vide l'historique de conversation."""
        self.context.clear()
    
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
        
        return self._sessions[channel.id]
    
    def get_session(self, channel_id: int) -> Optional[ChannelSession]:
        """Récupère une session existante."""
        return self._sessions.get(channel_id)
    
    def remove_session(self, channel_id: int) -> None:
        """Supprime une session."""
        if channel_id in self._sessions:
            del self._sessions[channel_id]
    
    def get_all_sessions(self) -> list[ChannelSession]:
        """Retourne toutes les sessions actives."""
        return list(self._sessions.values())
    
    def get_stats(self) -> dict:
        """Retourne les statistiques globales."""
        return {
            'active_sessions': len(self._sessions),
            'cache_stats': self.attachment_cache.get_stats()
        }

