"""### LLM > API
Façade publique de l'API GPT pour MARI4."""

import logging
from datetime import timedelta
from typing import Callable, Optional, Iterable, Awaitable
from dataclasses import dataclass

import discord

from .client import MariaLLMClient
from .session import ChannelSessionManager, ChannelSession
from .tools import ToolRegistry, Tool
from .context import AssistantRecord, MessageRecord

logger = logging.getLogger(f'MARI4.llm.api')

# RESPONSE --------------------------------------------------------

@dataclass
class MariaResponse:
    """Réponse d'une complétion GPT."""
    text: str
    assistant_record: AssistantRecord
    tool_responses: list
    discord_message: Optional[discord.Message] = None
    
    @property
    def has_tools(self) -> bool:
        """True si des outils ont été utilisés."""
        return len(self.tool_responses) > 0

# SESSION HANDLE --------------------------------------------------

class MariaSessionHandle:
    """Handle public pour accéder à une session (lecture seule)."""
    
    def __init__(self, session: ChannelSession):
        self._session = session
    
    @property
    def channel_id(self) -> int:
        """ID du salon."""
        return self._session.channel_id
    
    def get_stats(self) -> dict:
        """Statistiques de la session."""
        return self._session.get_stats()
    
    def get_recent_messages(self, count: int = 10) -> list[MessageRecord]:
        """Récupère les N derniers messages."""
        return self._session.context.get_recent_messages(count)
    
    def get_context_stats(self) -> dict:
        """Statistiques du contexte."""
        return self._session.context.get_stats()

# API PRINCIPALE --------------------------------------------------

class MariaGptApi:
    """Façade publique de l'API GPT pour MARI4.
    
    Point d'entrée unique pour toutes les interactions avec le système GPT.
    """
    
    def __init__(self,
                 api_key: str,
                 developer_prompt_template: Callable[[], str],
                 *,
                 completion_model: str = 'gpt-5-mini',
                 transcription_model: str = 'gpt-4o-transcribe',
                 max_completion_tokens: int = 1000,
                 context_window: int = 512 * 32,
                 context_age_hours: int = 2,
                 on_completion: Optional[Callable] = None):
        """Initialise l'API GPT.
        
        Args:
            api_key: Clé API OpenAI
            developer_prompt_template: Fonction retournant le prompt système
            completion_model: Modèle pour les complétions
            transcription_model: Modèle pour les transcriptions
            max_completion_tokens: Tokens max par complétion
            context_window: Taille de la fenêtre de contexte en tokens
            context_age_hours: Âge max des messages en heures
            on_completion: Callback optionnel après chaque complétion
        """
        # Client LLM
        self.client = MariaLLMClient(
            api_key=api_key,
            completion_model=completion_model,
            transcription_model=transcription_model,
            max_completion_tokens=max_completion_tokens,
            on_completion=on_completion
        )
        
        # Registre des outils
        self.tool_registry = ToolRegistry()
        
        # Gestionnaire de sessions
        self.session_manager = ChannelSessionManager(
            client=self.client,
            tool_registry=self.tool_registry,
            developer_prompt_template=developer_prompt_template,
            context_window=context_window,
            context_age=timedelta(hours=context_age_hours)
        )
        
    
    async def ensure_session(self, channel: discord.abc.Messageable) -> MariaSessionHandle:
        """Récupère ou crée une session pour un salon.
        
        Args:
            channel: Salon Discord
            
        Returns:
            Handle de session (lecture seule)
        """
        session = self.session_manager.get_or_create_session(channel)
        return MariaSessionHandle(session)
    
    async def ingest_message(self, channel: discord.abc.Messageable, message: discord.Message, is_context_only: bool = False) -> None:
        """Ingère un message Discord dans le contexte d'un salon.
        
        Tous les messages doivent être ingérés pour alimenter le contexte.
        L'ingestion ne déclenche PAS de complétion.
        
        Args:
            channel: Salon Discord
            message: Message à ingérer
            is_context_only: Si True, marque le message comme contexte uniquement
        """
        session = self.session_manager.get_or_create_session(channel)
        await session.ingest_message(message, is_context_only)
    
    async def run_completion(self, 
                            channel: discord.abc.Messageable,
                            trigger_message: Optional[discord.Message] = None,
                            status_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> MariaResponse:
        """Exécute une complétion GPT pour un salon.
        
        Thread-safe : si plusieurs complétions sont demandées simultanément,
        elles seront exécutées séquentiellement.
        
        Args:
            channel: Salon Discord
            trigger_message: Message ayant déclenché la complétion (optionnel)
            status_callback: Callback appelé avec les statuts pendant l'exécution (optionnel)
            
        Returns:
            MariaResponse avec le texte et les métadonnées
        """
        session = self.session_manager.get_or_create_session(channel)
        
        # Exécution (thread-safe via lock interne)
        assistant_record = await session.run_completion(trigger_message, status_callback=status_callback)
        
        # Extraction des tool responses du contexte
        tool_responses = []
        messages = session.context.get_messages()
        
        # Chercher les tool responses après le dernier assistant
        found_assistant = False
        for msg in reversed(messages):
            if msg == assistant_record:
                found_assistant = True
                continue
            if found_assistant and msg.role == 'tool':
                tool_responses.insert(0, msg)
            elif found_assistant and msg.role != 'tool':
                break
        
        # Construire la réponse
        response = MariaResponse(
            text=assistant_record.full_text,
            assistant_record=assistant_record,
            tool_responses=tool_responses
        )
        
        return response
    
    async def run_autonomous_task(self,
                                  channel: discord.abc.Messageable,
                                  user_name: str,
                                  user_id: int,
                                  task_prompt: str) -> MariaResponse:
        """Exécute une tâche autonome sans message Discord déclencheur.
        
        Utilisé pour les tâches planifiées, rappels, etc.
        Injecte directement du texte dans le contexte comme si c'était un message utilisateur.
        
        Args:
            channel: Salon Discord où exécuter la tâche
            user_name: Nom de l'utilisateur concerné
            user_id: ID de l'utilisateur concerné
            task_prompt: Prompt de la tâche à exécuter
            
        Returns:
            MariaResponse avec le texte et les métadonnées
        """
        session = self.session_manager.get_or_create_session(channel)
        
        # Exécution avec prompt direct (thread-safe via lock interne)
        assistant_record = await session.run_autonomous_task(
            user_name=user_name,
            user_id=user_id,
            task_prompt=task_prompt
        )
        
        # Extraction des tool responses du contexte
        tool_responses = []
        messages = session.context.get_messages()
        
        # Chercher les tool responses après le dernier assistant
        found_assistant = False
        for msg in reversed(messages):
            if msg == assistant_record:
                found_assistant = True
                continue
            if found_assistant and msg.role == 'tool':
                tool_responses.insert(0, msg)
            elif found_assistant and msg.role != 'tool':
                break
        
        # Construire la réponse
        response = MariaResponse(
            text=assistant_record.full_text,
            assistant_record=assistant_record,
            tool_responses=tool_responses
        )
        
        return response
    
    async def forget(self, channel: discord.abc.Messageable) -> None:
        """Vide l'historique de conversation d'un salon.
        
        Args:
            channel: Salon Discord
        """
        session = self.session_manager.get_session(channel.id)
        if session:
            session.forget()
    
    def update_tools(self, tools: Iterable[Tool]) -> None:
        """Met à jour le registre des outils.
        
        Remplace les outils existants par les nouveaux.
        
        Args:
            tools: Liste des outils à enregistrer
        """
        self.tool_registry.clear()
        self.tool_registry.register_multiple(*tools)
    
    def add_tools(self, *tools: Tool) -> None:
        """Ajoute des outils au registre.
        
        Args:
            *tools: Outils à ajouter
        """
        self.tool_registry.register_multiple(*tools)
    
    def remove_tool(self, name: str) -> None:
        """Retire un outil du registre.
        
        Args:
            name: Nom de l'outil à retirer
        """
        self.tool_registry.unregister(name)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Récupère un outil par son nom.
        
        Args:
            name: Nom de l'outil
            
        Returns:
            Tool ou None
        """
        return self.tool_registry.get(name)
    
    def get_all_tools(self) -> list[Tool]:
        """Récupère tous les outils enregistrés."""
        return self.tool_registry.get_all()
    
    def get_stats(self) -> dict:
        """Retourne les statistiques globales de l'API."""
        return {
            'client_stats': self.client.get_stats(),
            'session_stats': self.session_manager.get_stats(),
            'tools_count': len(self.tool_registry)
        }
    
    async def close(self) -> None:
        """Ferme l'API et libère les ressources."""
        await self.client.close()

