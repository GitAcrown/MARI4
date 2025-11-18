"""### Chat Cog
Chatbot principal utilisant la nouvelle API GPT."""

import asyncio
import json
import logging
import re
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Literal, Union, Optional
from collections import deque

import discord
from discord import Interaction, app_commands, ui
from discord.ext import commands

from common import dataio
from common.llm import MariaGptApi, Tool, ToolCallRecord, ToolResponseRecord, AssistantRecord
from common.memory import MemoryManager
from .scheduler import TaskScheduler, ScheduledTask

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

# Fuseau horaire de Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# Base de données pour les tâches planifiées
SCHEDULER_DB_PATH = 'data/scheduler.db'

# Template du prompt développeur
DEVELOPER_PROMPT_TEMPLATE = lambda args: f"""Tu es un bot Discord nommée MARIA conversant dans un salon écrit. Tu es genrée au féminin.

STYLE:
• Soit concise, directe et familière, évite les émojis
• Parle un français correct, sans abréviations sauf courantes
• Adopte le ton de l'historique des messages du salon
• Pas de formules robotiques, de questions subsidiaires inutiles ou de réponses trop verbeuses
• TOUJOURS deviner l'intention et ne pas demander de précisions dans les demandes sauf si absolument nécessaire
• Public cible : jeunes adultes début gen Z, habitués au trash, humour noir, d'internet

CONTEXTE:
• Les messages du salon sont fournis pour contexte, mais tu ne répond qu'au dernier message qui te mentionne ou ceux qui parlent indirectement de toi
• Les messages marqués "[CONTEXTE]" sont juste pour info - ne les commente pas, ne réponds pas à leurs questions

OUTILS:
• Utilise tous les outils de manière proactive, en les combinant et les utilisant de manière autonome et sans demander de permission ou de confirmation

MÉMOIRE:
• Utilise update_user_profile uniquement si l'auteur partage une info durable, nouvelle ou une mise à jour (en évitant les doublons)
• L'outil ne doit être utilisé que pour l'auteur du message (le demandeur)
• Ne précise pas explicitement que tu as retenu une information

RECHERCHE:
• Utilise les outils de recherche web de manière PROACTIVE avant de répondre A TOUTE QUESTION dont il te manque des informations (ou si trop récentes/actuelles)
• Utilise read_web_page dès que les extraits de la recherche web ne suffisent pas à répondre à la question
• Adapte la langue de recherche à la demande

FORMAT:
Messages utilisateurs : "[id] username (user_id) : message"
→ "[id] username (user_id)" est un identifiant technique. Ne le reproduis JAMAIS.
→ Le contenu du message est après " : "
→ Tes réponses : écris juste ton texte, sans préfixe ni métadonnées
→ Les données entre '<>' sont des métadonnées, ne les reproduis jamais

{args.get('user_profile', '')}

Date: {args['weekday']} {args['datetime']} (Paris) | Connaissances: sept 2024"""

# PARAMÈTRES ------------------------------------------------------

VALID_CHATBOT_CHANNELS = Union[discord.TextChannel, discord.VoiceChannel, discord.Thread]
MAX_EDITION_AGE = timedelta(minutes=2)

# UI VIEWS --------------------------------------------------------

class InfoView(ui.LayoutView):
    """Vue simple pour afficher les informations du bot."""
    def __init__(self, bot: commands.Bot, channel, context_stats: dict, stats: dict, client_stats: dict, session_stats: dict, config: str, tools: list):
        super().__init__(timeout=300)
        
        container = ui.Container()
        
        # Header
        header = ui.TextDisplay(f"## {bot.user.name}")
        container.add_item(header)
        subtitle = ui.TextDisplay("*Assistante IA pour Discord*")
        container.add_item(subtitle)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.large))
        
        # Session
        session_title = ui.TextDisplay(f"### Session · {channel.name}")
        container.add_item(session_title)
        
        # Dernière réponse
        if stats.get('last_completion'):
            last = stats['last_completion']
            delta = datetime.now(timezone.utc) - last
            minutes = int(delta.total_seconds() / 60)
            last_response = f"il y a {minutes}m"
        else:
            last_response = "jamais"
        
        context_window = context_stats.get('context_window', 0)
        if context_window >= 1000:
            limit_display = f"{context_window / 1000:.1f}k".rstrip('0').rstrip('.')
        elif context_window > 0:
            limit_display = str(context_window)
        else:
            limit_display = "?"
        
        session_text = f"**Messages en contexte** · `{context_stats['total_messages']}`\n"
        session_text += f"**Tokens utilisés** · `{context_stats['total_tokens']} / {limit_display}` ({context_stats['window_usage_pct']:.1f}%)\n"
        session_text += f"**Dernière réponse** · {last_response}"
        
        session_info = ui.TextDisplay(session_text)
        container.add_item(session_info)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Outils
        if tools:
            tools_title = ui.TextDisplay(f"### Outils disponibles · {len(tools)}")
            container.add_item(tools_title)
            
            tools_list = ', '.join([t.name for t in tools[:5]])
            if len(tools) > 5:
                tools_list += f" et {len(tools) - 5} autres"
            tools_display = ui.TextDisplay(tools_list)
            container.add_item(tools_display)
            container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Configuration
        config_title = ui.TextDisplay("### Configuration serveur")
        container.add_item(config_title)
        
        mode_text = {
            'off': 'Désactivé',
            'strict': 'Mentions directes uniquement',
            'greedy': 'Mentions + nom du bot'
        }.get(config, config.upper())
        
        config_info = ui.TextDisplay(f"**Mode** · `{mode_text}`")
        container.add_item(config_info)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Stats globales
        global_title = ui.TextDisplay("### Statistiques globales")
        container.add_item(global_title)
        
        global_text = f"**Complétions** · `{client_stats['completions']}`\n"
        global_text += f"**Transcriptions** · `{client_stats['transcriptions']}`\n"
        global_text += f"**Sessions actives** · `{session_stats['active_sessions']}`"
        
        global_info = ui.TextDisplay(global_text)
        container.add_item(global_info)
        
        container.add_item(ui.Separator())
        footer = ui.TextDisplay("-# Utilisez `/chatbot` pour configurer MARIA")
        container.add_item(footer)
        
        self.add_item(container)

class TasksListView(ui.LayoutView):
    """Vue pour afficher la liste des tâches planifiées."""
    
    def __init__(self, tasks: list[ScheduledTask], bot: commands.Bot):
        super().__init__(timeout=300)
        self.tasks = tasks
        self.bot = bot
        self._setup_layout()
    
    def _setup_layout(self):
        """Configure la mise en page."""
        container = ui.Container()
        
        # Header
        header = ui.TextDisplay(f"## Tâches planifiées")
        container.add_item(header)
        subtitle = ui.TextDisplay(f"-# {len(self.tasks)} tâche(s) affichée(s)")
        container.add_item(subtitle)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.large))
        
        # Afficher les tâches
        for i, task in enumerate(self.tasks):
            # Statut
            status_text = {
                'pending': 'En attente',
                'completed': 'Terminée',
                'failed': 'Échouée',
                'cancelled': 'Annulée'
            }.get(task.status, task.status)
            
            # Titre de la tâche
            task_title = ui.TextDisplay(f"### Tâche #{task.id} · {status_text}")
            container.add_item(task_title)
            
            # Informations
            execute_at_paris = task.execute_at.astimezone(PARIS_TZ)
            execute_str = execute_at_paris.strftime("%d/%m/%Y à %H:%M")
            execute_timestamp = int(task.execute_at.timestamp())
            
            info_text = f"**Exécution** · {execute_str} (<t:{execute_timestamp}:R>)\n"
            info_text += f"**Utilisateur** · <@{task.user_id}>\n"
            info_text += f"**Description** · {task.task_description}"
            
            task_info = ui.TextDisplay(info_text)
            container.add_item(task_info)
            
            # Séparateur entre les tâches
            if i < len(self.tasks) - 1:
                container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Footer
        container.add_item(ui.Separator())
        footer = ui.TextDisplay("-# Utilisez !canceltask <id> pour annuler une tâche")
        container.add_item(footer)
        
        self.add_item(container)

class UserTasksView(ui.LayoutView):
    """Vue interactive pour gérer ses propres tâches planifiées."""
    
    def __init__(self, tasks: list[ScheduledTask], user: discord.User, scheduler):
        super().__init__(timeout=300)
        self.tasks = tasks
        self.user = user
        self.scheduler = scheduler
        self._setup_layout()
    
    def _setup_layout(self):
        """Configure la mise en page avec boutons interactifs."""
        container = ui.Container()
        
        # Header
        header = ui.TextDisplay(f"## Mes tâches planifiées")
        container.add_item(header)
        subtitle = ui.TextDisplay(f"-# {len(self.tasks)} tâche(s)")
        container.add_item(subtitle)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.large))
        
        if not self.tasks:
            no_tasks = ui.TextDisplay("Aucune tâche en attente pour le moment.")
            container.add_item(no_tasks)
        else:
            # Afficher les tâches en attente avec boutons
            for i, task in enumerate(self.tasks):
                # Informations
                execute_timestamp = int(task.execute_at.timestamp())
                
                info_text = f"### ⏳ Tâche #{task.id}\n"
                info_text += f"**Exécution** · <t:{execute_timestamp}:f> (<t:{execute_timestamp}:R>)\n"
                info_text += f"**Description** · {task.task_description}"
                
                task_info = ui.TextDisplay(info_text)
                
                # Bouton annuler (toutes les tâches affichées sont pending)
                cancel_btn = CancelTaskButton(task.id, self.user.id, self.scheduler)
                
                # Créer la section
                section = ui.Section(task_info, accessory=cancel_btn)
                
                container.add_item(section)
                
                # Séparateur entre les tâches
                if i < len(self.tasks) - 1:
                    container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Footer
        container.add_item(ui.Separator())
        footer = ui.TextDisplay("-# Demandez à MARIA pour programmer de nouvelles tâches")
        container.add_item(footer)
        
        self.add_item(container)

class CancelTaskButton(ui.Button):
    """Bouton pour annuler une tâche."""
    
    def __init__(self, task_id: int, user_id: int, scheduler):
        super().__init__(
            style=discord.ButtonStyle.danger,
            label="Annuler",
            custom_id=f"cancel_task_{task_id}"
        )
        self.task_id = task_id
        self.user_id = user_id
        self.scheduler = scheduler
    
    async def callback(self, interaction: discord.Interaction):
        """Annule la tâche quand le bouton est cliqué."""
        # Vérifier que c'est bien l'utilisateur concerné
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Cette tâche ne vous appartient pas.",
                ephemeral=True
            )
            return
        
        # Annuler la tâche
        success = self.scheduler.cancel_task(self.task_id, user_id=self.user_id)
        
        if success:
            # Rafraîchir l'affichage directement (sans message)
            tasks = self.scheduler.get_user_tasks(self.user_id)
            new_view = UserTasksView(tasks, interaction.user, self.scheduler)
            await interaction.response.edit_message(view=new_view)
        else:
            await interaction.response.send_message(
                f"Impossible d'annuler la tâche #{self.task_id}.",
                ephemeral=True
            )

class ProfileModal(ui.Modal, title="Votre profil"):
    """Modal pour afficher et modifier le profil utilisateur."""
    
    def __init__(self, memory_manager, user_id: int, current_content: str):
        super().__init__()
        self.memory_manager = memory_manager
        self.user_id = user_id
        
        self.content_input = ui.TextInput(
            label="Profil",
            style=discord.TextStyle.paragraph,
            placeholder="Informations sur vous (nom, âge, métier, compétences, préférences...)",
            default=current_content if current_content else None,
            min_length=10,
            max_length=1000,
            required=False
        )
        self.add_item(self.content_input)
    
    async def on_submit(self, interaction: Interaction):
        """Sauvegarde les modifications ou supprime le profil."""
        new_content = self.content_input.value.strip()
        
        # Si vide, supprimer le profil
        if not new_content:
            success = self.memory_manager.delete_profile(self.user_id)
            if success:
                await interaction.response.send_message(
                    "Profil supprimé.",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "Aucun profil à supprimer.",
                    ephemeral=True
                )
            return
        
        # Sinon, sauvegarder
        profile = self.memory_manager.get_profile(self.user_id)
        if profile:
            profile.content = new_content
            profile.updated_at = datetime.now(timezone.utc)
        else:
            from common.memory import UserProfile
            profile = UserProfile(
                user_id=self.user_id,
                content=new_content,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                messages_since_update=0
            )
        
        self.memory_manager._save_profile(profile)
        
        # Convertir UTC vers Paris pour l'affichage
        last_update_paris = profile.updated_at.astimezone(PARIS_TZ)
        last_update = last_update_paris.strftime("%d/%m/%y à %H:%M")
        
        await interaction.response.send_message(
            f"**Profil mis à jour**\n\n{new_content}\n\n-# Mis à jour le {last_update}",
            ephemeral=True
        )

# COG -------------------------------------------------------------

class Chat(commands.Cog):
    """Cog principal du chatbot avec nouvelle API GPT."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Configuration par serveur
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'chatbot_mode': 'strict'  # off, strict, greedy
            }
        )
        self.data.map_builders(discord.Guild, guild_config)
        
        # Système de mémoire
        self.memory = MemoryManager(api_key=self.bot.config['OPENAI_API_KEY'])
        self.memory.start_background_updater()
        
        # Système de tâches planifiées
        self.scheduler = TaskScheduler(
            db_path=SCHEDULER_DB_PATH,
            executor=self._execute_autonomous_task
        )
        
        # Fonction pour générer le prompt système dynamiquement
        def get_developer_prompt():
            """Génère le prompt système avec date/heure actuelle et profils utilisateurs."""
            now = datetime.now(PARIS_TZ)
            user_profile = getattr(get_developer_prompt, '_user_profile', '')
            return DEVELOPER_PROMPT_TEMPLATE({
                'weekday': now.strftime('%A'),
                'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
                'user_profile': user_profile
            })
        
        self._get_developer_prompt = get_developer_prompt
        
        # Initialisation de l'API GPT
        self.gpt_api = MariaGptApi(
            api_key=self.bot.config['OPENAI_API_KEY'],
            developer_prompt_template=get_developer_prompt,
            completion_model='gpt-5-mini',
            transcription_model='gpt-4o-transcribe',
            max_completion_tokens=1024,
            context_window=24576,  # 24k tokens
            context_age_hours=2  # 2h pour cohérence avec DEFAULT_CONTEXT_AGE
        )
        
        # Messages déjà traités (éviter doublons)
        self._processed_messages = deque(maxlen=100)
        
    
    async def cog_load(self):
        """Appelé quand le cog est chargé."""
        # Démarrer le worker de tâches planifiées
        await self.scheduler.start_worker()
    
    async def cog_unload(self):
        """Appelé quand le cog est déchargé."""
        # Arrêter le worker proprement
        await self.scheduler.stop_worker()
        await self.memory.close()
        await self.gpt_api.close()
        self.data.close_all()
    
    async def _register_tools_from_cogs(self):
        """Enregistre les outils depuis les autres cogs."""
        tools = []
        for cog in self.bot.cogs.values():
            if cog.qualified_name == self.qualified_name:
                continue
            if hasattr(cog, 'GLOBAL_TOOLS'):
                tools.extend(cog.GLOBAL_TOOLS)
        
        update_profile_tool = Tool(
            name='update_user_profile',
            description=(
                "Enregistre les infos UTILES de l'auteur pour personnaliser les futures interactions avec lui. "
                "A RETENIR: identité (prenom, age, metier, localisation), préférences de communication (ton, niveau de détail, sujets à éviter), "
                "contexte personnel durable (projets, compétences, centres d'intérêt récurrents), contraintes spécifiques. "
                "A NE PAS RETENIR: opinions temporaires, actions ponctuelles, questions posées, infos sur d'autres personnes, faits généraux. "
                "Utilise uniquement si l'auteur partage une info durable et nouvelle. JAMAIS pour d'autres personnes."
            ),
            properties={},
            function=self._tool_update_user_profile
        )
        tools.append(update_profile_tool)
        
        schedule_task_tool = Tool(
            name='schedule_task',
            description=(
                "Programme une tâche à exécuter plus tard de manière autonome. "
                "Tu pourras utiliser tous tes outils (recherche web, etc.) au moment de l'exécution. "
                "UTILISE POUR: rappels, recherches différées, messages programmés. "
                "EXEMPLES: 'Rappelle-moi de sortir les poubelles dans 2h', 'Recherche les news sur SpaceX demain matin', "
                "'Envoie un résumé des messages importants ce soir à 20h'. "
                "Le système te réveillera automatiquement au moment prévu pour exécuter la tâche."
            ),
            properties={
                'task_description': {
                    'type': 'string',
                    'description': "Description claire de la tâche à exécuter (ex: 'Rappeler à l'utilisateur de sortir les poubelles')"
                },
                'delay_minutes': {
                    'type': 'integer',
                    'description': "Délai en minutes avant l'exécution (ex: 120 pour 2 heures)"
                },
                'delay_hours': {
                    'type': 'integer',
                    'description': "Délai en heures avant l'exécution (ex: 24 pour demain)"
                }
            },
            function=self._tool_schedule_task
        )
        tools.append(schedule_task_tool)
        
        cancel_task_tool = Tool(
            name='cancel_scheduled_task',
            description=(
                "Annule une tâche programmée précédemment. "
                "L'utilisateur doit être l'auteur de la tâche pour pouvoir l'annuler. "
                "UTILISE POUR: annuler un rappel, supprimer une tâche programmée. "
                "EXEMPLES: 'Annule le rappel', 'Oublie la tâche que je t'ai demandée'. "
                "Tu dois connaître l'ID de la tâche (visible quand elle est créée)."
            ),
            properties={
                'task_id': {
                    'type': 'integer',
                    'description': "ID de la tâche à annuler (reçu lors de la création)"
                }
            },
            function=self._tool_cancel_scheduled_task
        )
        tools.append(cancel_task_tool)
        
        if tools:
            self.gpt_api.add_tools(*tools)
    
    # EXÉCUTEUR DE TÂCHES -----------------------------------------
    
    async def _execute_autonomous_task(self, channel_id: int, user_id: int, task_description: str, message_id: int = 0):
        """Exécute une tâche autonome via l'API GPT propre."""
        # Récupérer le salon
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Salon {channel_id} introuvable (supprimé ou bot expulsé)")
            return  # Ne pas raise, juste skip la tâche
        
        # Récupérer l'utilisateur
        try:
            user = await self.bot.fetch_user(user_id)
        except:
            logger.warning(f"Utilisateur {user_id} introuvable")
            return  # Ne pas raise, juste skip la tâche
        
        # Récupérer le message d'origine si disponible
        original_message = None
        if message_id:
            try:
                original_message = await channel.fetch_message(message_id)
            except:
                logger.warning(f"Message d'origine {message_id} introuvable")
                # Continuer sans reply
        
        # Injecter le profil utilisateur dans le prompt développeur
        author_profile = self.memory.get_profile_text(user.id)
        if author_profile:
            self._get_developer_prompt._user_profile = f"PROFILS:\n\n**{user.name}** (auteur):\n{author_profile}\n"
        else:
            self._get_developer_prompt._user_profile = ''
        
        # Formater le prompt de la tâche avec instructions spéciales
        task_prompt = f"""[TÂCHE AUTONOME PROGRAMMÉE]
Demande initiale de {user.name} ({user.id}) : {task_description}

Tu exécutes maintenant la tâche que TU as programmée précédemment à la demande de {user.name}.
IMPORTANT :
- Ne pose AUCUNE question subsidiaire (personne ne répondra)
- Fais de ton mieux avec les informations dont tu disposes
- Utilise tes outils (recherche web, etc.) si nécessaire
- Réponds comme si tu t'adressais directement à la personne concernée
- Ne rajoute pas de compte rendu ou de résumé administratif en fin de message
- Si tu ne peux pas tout faire, explique précisément ce que tu as réussi à accomplir"""
        
        # Exécuter via l'API propre (sans fake message et sans typing)
        try:
            response = await self.gpt_api.run_autonomous_task(
                channel=channel,
                user_name=user.name,
                user_id=user.id,
                task_prompt=task_prompt
            )
        finally:
            # Toujours réinitialiser le profil injecté pour ne pas contaminer les prochaines requêtes
            self._get_developer_prompt._user_profile = ''
        
        # Formater la réponse avec headers des outils
        text = response.text
        if response.tool_responses:
            headers = [tr.metadata.get('header') for tr in response.tool_responses if tr.metadata.get('header')]
            if headers:
                headers = list(dict.fromkeys(headers))
                text = '\n-# ' + '\n-# '.join(headers) + '\n' + text
        
        # Ajouter un footer avec mention si pas de message d'origine
        if not original_message:
            text += f"\n\n-# <@{user.id}>"
        
        # Envoyer en reply au message d'origine ou normalement
        try:
            if len(text) <= 2000:
                if original_message:
                    await original_message.reply(text)
                else:
                    await channel.send(text)
            else:
                # Découper si nécessaire
                chunks = []
                while len(text) > 1900:
                    chunk = text[:1900]
                    text = text[1900:]
                    chunks.append(chunk)
                chunks.append(text)
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        if original_message:
                            await original_message.reply(chunk)
                        else:
                            await channel.send(chunk)
                    else:
                        await channel.send(chunk)
        except discord.Forbidden:
            logger.error(f"Pas de permission pour envoyer dans {channel.id}")
            raise  # Re-raise pour marquer la tâche comme failed
        except discord.HTTPException as e:
            logger.error(f"Erreur HTTP Discord: {e}")
            raise  # Re-raise pour marquer la tâche comme failed
    
    # OUTILS ------------------------------------------------------
    
    async def _tool_schedule_task(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Programme une tâche à exécuter plus tard de manière autonome."""
        if not context_data or not hasattr(context_data, 'trigger_message') or not context_data.trigger_message:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Message déclencheur introuvable"},
                created_at=datetime.now(timezone.utc)
            )
        
        trigger_message = context_data.trigger_message
        args = tool_call.arguments
        
        # Vérifier la limite de tâches par utilisateur (max 10 en attente)
        pending_count = self.scheduler.count_pending_user_tasks(trigger_message.author.id)
        if pending_count >= 10:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': f"Limite atteinte : {pending_count}/10 tâches en attente. Annulez-en avant d'en créer d'autres."},
                created_at=datetime.now(timezone.utc)
            )
        
        # Extraire les paramètres avec validation
        task_description = args.get('task_description', '').strip()
        delay_minutes = args.get('delay_minutes', 0)
        delay_hours = args.get('delay_hours', 0)
        
        # Validation : Description non vide
        if not task_description:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Description de tâche manquante"},
                created_at=datetime.now(timezone.utc)
            )
        
        # Validation : Description pas trop longue (max 500 caractères)
        if len(task_description) > 500:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Description trop longue (max 500 caractères)"},
                created_at=datetime.now(timezone.utc)
            )
        
        # Calculer le délai total
        total_minutes = delay_minutes + (delay_hours * 60)
        
        # Validation : Délai minimum 2 minutes (pour éviter les conflits avec le message en cours)
        if total_minutes < 2:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Délai minimum : 2 minutes"},
                created_at=datetime.now(timezone.utc)
            )
        
        # Validation : Délai max 30 jours (empêcher l'abus)
        if total_minutes > 43200:  # 30 jours = 43200 minutes
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Délai maximum : 30 jours"},
                created_at=datetime.now(timezone.utc)
            )
        
        # Calculer la date d'exécution
        execute_at = datetime.now(timezone.utc) + timedelta(minutes=total_minutes)
        
        # Enregistrer la tâche via le scheduler (avec message_id pour reply)
        task_id = self.scheduler.schedule_task(
            channel_id=trigger_message.channel.id,
            user_id=trigger_message.author.id,
            task_description=task_description,
            execute_at=execute_at,
            message_id=trigger_message.id
        )
        
        # Formatter le délai pour l'affichage
        def _format_delay_text(total_min: int) -> str:
            days, rem_minutes = divmod(total_min, 1440)
            hours, minutes = divmod(rem_minutes, 60)
            parts = []
            if days:
                parts.append(f"{days} j")
            if hours:
                parts.append(f"{hours} h")
            if minutes:
                parts.append(f"{minutes} min")
            if not parts:
                return "moins d'une minute"
            return " ".join(parts)

        delay_text = _format_delay_text(total_minutes)
        
        # Convertir en heure de Paris pour l'affichage
        execute_at_paris = execute_at.astimezone(PARIS_TZ)
        execute_time = execute_at_paris.strftime("%H:%M")
        
        return ToolResponseRecord(
            tool_call_id=tool_call.id,
            response_data={
                'success': True,
                'task_id': task_id,
                'execute_at': execute_at.isoformat(),
                'delay': delay_text
            },
            created_at=datetime.now(timezone.utc),
            metadata={'header': f"Tâche programmée dans {delay_text} (vers {execute_time})"}
        )
    
    async def _tool_cancel_scheduled_task(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Annule une tâche planifiée précédemment."""
        if not context_data or not hasattr(context_data, 'trigger_message') or not context_data.trigger_message:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Message déclencheur introuvable"},
                created_at=datetime.now(timezone.utc)
            )
        
        trigger_message = context_data.trigger_message
        args = tool_call.arguments
        
        task_id = args.get('task_id')
        if not task_id:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "ID de tâche manquant"},
                created_at=datetime.now(timezone.utc)
            )
        
        # Annuler uniquement si la tâche appartient à l'utilisateur
        success = self.scheduler.cancel_task(task_id, user_id=trigger_message.author.id)
        
        if success:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'success': True, 'task_id': task_id},
                created_at=datetime.now(timezone.utc),
                metadata={'header': f"Tâche #{task_id} annulée"}
            )
        else:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': f"Tâche #{task_id} introuvable ou déjà terminée"},
                created_at=datetime.now(timezone.utc)
            )
    
    async def _tool_update_user_profile(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Met à jour le profil de l'auteur du message avec les infos récentes.
        
        Appelé par l'IA quand l'utilisateur partage des infos personnelles importantes.
        """
        if not context_data or not hasattr(context_data, 'trigger_message') or not context_data.trigger_message:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Message déclencheur introuvable"},
                created_at=datetime.now(timezone.utc)
            )
        
        trigger_message = context_data.trigger_message
        user_id = trigger_message.author.id
        user_name = trigger_message.author.name
        
        # Récupérer les 20 derniers messages de l'utilisateur
        recent_messages = []
        async for msg in trigger_message.channel.history(limit=20):
            if msg.author.id == user_id and not msg.author.bot:
                recent_messages.append(msg)
        
        if not recent_messages:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Aucun message récent trouvé"},
                created_at=datetime.now(timezone.utc)
            )
        
        success = await self.memory.force_update(user_id, recent_messages)
        new_content = self.memory.get_profile_text(user_id)
        if success:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'result': "Profil mis à jour avec succès.", 'new_content': new_content},
                created_at=datetime.now(timezone.utc),
                metadata={'header': f"Mise à jour du profil de ***{user_name}***"}
            )
        else:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': "Échec de la mise à jour"},
                created_at=datetime.now(timezone.utc)
            )
    
    # CONFIGURATION -----------------------------------------------
    
    def get_guild_config(self, guild: discord.Guild, key: str, cast: Union[type, None] = None):
        """Récupère une clé de configuration du serveur."""
        value = self.data.get(guild).get_dict_value('guild_config', key)
        if cast is not None and value is not None:
            return cast(value)
        return value
    
    def set_guild_config(self, guild: discord.Guild, key: str, value: Union[str, int, bool]) -> None:
        """Met à jour une clé de configuration du serveur."""
        self.data.get(guild).set_dict_value('guild_config', key, value)
    
    # DÉTECTION DE RÉPONSE ----------------------------------------
    
    def should_respond(self, message: discord.Message) -> bool:
        """Détermine si le bot doit répondre à un message."""
        if message.author.bot:
            return False
        
        mode = self.get_guild_config(message.guild, 'chatbot_mode', str)
        
        if mode == 'off':
            return False
        
        # Mode strict: mentions uniquement
        if mode == 'strict':
            mentioned = self.bot.user.mentioned_in(message)
            return mentioned
        
        # Mode greedy: mentions + nom du bot
        if mode == 'greedy':
            # Vérifier mention directe
            if self.bot.user.mentioned_in(message):
                return True
            
            # Chercher le nom complet du bot dans le message (insensible à la casse)
            bot_name_lower = self.bot.user.name.lower()
            message_lower = message.content.lower()
            
            # Seulement le nom complet
            if re.search(rf'\b{re.escape(bot_name_lower)}\b', message_lower):
                return True
            
        
        return False
    
    async def should_use_reply(self, message: discord.Message) -> bool:
        """Détermine si on doit utiliser reply ou un message normal.
        
        Utilise reply si :
        - Il y a eu plus de 3 messages dans les 2 dernières minutes
        - Ou si plusieurs personnes ont parlé récemment
        
        Sinon, message normal pour une conversation plus fluide.
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=2)
            
            recent_messages = []
            async for msg in message.channel.history(limit=10, after=cutoff):
                if not msg.author.bot:
                    recent_messages.append(msg)
            
            # Si plus de 3 messages récents, utiliser reply (salon actif)
            if len(recent_messages) > 3:
                return True
            
            # Si plusieurs auteurs différents, utiliser reply (conversation multi-personnes)
            authors = set(msg.author.id for msg in recent_messages)
            if len(authors) > 1:
                return True
            
            # Sinon, message normal (salon calme, conversation 1-1)
            return False
            
        except Exception as e:
            logger.error(f"Erreur should_use_reply: {e}")
            # En cas d'erreur, utiliser reply par défaut
            return True
    
    # EVENTS ------------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Écoute tous les messages pour ingestion et réponse."""
        if not isinstance(message.channel, VALID_CHATBOT_CHANNELS):
            return
        
        if message.author.bot:
            return
        
        # Décider si on répond
        should_respond = self.should_respond(message)
        
        # Ingérer le message (marqué comme contexte si le bot ne répond pas)
        await self.gpt_api.ingest_message(message.channel, message, is_context_only=not should_respond)
        
        if not should_respond:
            return
        
        # Éviter les doublons
        if message.id in self._processed_messages:
            return
        
        self._processed_messages.append(message.id)
        
        # Injecter les profils utilisateurs pertinents dans le prompt
        profiles_to_inject = []
        
        # 1. Profil de l'auteur (toujours prioritaire)
        author_profile = self.memory.get_profile_text(message.author.id)
        if author_profile:
            profiles_to_inject.append(f"**{message.author.name}** (auteur):\n{author_profile}")
        
        # 2. Profils des utilisateurs mentionnés
        for mentioned_user in message.mentions:
            if mentioned_user.bot or mentioned_user.id == message.author.id:
                continue
            
            mentioned_profile = self.memory.get_profile_text(mentioned_user.id)
            if mentioned_profile:
                profiles_to_inject.append(f"**{mentioned_user.name}** (mentionné):\n{mentioned_profile}")
        
        # 3. Profils des utilisateurs récemment actifs (max 2, seulement si pas de mentions)
        if not message.mentions:
            recent_users = set()
            async for msg in message.channel.history(limit=10):
                if msg.author.bot or msg.author.id == message.author.id:
                    continue
                recent_users.add(msg.author)
                if len(recent_users) >= 2:
                    break
            
            for recent_user in recent_users:
                recent_profile = self.memory.get_profile_text(recent_user.id)
                if recent_profile:
                    profiles_to_inject.append(f"**{recent_user.name}** (actif):\n{recent_profile}")
        
        # Injecter dans le prompt système
        if profiles_to_inject:
            self._get_developer_prompt._user_profile = "PROFILS:\n\n" + "\n\n".join(profiles_to_inject) + "\n"
        else:
            self._get_developer_prompt._user_profile = ''
        
        # Répondre
        async with message.channel.typing():
            # Message de statut (sera édité avec la réponse finale)
            status_message: Optional[discord.Message] = None
            
            async def update_status(status_text: str):
                """Callback pour mettre à jour le message de statut.
                
                Remplace le statut précédent au lieu de l'accumuler.
                """
                nonlocal status_message
                
                # Créer le message de statut s'il n'existe pas encore
                if status_message is None:
                    use_reply = await self.should_use_reply(message)
                    
                    try:
                        if use_reply:
                            status_message = await message.reply(
                                status_text,
                                mention_author=False,
                                allowed_mentions=discord.AllowedMentions.none()
                            )
                        else:
                            status_message = await message.channel.send(
                                status_text,
                                allowed_mentions=discord.AllowedMentions.none()
                            )
                    except Exception as e:
                        logger.warning(f"Erreur lors de la création du message de statut: {e}")
                else:
                    # Remplacer le statut précédent (un seul statut à la fois)
                    try:
                        await status_message.edit(content=status_text)
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'édition du message de statut: {e}")
            
            try:
                response = await self.gpt_api.run_completion(
                    message.channel,
                    trigger_message=message,
                    status_callback=update_status
                )
                
                # Gestion de la mémoire : en arrière-plan pour ne pas bloquer la réponse
                async def update_memory_background():
                    try:
                        recent_messages = []
                        async for msg in message.channel.history(limit=20):
                            if msg.author.id == message.author.id and not msg.author.bot:
                                recent_messages.append(msg)
                        
                        if recent_messages:
                            self.memory.increment_message_count(message.author.id)
                            await self.memory.check_and_schedule_update(message.author.id, recent_messages)
                    except Exception as e:
                        logger.debug(f"Erreur mise à jour mémoire en arrière-plan: {e}")
                
                # Lancer la mise à jour de la mémoire en arrière-plan
                asyncio.create_task(update_memory_background())
                
                # Formater la réponse avec headers des outils
                text = response.text
                if response.tool_responses:
                    headers = [tr.metadata.get('header') for tr in response.tool_responses if tr.metadata.get('header')]
                    if headers:
                        headers = list(dict.fromkeys(headers))  # Dédupliquer
                        text = '\n-# ' + '\n-# '.join(headers) + '\n' + text
                
                # Si on a un message de statut, l'éditer avec la réponse finale
                if status_message:
                    try:
                        # Découper si nécessaire (limite Discord 2000 caractères)
                        if len(text) <= 2000:
                            await status_message.edit(content=text)
                        else:
                            # Si trop long, envoyer en plusieurs messages
                            chunks = []
                            remaining = text
                            while len(remaining) > 2000:
                                chunk = remaining[:2000]
                                remaining = remaining[2000:]
                                chunks.append(chunk)
                            chunks.append(remaining)
                            
                            # Éditer le premier message avec le premier chunk
                            await status_message.edit(content=chunks[0])
                            
                            # Envoyer les autres chunks
                            for chunk in chunks[1:]:
                                await message.channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())
                    except Exception as e:
                        logger.error(f"Erreur lors de l'édition du message final: {e}")
                        # Fallback : envoyer un nouveau message
                        use_reply = await self.should_use_reply(message)
                        if use_reply:
                            await message.reply(text, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                        else:
                            await message.channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                else:
                    # Pas de message de statut, envoyer normalement
                    use_reply = await self.should_use_reply(message)
                    
                    # Découper si nécessaire (limite Discord 2000 caractères)
                    if len(text) <= 2000:
                        if use_reply:
                            await message.reply(text, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                        else:
                            await message.channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                    else:
                        # Découper en morceaux
                        chunks = []
                        while len(text) > 2000:
                            chunk = text[:2000]
                            text = text[2000:]
                            chunks.append(chunk)
                        chunks.append(text)
                        
                        for i, chunk in enumerate(chunks):
                            if use_reply and i == 0:
                                await message.reply(chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                            else:
                                await message.channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())
                
            except Exception as e:
                logger.error(f"Erreur lors de la complétion: {e}")
                # Supprimer le message de statut s'il existe
                if status_message:
                    try:
                        await status_message.delete()
                    except:
                        pass
                await message.reply("❌ Une erreur est survenue lors du traitement de votre message.", 
                                  mention_author=False)
    
    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Gère les éditions de messages récents."""
        if not isinstance(after.channel, VALID_CHATBOT_CHANNELS):
            return
        
        if after.author.bot:
            return
        
        # Ignorer si déjà traité
        if after.id in self._processed_messages:
            return
        
        # Ignorer si trop vieux
        if after.created_at < datetime.now(timezone.utc) - MAX_EDITION_AGE:
            return
        
        # Ingérer et potentiellement répondre
        await self.gpt_api.ingest_message(after.channel, after)
        
        if not self.should_respond(after):
            return
        
        self._processed_messages.append(after.id)
        
        async with after.channel.typing():
            try:
                response = await self.gpt_api.run_completion(
                    after.channel,
                    trigger_message=after
                )
                
                text = response.text
                
                # Headers des tools
                if response.tool_responses:
                    headers = []
                    for tool_resp in response.tool_responses:
                        header = tool_resp.metadata.get('header')
                        if header:
                            headers.append(header)
                    
                    if headers:
                        headers = list(dict.fromkeys(headers))
                        headers_text = '\n-# ' + '\n-# '.join(headers) + '\n'
                        text = headers_text + text
                
                # Décider si on utilise reply ou message normal
                use_reply = await self.should_use_reply(after)
                
                # Envoyer
                if len(text) <= 2000:
                    if use_reply:
                        await after.reply(text, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                    else:
                        await after.channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                else:
                    chunks = []
                    while len(text) > 2000:
                        chunk = text[:2000]
                        text = text[2000:]
                        chunks.append(chunk)
                    chunks.append(text)
                    
                    for i, chunk in enumerate(chunks):
                        if use_reply and i == 0:
                            await after.reply(chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                        else:
                            await after.channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())
                
            except Exception as e:
                logger.error(f"Erreur lors de la complétion (edit): {e}")
    
    # COMMANDES ---------------------------------------------------
    
    @app_commands.command(name='info')
    async def cmd_info(self, interaction: Interaction):
        """Affiche des informations sur le bot et son utilisation."""
        if not interaction.guild:
            return await interaction.response.send_message(
                "Cette commande n'est pas disponible en DM.",
                ephemeral=True
            )
        
        # Récupérer les stats
        handle = await self.gpt_api.ensure_session(interaction.channel)
        stats = handle.get_stats()
        context_stats = stats['context_stats']
        global_stats = self.gpt_api.get_stats()
        client_stats = global_stats['client_stats']
        session_stats = global_stats['session_stats']
        
        # Configuration
        config = self.get_guild_config(interaction.guild, 'chatbot_mode', str)
        
        # Outils
        tools = self.gpt_api.get_all_tools()
        
        # Créer la vue
        view = InfoView(
            bot=self.bot,
            channel=interaction.channel,
            context_stats=context_stats,
            stats=stats,
            client_stats=client_stats,
            session_stats=session_stats,
            config=config,
            tools=tools
        )
        
        await interaction.response.send_message(view=view, ephemeral=True)
    
    # Groupe de commandes chatbot
    chatbot_group = app_commands.Group(
        name='chatbot',
        description="Configuration du chatbot",
        default_permissions=discord.Permissions(manage_messages=True),
        guild_only=True
    )
    
    @chatbot_group.command(name='forget')
    async def chatbot_forget(self, interaction: Interaction):
        """Supprime l'historique de conversation du salon."""
        await self.gpt_api.forget(interaction.channel)
        await interaction.response.send_message(
            "✅ Historique de conversation supprimé pour ce salon.",
            ephemeral=True
        )
    
    @chatbot_group.command(name='mode')
    @app_commands.choices(
        mode=[
            app_commands.Choice(name='Désactivé', value='off'),
            app_commands.Choice(name='Mentions directes uniquement', value='strict'),
            app_commands.Choice(name='Mentions + nom du bot', value='greedy')
        ]
    )
    async def chatbot_mode(self, interaction: Interaction, mode: Literal['off', 'strict', 'greedy']):
        """Configure le mode de réponse du chatbot.
        
        :param mode: Mode de réponse du chatbot
        """
        # Répondre immédiatement pour éviter le timeout
        mode_text = {
            'off': 'désactivé',
            'strict': 'mentions directes uniquement',
            'greedy': 'mentions + nom du bot'
        }.get(mode, mode)
        
        await interaction.response.send_message(
            f"✅ Mode du chatbot modifié: **{mode_text}**",
            ephemeral=True
        )
        
        # Puis sauvegarder la config
        self.set_guild_config(interaction.guild, 'chatbot_mode', mode)
    
    @app_commands.command(name='tasks')
    async def tasks_cmd(self, interaction: Interaction):
        """Affiche et gérez vos tâches planifiées."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        # Récupérer les tâches de l'utilisateur
        tasks = self.scheduler.get_user_tasks(interaction.user.id, limit=20)
        
        # Créer la vue interactive
        view = UserTasksView(tasks, interaction.user, self.scheduler)
        await interaction.followup.send(view=view, ephemeral=True)
    
    @app_commands.command(name='memory')
    async def memory(self, interaction: Interaction):
        """Affiche et modifie votre profil enregistré par MARIA."""
        profile = self.memory.get_profile(interaction.user.id)
        current_content = profile.content if profile else ""
        
        # Ouvrir le modal directement
        modal = ProfileModal(self.memory, interaction.user.id, current_content)
        await interaction.response.send_modal(modal)
    
    # COMMANDES ADMIN (prefix commands) ------------------------------
    
    @commands.command(name='tasks')
    @commands.is_owner()
    async def cmd_tasks(self, ctx: commands.Context):
        """[Admin] Liste toutes les tâches planifiées."""
        tasks = self.scheduler.get_all_tasks(limit=20)
        
        if not tasks:
            await ctx.send("Aucune tâche planifiée.")
            return
        
        # Créer la vue LayoutView
        view = TasksListView(tasks, self.bot)
        await ctx.send(view=view)

    @commands.command(name='contextpeek')
    @commands.is_owner()
    async def cmd_contextpeek(self, ctx: commands.Context, limit: int = 15):
        """[Admin] Affiche un extrait des derniers messages en contexte pour ce salon."""
        limit = max(1, min(limit, 50))
        
        session = self.gpt_api.session_manager.get_session(ctx.channel.id)
        if not session:
            await ctx.send("Aucun historique enregistré pour ce salon.")
            return
        
        messages = session.context.get_recent_messages(limit)
        if not messages:
            await ctx.send("Historique vide pour ce salon.")
            return
        
        entries = []
        for msg in messages:
            created = msg.created_at.astimezone(PARIS_TZ).strftime("%d/%m %H:%M:%S")
            role = msg.role.upper()
            
            flags = []
            if msg.metadata.get('autonomous_task'):
                flags.append("auto")
            
            if isinstance(msg, AssistantRecord) and msg.tool_calls:
                flags.append("tools=" + ','.join(tc.function_name for tc in msg.tool_calls))
            
            header = f"[{created}] {role}"
            if flags:
                header += " (" + ", ".join(flags) + ")"
            
            if msg.role == 'tool' and isinstance(msg, ToolResponseRecord):
                preview = json.dumps(msg.response_data, ensure_ascii=False)
            else:
                preview = msg.full_text.strip()
            
            preview = re.sub(r"\s+", " ", preview) or "[contenu vide]"
            if len(preview) > 200:
                preview = preview[:197] + "…"
            
            entries.append(f"{header}\n→ {preview}")
        
        blocks = []
        current = ""
        for entry in entries:
            addition = entry + "\n\n"
            if len(current) + len(addition) > 1900:
                blocks.append(current.rstrip())
                current = ""
            current += addition
        if current.strip():
            blocks.append(current.rstrip())
        
        for block in blocks:
            await ctx.send(f"```\n{block}\n```")
    
    @commands.command(name='canceltask')
    @commands.is_owner()
    async def cmd_cancel_task(self, ctx: commands.Context, task_id: int):
        """[Admin] Annule une tâche planifiée."""
        success = self.scheduler.cancel_task(task_id)
        if success:
            await ctx.send(f"Tâche #{task_id} annulée.")
        else:
            await ctx.send(f"Tâche #{task_id} introuvable ou déjà terminée.")
    
    @commands.command(name='profiles')
    @commands.is_owner()
    async def cmd_profiles(self, ctx: commands.Context):
        """[Admin] Liste tous les profils enregistrés."""
        stats = self.memory.get_stats()
        
        if stats['total_profiles'] == 0:
            await ctx.send("Aucun profil enregistré.")
            return
        
        # Récupérer tous les profils
        cursor = self.memory.conn.execute(
            'SELECT user_id, content, updated_at, messages_since_update FROM user_profiles ORDER BY updated_at DESC LIMIT 20'
        )
        rows = cursor.fetchall()
        
        output = f"**Profils enregistrés** ({stats['total_profiles']} total, 20 derniers)\n\n"
        
        for row in rows:
            user_id = row['user_id']
            content = row['content']
            updated_at = datetime.fromisoformat(row['updated_at'])
            messages_since = row['messages_since_update']
            
            # Essayer de récupérer le nom de l'utilisateur
            try:
                user = await self.bot.fetch_user(user_id)
                user_name = user.name
            except:
                user_name = f"User {user_id}"
            
            # Convertir en heure de Paris
            updated_paris = updated_at.astimezone(PARIS_TZ)
            time_str = updated_paris.strftime("%d/%m %H:%M")
            
            # Tronquer le contenu
            preview = content[:100] + "..." if len(content) > 100 else content
            
            output += f"**{user_name}** (`{user_id}`)\n"
            output += f"MAJ: {time_str} · Messages: {messages_since}\n"
            output += f"{preview}\n\n"
            
            # Découper si trop long
            if len(output) > 1800:
                await ctx.send(output)
                output = ""
        
        if output:
            await ctx.send(output)
    
    @commands.command(name='profile')
    @commands.is_owner()
    async def cmd_profile(self, ctx: commands.Context, user_id: int):
        """[Admin] Affiche le profil complet d'un utilisateur."""
        profile = self.memory.get_profile(user_id)
        
        if not profile:
            await ctx.send(f"Aucun profil pour l'utilisateur `{user_id}`.")
            return
        
        # Essayer de récupérer le nom
        try:
            user = await self.bot.fetch_user(user_id)
            user_name = user.name
        except:
            user_name = f"User {user_id}"
        
        # Convertir les dates
        created_paris = profile.created_at.astimezone(PARIS_TZ)
        updated_paris = profile.updated_at.astimezone(PARIS_TZ)
        
        output = f"**Profil de {user_name}** (`{user_id}`)\n\n"
        output += f"**Créé:** {created_paris.strftime('%d/%m/%Y à %H:%M')}\n"
        output += f"**Mis à jour:** {updated_paris.strftime('%d/%m/%Y à %H:%M')}\n"
        output += f"**Messages depuis MAJ:** {profile.messages_since_update}\n\n"
        output += f"**Contenu:**\n{profile.content}"
        
        # Découper si nécessaire
        if len(output) <= 2000:
            await ctx.send(output)
        else:
            chunks = []
            while len(output) > 2000:
                chunk = output[:2000]
                output = output[2000:]
                chunks.append(chunk)
            chunks.append(output)
            
            for chunk in chunks:
                await ctx.send(chunk)
 
async def setup(bot):
    await bot.add_cog(Chat(bot))

