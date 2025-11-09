"""### Chat Cog
Chatbot principal utilisant la nouvelle API GPT."""

import logging
import re
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Literal, Union
from collections import deque

import discord
from discord import Interaction, app_commands, ui
from discord.ext import commands

from common import dataio
from common.llm import MariaGptApi, Tool, ToolCallRecord, ToolResponseRecord
from common.memory import MemoryManager

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

# Fuseau horaire de Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# Template du prompt développeur
DEVELOPER_PROMPT_TEMPLATE = lambda args: f"""Tu es le bot MARIA sur Discord. Parle comme une vraie personne, genrée au féminin.

STYLE & COMPORTEMENT:
• Concise, directe, décontractée, langage gen Z d'internet, évite les émojis
• Adapte le ton au contexte (sérieuse, sarcastique, etc.)
• Pas de formules robotiques ou de follow-up inutiles
• Devine l'intention, ne demande pas de précisions
• Encouragée à donner des avis et opinions si le contexte s'y prête

CONTEXTE & MESSAGES:
• Les messages du salon sont fournis pour contexte, mais tu ne réponds QU'AUX MESSAGES QUI TE MENTIONNENT ou qui parlent indirectement de toi
• Les messages marqués "[CONTEXTE]" sont juste pour info - ne les commente pas, ne réponds pas à leurs questions
• Réponds uniquement au dernier message qui t'a mentionnée (celui sans "[CONTEXTE]")

MÉMOIRE:
• Utilise update_user_profile uniquement si l'auteur partage une info durable et nouvelle (identité, contexte de vie, limites, préférences explicites de ton, surnom, sujets à éviter, etc.)
• Ignore les infos temporaires ou évidentes si elles sont déjà dans le profil injecté
• Pas de doublon: si l'info est déjà présente presque à l'identique, ne rappelle pas l'outil
• Ne demande pas la permission, fais-le naturellement MAIS uniquement pour l'auteur du message
• Ne précise pas forcément explicitement que tu retiens une information

RECHERCHE:
• Info recente inconnue ? Utilise search_web pour avoir des extraits
• Si les extraits suffisent, reponds directement
• Si besoin de plus de details ou si l'utilisateur donne une URL, utilise read_web_page
• Adapte la langue de recherche au contexte
• Fais confiance aux resultats, ne dis JAMAIS que tu n'as pas acces a internet

FORMAT:
Messages utilisateurs : "[id] username (user_id) : message"
→ Le "[id] username (user_id)" est un identifiant technique. Ne le reproduis JAMAIS.
→ Le vrai message est après " : "
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
        
        session_text = f"**Messages en contexte** · `{context_stats['total_messages']}`\n"
        session_text += f"**Tokens utilisés** · `{context_stats['total_tokens']} / 16k` ({context_stats['window_usage_pct']:.1f}%)\n"
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
        footer = ui.TextDisplay("-# Utilisez /chatbot pour configurer MARIA")
        container.add_item(footer)
        
        self.add_item(container)

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
            context_window=32768,  # 32k tokens
            context_age_hours=6
        )
        
        # Messages déjà traités (éviter doublons)
        self._processed_messages = deque(maxlen=100)
        
        logger.info("Chat cog initialisé avec API GPT")
    
    async def cog_load(self):
        """Appelé quand le cog est chargé."""
        # Les outils seront enregistrés dans bot.py après que tous les cogs soient chargés
        pass
    
    async def cog_unload(self):
        """Appelé quand le cog est déchargé."""
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
                logger.info(f"Outils chargés depuis {cog.qualified_name}: {len(cog.GLOBAL_TOOLS)}")
        
        update_profile_tool = Tool(
            name='update_user_profile',
            description=(
                "Enregistre les infos UTILES de l'auteur pour personnaliser les futures interactions. "
                "A RETENIR: identite (prenom, age, metier, localisation), preferences de communication (ton, niveau de detail, sujets a eviter), "
                "contexte personnel durable (projets, competences, centres d'interet recurrents), contraintes specifiques. "
                "A NE PAS RETENIR: opinions temporaires, actions ponctuelles, questions posees, infos sur d'autres personnes, faits generaux. "
                "Utilise uniquement si l'auteur partage une info durable et nouvelle. JAMAIS pour d'autres personnes."
            ),
            properties={},
            function=self._tool_update_user_profile
        )
        tools.append(update_profile_tool)
        
        if tools:
            self.gpt_api.add_tools(*tools)
            logger.info(f"Total outils enregistrés: {len(tools)}")
    
    # OUTILS ------------------------------------------------------
    
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
        
        if success:
            logger.info(f"Profil de {user_name} mis à jour par l'IA")
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'result': "Profil mis à jour avec succès."},
                created_at=datetime.now(timezone.utc),
                metadata={'header': f"Mise à jour du profil de {user_name}"}
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
            logger.debug(f"Mode strict - Mentioned: {mentioned}")
            return mentioned
        
        # Mode greedy: mentions + nom du bot
        if mode == 'greedy':
            # Vérifier mention directe
            if self.bot.user.mentioned_in(message):
                logger.debug(f"Mode greedy - Mention directe détectée")
                return True
            
            # Chercher le nom complet du bot dans le message (insensible à la casse)
            bot_name_lower = self.bot.user.name.lower()
            message_lower = message.content.lower()
            
            # Seulement le nom complet
            if re.search(rf'\b{re.escape(bot_name_lower)}\b', message_lower):
                logger.debug(f"Mode greedy - Nom du bot '{self.bot.user.name}' détecté")
                return True
            
            logger.debug(f"Mode greedy - Aucune correspondance trouvée")
        
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
            try:
                response = await self.gpt_api.run_completion(
                    message.channel,
                    trigger_message=message
                )
                
                # Gestion de la mémoire : incrémenter compteur et planifier MAJ si nécessaire
                recent_messages = []
                async for msg in message.channel.history(limit=20):
                    if msg.author.id == message.author.id and not msg.author.bot:
                        recent_messages.append(msg)
                
                if recent_messages:
                    self.memory.increment_message_count(message.author.id)
                    await self.memory.check_and_schedule_update(message.author.id, recent_messages)
                
                # Formater la réponse avec headers des outils
                text = response.text
                if response.tool_responses:
                    headers = [tr.metadata.get('header') for tr in response.tool_responses if tr.metadata.get('header')]
                    if headers:
                        headers = list(dict.fromkeys(headers))  # Dédupliquer
                        text = '\n-# ' + '\n-# '.join(headers) + '\n' + text
                
                # Décider si on utilise reply ou message normal
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
    
    @app_commands.command(name='memory')
    async def memory(self, interaction: Interaction):
        """Affiche et modifie votre profil enregistré par MARIA."""
        profile = self.memory.get_profile(interaction.user.id)
        current_content = profile.content if profile else ""
        
        # Ouvrir le modal directement
        modal = ProfileModal(self.memory, interaction.user.id, current_content)
        await interaction.response.send_modal(modal)
 
async def setup(bot):
    await bot.add_cog(Chat(bot))

