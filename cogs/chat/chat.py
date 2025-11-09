"""### Chat Cog
Chatbot principal utilisant la nouvelle API GPT."""

import logging
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Literal, Union
from collections import deque

import discord
from discord import Interaction, app_commands, ui
from discord.ext import commands

from common import dataio
from common.llm import MariaGptApi
from common.memory import MemoryManager

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

# Fuseau horaire de Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# Template du prompt développeur
DEVELOPER_PROMPT_TEMPLATE = lambda args: f"""Tu es le bot MARIA sur Discord. Parle comme une vraie personne.

{args.get('user_profile', '')}

STYLE & COMPORTEMENT:
• Concise, directe, décontractée, langage gen Z d'internet
• Adapte le ton au contexte (sérieuse, sarcastique, etc.)
• Pas de formules robotiques ou de follow-up inutiles
• Devine l'intention, ne demande pas de précisions
• Peux donner avis et opinions librement si le contexte s'y prête

RECHERCHE:
• Info récente inconnue ? Utilise search_and_read (cherche + lit automatiquement)
• Si l'utilisateur donne une URL précise, utilise read_web_page
• Adapte la langue de recherche au contexte

FORMAT:
Messages utilisateurs : "[id] username (user_id) : message"
→ Le "[id] username (user_id)" est un identifiant technique. Ne le reproduis JAMAIS.
→ Le vrai message est après " : "
→ Tes réponses : écris juste ton texte, sans préfixe ni métadonnées
→ Les données entre '<>' sont des métadonnées, ne les reproduis jamais

Date: {args['weekday']} {args['datetime']} (Paris) | Connaissances: sept 2024"""

# PARAMÈTRES ------------------------------------------------------

VALID_CHATBOT_CHANNELS = Union[discord.TextChannel, discord.VoiceChannel, discord.Thread]
MAX_EDITION_AGE = timedelta(minutes=2)

# UI VIEWS --------------------------------------------------------

class InfoView(ui.LayoutView):
    """Vue pour afficher les informations du bot."""
    def __init__(self, bot: commands.Bot, channel: discord.TextChannel, context_stats: dict, stats: dict, global_stats: dict, config: str, tools: list):
        super().__init__(timeout=300)
        self.bot = bot
        
        container = ui.Container()
        
        # Header
        header = ui.TextDisplay(f"## {bot.user.name}")
        container.add_item(header)
        subtitle = ui.TextDisplay("*Assistante IA pour Discord*")
        container.add_item(subtitle)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.large))
        
        # Session info
        session_title = ui.TextDisplay(f"### Session · {channel.name}")
        container.add_item(session_title)
        
        session_text = f"**Messages en contexte** · `{context_stats['total_messages']}`\n"
        session_text += f"**Tokens utilisés** · `{context_stats['total_tokens']} / 16k` ({context_stats['window_usage_pct']:.1f}%)\n"
        if stats.get('last_completion'):
            last = stats['last_completion']
            delta = datetime.now(timezone.utc) - last
            minutes = int(delta.total_seconds() / 60)
            session_text += f"**Dernière réponse** · il y a {minutes}m"
        else:
            session_text += "**Dernière réponse** · jamais"
        
        session_info = ui.TextDisplay(session_text)
        container.add_item(session_info)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Tools
        if tools:
            tools_title = ui.TextDisplay(f"### Outils disponibles · {len(tools)}")
            container.add_item(tools_title)
            
            tools_list = '\n'.join([f"• {t.name}" for t in tools[:5]])
            if len(tools) > 5:
                tools_list += f"\n• ... et {len(tools) - 5} autres"
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
        client_stats = global_stats['client_stats']
        session_stats = global_stats['session_stats']
        
        global_title = ui.TextDisplay("### Statistiques globales")
        container.add_item(global_title)
        
        global_text = f"**Complétions** · `{client_stats['completions']}`\n"
        global_text += f"**Transcriptions** · `{client_stats['transcriptions']}`\n"
        global_text += f"**Sessions actives** · `{session_stats['active_sessions']}`\n"
        global_text += f"**Cache** · `{session_stats['cache_stats']['transcript_cache_size']} transcriptions, {session_stats['cache_stats']['video_cache_size']} vidéos`"
        
        global_info = ui.TextDisplay(global_text)
        container.add_item(global_info)
        
        # Thumbnail
        thumb = ui.Thumbnail(media=bot.user.display_avatar.url)
        container.add_item(thumb)
        
        # Footer
        container.add_item(ui.Separator())
        footer = ui.TextDisplay("-# Utilisez /chatbot pour configurer MARIA")
        container.add_item(footer)
        
        self.add_item(container)

class MemoryProfileView(ui.LayoutView):
    """Vue pour afficher le profil mémoire d'un utilisateur."""
    def __init__(self, user: discord.User, profile):
        super().__init__(timeout=300)
        self.user = user
        
        container = ui.Container()
        
        # Header
        header = ui.TextDisplay(f"## Votre carte d'identité")
        container.add_item(header)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.large))
        
        # Profile content
        profile_text = ui.TextDisplay(f">>> {profile.content}")
        container.add_item(profile_text)
        container.add_item(ui.Separator(spacing=discord.SeparatorSpacing.small))
        
        # Metadata
        messages_count = profile.messages_since_update
        last_update = profile.updated_at.strftime("%d/%m/%Y à %H:%M")
        
        meta_text = f"**Dernière mise à jour** · {last_update}\n"
        meta_text += f"**Messages depuis** · {messages_count}"
        meta_display = ui.TextDisplay(meta_text)
        container.add_item(meta_display)
        
        # Footer
        container.add_item(ui.Separator())
        footer = ui.TextDisplay("-# Utilisez /memory reset pour effacer ces informations")
        container.add_item(footer)
        
        self.add_item(container)

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
        
        # Utilisateur actuel pour le prompt (thread-local via channel)
        self._current_user_profiles = {}  # {channel_id: user_profile_text}
        
        # Fonction pour générer le prompt développeur
        def get_developer_prompt():
            now = datetime.now(PARIS_TZ)
            # Récupérer le profil utilisateur si disponible (sera set avant completion)
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
            max_completion_tokens=1000,
            context_window=16384,
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
        
        if tools:
            self.gpt_api.add_tools(*tools)
            logger.info(f"Total outils enregistrés: {len(tools)}")
    
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
            import re
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
            # Récupérer les messages récents (2 minutes)
            from datetime import datetime, timedelta, timezone
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
        
        # Ingérer TOUS les messages (pour le contexte)
        await self.gpt_api.ingest_message(message.channel, message)
        
        # Incrémenter le compteur de messages pour la mémoire
        self.memory.increment_message_count(message.author.id)
        
        # Décider si on répond
        if not self.should_respond(message):
            return
        
        # Éviter les doublons
        if message.id in self._processed_messages:
            return
        
        self._processed_messages.append(message.id)
        
        # Injecter les profils utilisateurs pertinents dans le prompt
        profiles_to_inject = []
        
        # 1. Profil de l'auteur du message (toujours)
        author_profile = self.memory.get_profile_text(message.author.id)
        if author_profile:
            profiles_to_inject.append(f"**{message.author.name}** (auteur du message):\n{author_profile}")
        
        # 2. Profils des utilisateurs mentionnés
        for mentioned_user in message.mentions:
            if mentioned_user.bot:
                continue
            if mentioned_user.id == message.author.id:
                continue  # Déjà ajouté
            
            mentioned_profile = self.memory.get_profile_text(mentioned_user.id)
            if mentioned_profile:
                profiles_to_inject.append(f"**{mentioned_user.name}** (mentionné):\n{mentioned_profile}")
        
        # 3. Profils des utilisateurs récemment actifs dans la conversation (max 2 derniers)
        if not message.mentions:  # Seulement si pas de mentions explicites
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
                    profiles_to_inject.append(f"**{recent_user.name}** (récemment actif):\n{recent_profile}")
        
        # Construire le texte final
        if profiles_to_inject:
            self._get_developer_prompt._user_profile = "CARTES D'IDENTITÉ DES UTILISATEURS:\n\n" + "\n\n".join(profiles_to_inject) + "\n"
        else:
            self._get_developer_prompt._user_profile = ''
        
        # Répondre
        async with message.channel.typing():
            try:
                response = await self.gpt_api.run_completion(
                    message.channel,
                    trigger_message=message
                )
                
                # Planifier mise à jour du profil (async, non-bloquant)
                # Récupérer les messages récents de l'utilisateur
                recent_messages = []
                async for msg in message.channel.history(limit=20):
                    if msg.author.id == message.author.id and not msg.author.bot:
                        recent_messages.append(msg)
                
                if recent_messages:
                    await self.memory.check_and_schedule_update(message.author.id, recent_messages)
                
                # Formater la réponse avec les headers des tools
                text = response.text
                
                # Ajouter les headers des tools utilisés
                if response.tool_responses:
                    headers = []
                    for tool_resp in response.tool_responses:
                        header = tool_resp.metadata.get('header')
                        if header:
                            headers.append(header)
                    
                    if headers:
                        # Dédupliquer et inverser l'ordre
                        headers = list(dict.fromkeys(headers))
                        headers_text = '\n-# ' + '\n-# '.join(headers) + '\n'
                        text = headers_text + text
                
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
        
        # Configuration
        config = self.get_guild_config(interaction.guild, 'chatbot_mode', str)
        
        # Créer la vue
        tools = self.gpt_api.get_all_tools()
        view = InfoView(
            bot=self.bot,
            channel=interaction.channel,
            context_stats=context_stats,
            stats=stats,
            global_stats=global_stats,
            config=config,
            tools=tools
        )
        
        await interaction.response.send_message(view=view)
    
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
    
    # Groupe de commandes memory
    memory_group = app_commands.Group(
        name='memory',
        description="Gestion de la mémoire long terme"
    )
    
    @memory_group.command(name='show')
    async def memory_show(self, interaction: Interaction):
        """Affiche votre carte d'identité enregistrée par MARIA."""
        profile = self.memory.get_profile(interaction.user.id)
        
        if not profile or not profile.content:
            return await interaction.response.send_message(
                "Aucune information enregistrée pour le moment.",
                ephemeral=True
            )
        
        # Créer la vue
        view = MemoryProfileView(user=interaction.user, profile=profile)
        
        await interaction.response.send_message(view=view, ephemeral=True)
    
    @memory_group.command(name='reset')
    async def memory_reset(self, interaction: Interaction):
        """Efface toutes les informations que MARIA a enregistrées sur vous."""
        success = self.memory.delete_profile(interaction.user.id)
        
        if success:
            await interaction.response.send_message(
                "✅ Toutes vos informations ont été effacées.",
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "❌ Aucune information à effacer.",
                ephemeral=True
            )
 
async def setup(bot):
    await bot.add_cog(Chat(bot))

