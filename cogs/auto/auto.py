"""### Auto Cog
Fonctionnalités automatiques déclenchées par emoji (transcription audio)."""

import io
import asyncio
import logging
from typing import Literal, Union
from pathlib import Path

import discord
from discord import Interaction, app_commands
from discord.ext import commands
from openai import AsyncOpenAI

from common import dataio

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

PROPOSAL_EMOJI = '💡'
TRANSCRIPTION_MODEL = 'gpt-4o-transcribe'

# Répertoire temporaire
TEMP_DIR = Path('./temp')
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# COG -------------------------------------------------------------

class Auto(commands.Cog):
    """Cog pour les fonctionnalités automatiques."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Configuration par serveur
        guild_settings = dataio.DictTableBuilder(
            name='guild_settings',
            default_values={
                'audio_transcription': True,
                'expiration': 300  # Secondes avant expiration de la proposition
            }
        )
        self.data.map_builders(discord.Guild, guild_settings)
        
        # Client OpenAI
        self._client = AsyncOpenAI(
            api_key=self.bot.config['OPENAI_API_KEY']
        )
        
        # Propositions actives {message_id: set(types)}
        self._proposals: dict[int, set[str]] = {}
        
        logger.info("Auto cog initialisé")
    
    async def cog_unload(self):
        """Nettoyage."""
        await self._client.close()
        self.data.close_all()
    
    # CONFIGURATION -----------------------------------------------
    
    def get_guild_config(self, guild: discord.Guild, key: str):
        """Récupère la configuration d'une guilde."""
        return self.data.get(guild).get_dict_value('guild_settings', key)
    
    def set_guild_config(self, guild: discord.Guild, key: str, value: Union[str, int, bool]) -> None:
        """Met à jour la configuration d'une guilde."""
        self.data.get(guild).set_dict_value('guild_settings', key, value)
    
    # PROPOSITIONS ------------------------------------------------
    
    def add_proposal(self, message: discord.Message, proposal_type: str) -> None:
        """Ajoute une proposition."""
        if message.id not in self._proposals:
            self._proposals[message.id] = set()
        self._proposals[message.id].add(proposal_type)
    
    def remove_proposal(self, message: discord.Message, proposal_type: str) -> None:
        """Supprime une proposition."""
        if message.id in self._proposals and proposal_type in self._proposals[message.id]:
            self._proposals[message.id].remove(proposal_type)
            if not self._proposals[message.id]:
                del self._proposals[message.id]
    
    def has_proposal(self, message: discord.Message, proposal_type: str) -> bool:
        """Vérifie si une proposition existe."""
        return message.id in self._proposals and proposal_type in self._proposals[message.id]
    
    def get_proposals(self, message: discord.Message) -> set[str]:
        """Récupère toutes les propositions pour un message."""
        return self._proposals.get(message.id, set())
    
    async def _schedule_expiration(self, message: discord.Message, expiration: int) -> None:
        """Programme l'expiration des propositions."""
        await asyncio.sleep(expiration)
        if message.id in self._proposals:
            try:
                await message.clear_reaction(PROPOSAL_EMOJI)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                pass
            self._proposals.pop(message.id, None)
    
    # TRANSCRIPTION -----------------------------------------------
    
    async def extract_audio(self, message: discord.Message) -> io.BytesIO | None:
        """Extrait le fichier audio d'un message."""
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
        return None
    
    async def transcribe_audio(self, audio_file: io.BytesIO) -> str:
        """Transcrit un fichier audio."""
        try:
            audio_file.seek(0)
            transcript = await self._client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file
            )
            return transcript.text.strip()
        except Exception as e:
            logger.error(f"Erreur transcription: {e}")
            raise
        finally:
            audio_file.close()
    
    # EVENTS ------------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Détecte les messages et ajoute les propositions."""
        if message.author.bot or not message.guild:
            return
        
        if isinstance(message.channel, discord.DMChannel):
            return
        
        proposals_added = []
        expiration = int(self.get_guild_config(message.guild, 'expiration'))
        
        # Transcription audio
        if bool(self.get_guild_config(message.guild, 'audio_transcription')):
            if any(a.content_type and a.content_type.startswith('audio') for a in message.attachments):
                self.add_proposal(message, 'transcription')
                proposals_added.append('transcription')
        
        # Ajouter l'emoji si des propositions existent
        if proposals_added:
            try:
                await message.add_reaction(PROPOSAL_EMOJI)
                # Programmer l'expiration
                if expiration > 0:
                    asyncio.create_task(self._schedule_expiration(message, expiration))
            except (discord.Forbidden, discord.HTTPException) as e:
                logger.warning(f"Impossible d'ajouter la réaction: {e}")
                # Supprimer les propositions si on ne peut pas ajouter l'emoji
                for prop in proposals_added:
                    self.remove_proposal(message, prop)
    
    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Traite les réactions pour exécuter les propositions."""
        if user.bot:
            return
        
        if str(reaction.emoji) != PROPOSAL_EMOJI:
            return
        
        message = reaction.message
        proposals = self.get_proposals(message)
        
        if not proposals:
            return
        
        # Actuellement, un seul type de proposition (transcription)
        # Si plusieurs types à l'avenir, implémenter le menu de sélection
        proposal_type = list(proposals)[0]
        await self._execute_proposal(message, proposal_type, user)
    
    async def _execute_proposal(self, message: discord.Message, proposal_type: str, user: discord.User):
        """Exécute une proposition."""
        try:
            async with message.channel.typing():
                if proposal_type == 'transcription':
                    audio_file = await self.extract_audio(message)
                    if not audio_file:
                        await message.reply("❌ Aucun fichier audio trouvé.", mention_author=False, delete_after=10)
                        return
                    
                    transcript = await self.transcribe_audio(audio_file)
                    content = f">>> {transcript}\n-# Transcription demandée par {user.mention}"
                    await message.reply(content, mention_author=False, allowed_mentions=discord.AllowedMentions.none())
        
        except Exception as e:
            logger.error(f"Erreur exécution proposition {proposal_type}: {e}")
            await message.reply(f"❌ Erreur: {str(e)}", mention_author=False, delete_after=10)
        
        finally:
            # Nettoyer
            try:
                await message.clear_reaction(PROPOSAL_EMOJI)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                pass
            self._proposals.pop(message.id, None)
    
    # COMMANDES ---------------------------------------------------
    
    auto_group = app_commands.Group(
        name='auto',
        description="Paramètres des fonctionnalités automatiques",
        default_permissions=discord.Permissions(manage_messages=True)
    )
    
    @auto_group.command(name='transcription')
    async def auto_transcription(self, interaction: Interaction, enabled: bool):
        """Active ou désactive la transcription audio automatique.
        
        :param enabled: True pour activer, False pour désactiver
        """
        if not interaction.guild:
            return await interaction.response.send_message("Commande serveur uniquement.", ephemeral=True)
        
        self.set_guild_config(interaction.guild, 'audio_transcription', enabled)
        status = "activée" if enabled else "désactivée"
        await interaction.response.send_message(f"✅ Transcription audio {status}.", ephemeral=True)
    


async def setup(bot):
    await bot.add_cog(Auto(bot))

