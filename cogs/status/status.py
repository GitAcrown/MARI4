"""### Status Cog
Gestion élégante du statut Discord du bot."""

import logging
import discord
from discord.ext import commands, tasks
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

STATUS_UPDATE_INTERVAL = 120  # minutes
STATUS_MODEL = 'gpt-4.1-nano'

STATUS_SYSTEM_PROMPT = """Tu dois créer un très court (3-4 mots MAXIMUM) TEXTE DE STATUT DISCORD pour un chatbot IA (appelée MARIA) qui se genre au féminin.

CONSIGNES:
- Le statut doit être grammaticalement correct et avoir du sens, même s'il est court
- Privilégier des phrases courtes mais complètes ou des expressions idiomatiques connues
- Utilise l'humour, l'autodérision et le sarcasme avec des formulations correctes
- Refs à la pop culture encouragés (memes internet récents et francophones, jeux vidéo, séries, films, anime etc.)
- Le statut est réalisé du point de vue de l'IA, mais éviter les phrases descriptives comme "je suis ..."
- Pas de jargon technique, langage d'IA ou de termes clichés liés à la technologie
- Préférer le français, anglais seulement si références culturelles pertinentes
- Pas d'emoji, pas de point en fin de phrase
- Exemples d'inspiration (ne pas copier) : "Bug en cours", "Café requis", "Mode sieste", "Erreur 404", "Meilleure que HAL", "R2D2 sur nous"

La réponse doit être un JSON avec la clé "status" contenant le texte du statut, sans autres informations."""

class StatusResponse(BaseModel):
    """Réponse du modèle pour le statut."""
    status: str

class Status(commands.Cog):
    """Cog gérant le statut Discord du bot."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        
        # Client OpenAI
        self._client = AsyncOpenAI(
            api_key=self.bot.config['OPENAI_API_KEY']
        )
        
        logger.info("Status cog initialisé")
    
    async def cog_load(self):
        """Démarrer la tâche de mise à jour."""
        self.update_status.start()
    
    async def cog_unload(self):
        """Arrêter la tâche de mise à jour."""
        self.update_status.stop()
        await self._client.close()
    
    async def generate_status(self) -> str:
        """Génère un nouveau statut avec l'IA."""
        try:
            response = await self._client.beta.chat.completions.parse(
                model=STATUS_MODEL,
                messages=[{"role": "developer", "content": STATUS_SYSTEM_PROMPT}],
                temperature=1.1,
                max_completion_tokens=50,
                response_format=StatusResponse
            )
            
            if not response.choices[0].message.parsed:
                raise Exception("Pas de statut généré")
            
            return response.choices[0].message.parsed.status
            
        except Exception as e:
            logger.error(f"Erreur génération statut: {e}")
            # Fallback
            return "En pause"
    
    @tasks.loop(minutes=STATUS_UPDATE_INTERVAL)
    async def update_status(self):
        """Mise à jour périodique du statut."""
        try:
            new_status = await self.generate_status()
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.custom,
                    name='custom',
                    state=new_status
                )
            )
            logger.info(f"Statut mis à jour: {new_status}")
        except Exception as e:
            logger.error(f"Erreur mise à jour statut: {e}")
    
    @update_status.before_loop
    async def before_update_status(self):
        """Attendre que le bot soit prêt."""
        await self.bot.wait_until_ready()
    
    # Commande manuelle
    @commands.command(name='refreshstatus', hidden=True)
    @commands.is_owner()
    async def force_refresh_status(self, ctx: commands.Context):
        """Force la mise à jour du statut."""
        try:
            new_status = await self.generate_status()
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.custom,
                    name='custom',
                    state=new_status
                )
            )
            await ctx.send(f"✅ Statut mis à jour: `{new_status}`")
        except Exception as e:
            logger.error(f"Erreur refresh statut: {e}")
            await ctx.send(f"❌ Erreur: {e}")

async def setup(bot):
    await bot.add_cog(Status(bot))

