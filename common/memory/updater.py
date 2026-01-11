"""### Memory > Updater
Mini IA pour mettre à jour les profils utilisateur."""

import logging
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
import discord

logger = logging.getLogger('MARI4.memory.updater')

# Modèle économique pour les mises à jour
UPDATE_MODEL = 'gpt-4.1-nano'
MAX_TOKENS = 700  # Limite pour le profil

# Schéma Pydantic pour le profil
class UserProfileSchema(BaseModel):
    """Schéma pour les profils utilisateurs."""
    content: str = ""  # Profil complet en texte libre
    no_change: bool = False  # True si aucune nouvelle info

# Prompt pour la mini-IA (optimisé pour plus de mises à jour)
PROFILE_UPDATE_PROMPT = """Mets à jour le profil utilisateur à partir des nouveaux messages.

PROFIL ACTUEL:
{current_profile}

NOUVEAUX MESSAGES:
{messages}

INSTRUCTIONS:

RÈGLE #0 - IMPORTANT:
En cas de DOUTE sur la pertinence d'une info, GARDE-LA plutôt que de l'ignorer.
Il vaut mieux un profil légèrement verbeux qu'un profil incomplet.

1. ANALYSE les nouveaux messages pour extraire UNIQUEMENT les infos durables:
   - Identité: prénom, âge, métier, ville, nationalité, sexe
   - Préférences de communication: ton souhaité (tutoiement/vouvoiement), niveau de détail
   - Compétences, passions, centres d'intérêt récurrents
   - Contraintes durables (santé, disponibilité, etc.)

2. IGNORE tout le reste:
   - Opinions temporaires, actions ponctuelles ("J'ai fait X hier", "Je vais faire Y")
   - Questions posées par l'utilisateur
   - Infos sur d'autres personnes
   - Projets ponctuels ou de courte durée

3. MISE À JOUR:
   - Si nouvelle info durable : AJOUTE au profil
   - Si info contradictoire avec profil actuel : REMPLACE par la plus récente
   - Si aucune nouvelle info durable : mets no_change à true

4. NETTOYAGE (à chaque fois):
   - Supprime formulations meta ("X non renseigné", "Absence de", "X non défini", etc.)
   - Fusionne doublons et répétitions
   - Reste concis mais complet

FORMAT:
Le champ "content" contient UNIQUEMENT le profil en texte brut. PAS de préfixes, PAS de structure avec tirets.
Exemple: "Théo, 24 ans, développeur web à Lyon. Préfère le tutoiement et les explications détaillées. Passionné de jeux vidéo et de musique électronique."
"""

class ProfileUpdater:
    """Mini IA pour mettre à jour les profils utilisateur."""
    
    def __init__(self, api_key: str):
        """Initialise l'updater.
        
        Args:
            api_key: Clé API OpenAI
        """
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def update_profile(
        self, 
        current_profile: Optional[str], 
        messages: list[discord.Message],
        force: bool = False
    ) -> Optional[str]:
        """Met à jour un profil utilisateur.
        
        Args:
            current_profile: Profil actuel (None si première fois)
            messages: Derniers messages de l'utilisateur
            force: Si True, ignore le flag no_change
            
        Returns:
            Nouveau profil ou None si aucun changement
        """
        if not messages:
            return None
        
        # Préparer le contexte
        current = current_profile or "Aucune information pour l'instant."
        messages_text = self._format_messages(messages)
        
        # Appel à la mini IA avec structured output
        try:
            response = await self.client.beta.chat.completions.parse(
                model=UPDATE_MODEL,
                messages=[
                    {
                        "role": "developer",
                        "content": PROFILE_UPDATE_PROMPT.format(
                            current_profile=current,
                            messages=messages_text
                        )
                    }
                ],
                temperature=0.1,  # Très bas pour éviter la créativité
                max_completion_tokens=MAX_TOKENS,
                response_format=UserProfileSchema
            )
            
            if not response.choices[0].message.parsed:
                logger.warning("Pas de profil parsé")
                return None
            
            parsed = response.choices[0].message.parsed
            
            # Si aucun changement (sauf si force=True)
            if parsed.no_change and not force:
                return None
            
            # Construire le profil formaté
            new_profile = self._format_profile(parsed)
            
            # Validation basique
            if len(new_profile) < 10:
                logger.warning("Profil trop court")
                return None
            
            # Logger les changements
            if current_profile and current_profile != "Aucune information pour l'instant.":
                # Calculer ce qui a été ajouté (approximatif)
                old_length = len(current_profile)
                new_length = len(new_profile)
                diff = new_length - old_length
                logger.info(f"Profil mis a jour: {new_length} caracteres ({diff:+d} caracteres)")
                logger.debug(f"Nouveau contenu: {new_profile[:200]}...")
            else:
                logger.info(f"Profil cree: {len(new_profile)} caracteres")
                logger.debug(f"Contenu initial: {new_profile[:200]}...")
            
            return new_profile
            
        except Exception as e:
            logger.error(f"Erreur mise à jour profil: {e}")
            return None
    
    def _format_profile(self, schema: UserProfileSchema) -> str:
        """Retourne le contenu du profil.
        
        Args:
            schema: Schéma Pydantic parsé
            
        Returns:
            Texte du profil
        """
        return schema.content.strip()
    
    def _format_messages(self, messages: list[discord.Message]) -> str:
        """Formate les messages pour le prompt."""
        formatted = []
        for msg in reversed(messages[-25:]):  # Max 25 derniers messages (plus de contexte), du plus ancien au plus récent
            # Extraire juste le texte, sans les métadonnées Discord
            content = msg.content.strip()
            if content:
                formatted.append(f"- {content}")
        return '\n'.join(formatted)
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()

