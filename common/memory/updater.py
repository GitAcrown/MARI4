"""### Memory > Updater
Mini IA pour mettre à jour les cartes d'identité."""

import logging
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
import discord

logger = logging.getLogger('MARI4.memory.updater')

# Modèle économique pour les mises à jour
UPDATE_MODEL = 'gpt-4.1-nano'
MAX_TOKENS = 300  # Limite pour la carte d'identité

# Schéma Pydantic pour le profil
class UserProfileSchema(BaseModel):
    """Schéma structuré pour les profils utilisateurs."""
    identite: str  # Nom/pseudo, âge, localisation
    activite: str  # Études/travail, projets en cours
    tech: str  # Langages, outils, frameworks, compétences techniques
    preferences: str  # Goûts, aversions, habitudes
    contexte: str  # Autres informations pertinentes
    no_change: bool = False  # True si aucune nouvelle info

# Prompt strict pour éviter les hallucinations
PROFILE_UPDATE_PROMPT = """Tu dois mettre à jour une carte d'identité utilisateur pour une IA.

RÈGLES STRICTES:
1. N'écris QUE des faits explicitement mentionnés par l'utilisateur dans les messages récents
2. JAMAIS d'inférence ou de supposition
3. Si aucune nouvelle info pertinente → met "no_change" à true et garde les valeurs actuelles
4. Chaque champ doit être concis mais informatif
5. Si un champ n'a pas d'info, laisse-le vide ""

Carte actuelle:
{current_profile}

Derniers messages de l'utilisateur:
{messages}

Mets à jour les champs avec les nouvelles informations trouvées, ou indique no_change=true si rien de nouveau."""

class ProfileUpdater:
    """Mini IA pour mettre à jour les profils utilisateur."""
    
    def __init__(self, api_key: str):
        """Initialise l'updater.
        
        Args:
            api_key: Clé API OpenAI
        """
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info("ProfileUpdater initialisé")
    
    async def update_profile(
        self, 
        current_profile: Optional[str], 
        messages: list[discord.Message]
    ) -> Optional[str]:
        """Met à jour une carte d'identité.
        
        Args:
            current_profile: Carte actuelle (None si première fois)
            messages: Derniers messages de l'utilisateur
            
        Returns:
            Nouvelle carte ou None si aucun changement
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
            
            # Si aucun changement
            if parsed.no_change:
                logger.debug("Aucun changement détecté")
                return None
            
            # Construire le profil formaté
            new_profile = self._format_profile(parsed)
            
            # Validation basique
            if len(new_profile) < 10:
                logger.warning("Profil trop court")
                return None
            
            logger.info(f"Profil mis à jour: {len(new_profile)} caractères")
            return new_profile
            
        except Exception as e:
            logger.error(f"Erreur mise à jour profil: {e}")
            return None
    
    def _format_profile(self, schema: UserProfileSchema) -> str:
        """Formate le schéma en texte structuré.
        
        Args:
            schema: Schéma Pydantic parsé
            
        Returns:
            Texte formaté pour l'IA
        """
        parts = []
        if schema.identite:
            parts.append(f"IDENTITÉ: {schema.identite}")
        if schema.activite:
            parts.append(f"ACTIVITÉ: {schema.activite}")
        if schema.tech:
            parts.append(f"TECH: {schema.tech}")
        if schema.preferences:
            parts.append(f"PRÉFÉRENCES: {schema.preferences}")
        if schema.contexte:
            parts.append(f"CONTEXTE: {schema.contexte}")
        
        return " | ".join(parts)
    
    def _format_messages(self, messages: list[discord.Message]) -> str:
        """Formate les messages pour le prompt."""
        formatted = []
        for msg in messages[-15:]:  # Max 15 derniers messages
            formatted.append(f"- {msg.content}")
        return '\n'.join(formatted)
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()

