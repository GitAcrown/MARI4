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
MAX_TOKENS = 500  # Limite pour le profil

# Schéma Pydantic pour le profil
class UserProfileSchema(BaseModel):
    """Schéma structuré pour les profils utilisateurs."""
    identite: str = ""  # Nom/pseudo, âge, localisation
    activite: str = ""  # Études/travail, projets en cours
    tech: str = ""  # Langages, outils, frameworks, compétences techniques
    preferences: str = ""  # Goûts, aversions, habitudes
    contexte: str = ""  # Autres informations pertinentes
    no_change: bool = False  # True si aucune nouvelle info

# Prompt strict pour éviter les hallucinations
PROFILE_UPDATE_PROMPT = """Tu dois mettre à jour un profil utilisateur pour une IA.

PROFIL ACTUEL:
{current_profile}

NOUVEAUX MESSAGES:
{messages}

INSTRUCTIONS:
1. Analyse les nouveaux messages pour trouver des infos personnelles (nom, âge, métier, compétences, préférences, etc.)
2. Si tu trouves des nouvelles infos OU des modifications aux infos existantes:
   - Mets à jour les champs concernés en FUSIONNANT avec le profil actuel
   - Met "no_change" à FALSE
   - Garde les infos existantes qui ne changent pas
3. Si AUCUNE nouvelle info pertinente dans les messages:
   - Met "no_change" à TRUE
   - Remplis quand même tous les champs avec le contenu actuel (copie-le)

RÈGLES:
- N'écris QUE des faits explicitement mentionnés
- JAMAIS d'inférence ou de supposition
- Sois concis mais complet
- Si un champ n'a jamais eu d'info, laisse-le vide ""

Exemple: Si l'utilisateur dit "j'ai 25 ans" et le profil actuel dit "Nom: Jean", tu dois mettre:
identite: "Nom: Jean, 25 ans"
no_change: false"""

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
        
        logger.debug(f"Mise à jour profil - {len(messages)} messages, force={force}")
        logger.debug(f"Messages: {messages_text[:200]}...")
        
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
            
            logger.debug(f"Réponse mini-IA: no_change={parsed.no_change}, identite={parsed.identite[:50] if parsed.identite else 'vide'}...")
            
            # Si aucun changement (sauf si force=True)
            if parsed.no_change and not force:
                logger.debug("Aucun changement détecté par la mini-IA")
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
            Texte formaté lisible
        """
        parts = []
        if schema.identite:
            parts.append(f"**Identité:**\n{schema.identite}")
        if schema.activite:
            parts.append(f"**Activité:**\n{schema.activite}")
        if schema.tech:
            parts.append(f"**Tech:**\n{schema.tech}")
        if schema.preferences:
            parts.append(f"**Préférences:**\n{schema.preferences}")
        if schema.contexte:
            parts.append(f"**Contexte:**\n{schema.contexte}")
        
        return "\n\n".join(parts)
    
    def _format_messages(self, messages: list[discord.Message]) -> str:
        """Formate les messages pour le prompt."""
        formatted = []
        for msg in reversed(messages[-15:]):  # Max 15 derniers messages, du plus ancien au plus récent
            # Extraire juste le texte, sans les métadonnées Discord
            content = msg.content.strip()
            if content:
                formatted.append(f"- {content}")
        return '\n'.join(formatted)
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()

