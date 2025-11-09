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

# Prompt pour la mini-IA
PROFILE_UPDATE_PROMPT = """Tu dois mettre a jour un profil utilisateur pour une IA.

PROFIL ACTUEL:
{current_profile}

NOUVEAUX MESSAGES:
{messages}

OBJECTIF:
Garder uniquement les informations UTILES pour personnaliser les interactions futures avec cette personne.

QUE RETENIR (exemples):
- Identite: prenom, age, metier, localisation
- Preferences de communication: ton souhaite, niveau de detail, sujets a eviter
- Contexte personnel: projets en cours, competences, centres d'interet recurrents
- Contraintes: limites, besoins specifiques, accessibilite

QUE NE PAS RETENIR (exemples):
- Opinions temporaires ou contextuelles ("j'aime pas ce film")
- Actions ponctuelles ("j'ai mange une pizza")
- Questions posees (sauf si elles revelent un besoin recurrent)
- Informations sur d'autres personnes mentionnees
- Faits generaux non lies a l'utilisateur

INSTRUCTIONS:
1. Lis le profil actuel et les nouveaux messages
2. Determine s'il y a de nouvelles infos pertinentes a ajouter
3. Si OUI: ecris le profil mis a jour dans le champ "content" et mets "no_change" a false
4. Si NON: recopie exactement le profil actuel dans "content" et mets "no_change" a true

REGLES STRICTES - CE QUI EST INTERDIT:
- AUCUNE supposition, inference ou deduction
- AUCUNE interpretation de ce que l'utilisateur "pourrait" vouloir dire
- AUCUNE generalisation a partir d'un exemple unique
- N'ecris QUE ce qui est EXPLICITEMENT dit par l'utilisateur lui-meme
- Si une info n'est pas claire ou certaine, NE L'ECRIS PAS

EXEMPLES DE CE QU'IL NE FAUT PAS FAIRE:
- Message: "je code en Python" -> N'ecris PAS "developpeur" (pas dit explicitement)
- Message: "j'aime ce jeu" -> N'ecris PAS "gamer" (trop general)
- Message: "je suis fatigue" -> N'ecris PAS "problemes de sommeil" (supposition)

IMPORTANT:
- Le champ "content" doit contenir UNIQUEMENT le texte du profil
- N'ecris JAMAIS "no_change" ou d'autres metadonnees dans le texte du profil
- "no_change" est un champ separe du schema JSON

FORMAT DU PROFIL:
- Style telegraphique, phrases courtes et factuelles
- Longueur: 400-700 caracteres (environ 6-8 phrases)
- Sois EXTREMEMENT conservateur: en cas de doute, n'ajoute rien"""

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
            
            logger.debug(f"Réponse mini-IA: no_change={parsed.no_change}, content={parsed.content[:50] if parsed.content else 'vide'}...")
            
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
        for msg in reversed(messages[-15:]):  # Max 15 derniers messages, du plus ancien au plus récent
            # Extraire juste le texte, sans les métadonnées Discord
            content = msg.content.strip()
            if content:
                formatted.append(f"- {content}")
        return '\n'.join(formatted)
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()

