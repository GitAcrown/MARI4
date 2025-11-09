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
- Identite STABLE: prenom, metier, localisation generale (ville/pays)
- Préférences de communication: ton souhaite, niveau de detail, sujets à éviter
- Competences professionnelles ou techniques importantes
- Contraintes durables: limites, besoins spécifiques, accessibilité

QUE NE PAS RETENIR (exemples):
- Age exact (change chaque annee, inutile)
- Goûts personnels temporaires ("aime les blondes", "intéresse par X produit")
- Projets ponctuels ou achats envisages
- Descriptions physiques ou preferences esthetiques
- Actions du moment ("cherche actuellement", "veut acheter")
- Informations sur d'autres personnes
- Détails anecdotiques sans impact sur les futures interactions

INSTRUCTIONS:
1. Lis le profil actuel et les nouveaux messages
2. Détermine s'il y a de nouvelles infos pertinentes à ajouter
3. Si OUI: écris le profil mis à jour dans le champ "content" et mets "no_change" à false
4. Si NON: recopie exactement le profil actuel dans "content" et mets "no_change" à true

REGLES STRICTES - CE QUI EST INTERDIT:
- AUCUNE supposition, inférence ou déduction
- AUCUNE interprétation de ce que l'utilisateur "pourrait" vouloir dire
- AUCUNE généralisation à partir d'un exemple unique
- N'écris QUE ce qui est EXPLICITEMENT dit par l'utilisateur lui-même
- Si une info n'est pas claire ou certaine, NE L'ECRIS PAS

EXEMPLES DE CE QU'IL NE FAUT PAS FAIRE:
- Message: "je code en Python" -> N'écris PAS "développeur" (pas dit explicitement)
- Message: "j'aime ce jeu" -> N'écris PAS "gamer" (trop général)
- Message: "je suis fatigue" -> N'écris PAS "problèmes de sommeil" (supposition)
- Message: "j'aime les blondes" -> N'écris PAS (préférence personnelle sans impact sur l'IA)

IMPORTANT:
- Le champ "content" doit contenir UNIQUEMENT le texte du profil
- N'écris JAMAIS "no_change" ou d'autres métadonnées dans le texte du profil
- "no_change" est un champ séparé du schéma JSON

FORMAT DU PROFIL:
- Style télégraphique, phrases courtes et factuelles
- Longueur: 400-700 caractères (environ 6-8 phrases)
- Sois EXTREMEMENT conservateur: en cas de doute, n'ajoute rien"""

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
        for msg in reversed(messages[-15:]):  # Max 15 derniers messages, du plus ancien au plus récent
            # Extraire juste le texte, sans les métadonnées Discord
            content = msg.content.strip()
            if content:
                formatted.append(f"- {content}")
        return '\n'.join(formatted)
    
    async def close(self):
        """Ferme le client."""
        await self.client.close()

