"""### Memory > Manager
Gestionnaire principal du système de mémoire."""

import logging
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict
from pathlib import Path
import discord

from .profile import UserProfile
from .updater import ProfileUpdater

logger = logging.getLogger('MARI4.memory.manager')

class MemoryManager:
    """Gestionnaire de mémoire long terme pour les utilisateurs."""
    
    def __init__(self, api_key: str, db_path: str = 'data/memory.db'):
        """Initialise le gestionnaire de mémoire.
        
        Args:
            api_key: Clé API OpenAI
            db_path: Chemin vers la base de données SQLite
        """
        self.updater = ProfileUpdater(api_key)
        
        # Créer le dossier data si nécessaire
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connexion SQLite
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Créer la table
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                messages_since_update INTEGER DEFAULT 0
            )'''
        )
        self.conn.commit()
        
        # Cache en mémoire
        self._profiles: Dict[int, UserProfile] = {}
        
        # Queue pour les mises à jour en attente
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._update_task: Optional[asyncio.Task] = None
        
        # Verrous pour éviter les race conditions lors des MAJ
        self._update_locks: Dict[int, asyncio.Lock] = {}
        
        logger.info("MemoryManager initialisé")
    
    def start_background_updater(self):
        """Démarre le worker de mise à jour en arrière-plan."""
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._background_updater())
            logger.info("Background updater démarré")
    
    async def _background_updater(self):
        """Worker qui traite les mises à jour en arrière-plan."""
        while True:
            try:
                user_id, messages = await self._update_queue.get()
                
                # Obtenir ou créer le verrou pour cet utilisateur
                if user_id not in self._update_locks:
                    self._update_locks[user_id] = asyncio.Lock()
                
                async with self._update_locks[user_id]:
                    profile = self.get_profile(user_id)
                    current_content = profile.content if profile else None
                    
                    new_content = await self.updater.update_profile(current_content, messages)
                    
                    if new_content:
                        if profile:
                            profile.content = new_content
                            profile.reset_counter()
                        else:
                            profile = UserProfile(
                                user_id=user_id,
                                content=new_content,
                                created_at=datetime.now(timezone.utc),
                                updated_at=datetime.now(timezone.utc),
                                messages_since_update=0
                            )
                        
                        self._save_profile(profile)
                        logger.info(f"Profil mis à jour pour user {user_id}")
                
                self._update_queue.task_done()
                
            except Exception as e:
                logger.error(f"Erreur background updater: {e}")
                await asyncio.sleep(5)
    
    def get_profile(self, user_id: int) -> Optional[UserProfile]:
        """Récupère le profil d'un utilisateur (cache + DB).
        
        Args:
            user_id: ID de l'utilisateur Discord
            
        Returns:
            UserProfile ou None
        """
        if user_id in self._profiles:
            return self._profiles[user_id]
        
        cursor = self.conn.execute(
            'SELECT * FROM user_profiles WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            profile = UserProfile(
                user_id=row['user_id'],
                content=row['content'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                messages_since_update=row['messages_since_update']
            )
            self._profiles[user_id] = profile
            return profile
        
        return None
    
    def get_profile_text(self, user_id: int) -> Optional[str]:
        """Récupère le texte du profil d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur Discord
            
        Returns:
            Texte du profil ou None
        """
        profile = self.get_profile(user_id)
        return profile.content if profile else None
    
    def increment_message_count(self, user_id: int):
        """Incrémente le compteur de messages pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur Discord
        """
        profile = self.get_profile(user_id)
        if profile:
            profile.increment_messages()
            self._save_profile(profile)
    
    async def check_and_schedule_update(self, user_id: int, recent_messages: list[discord.Message]):
        """Vérifie si une MAJ automatique est nécessaire et la planifie.
        
        Args:
            user_id: ID de l'utilisateur Discord
            recent_messages: Messages récents de l'utilisateur
        """
        profile = self.get_profile(user_id)
        
        if profile is None or profile.should_update():
            await self._update_queue.put((user_id, recent_messages))
            logger.debug(f"MAJ automatique planifiée pour user {user_id}")
    
    async def force_update(self, user_id: int, recent_messages: list[discord.Message]) -> bool:
        """Force une MAJ immédiate et synchrone du profil (appelé par l'IA).
        
        Args:
            user_id: ID de l'utilisateur Discord
            recent_messages: Messages récents de l'utilisateur
            
        Returns:
            True si succès, False sinon
        """
        if not recent_messages:
            logger.warning(f"Force update sans messages pour user {user_id}")
            return False
        
        try:
            # Obtenir ou créer le verrou pour cet utilisateur
            if user_id not in self._update_locks:
                self._update_locks[user_id] = asyncio.Lock()
            
            async with self._update_locks[user_id]:
                profile = self.get_profile(user_id)
                current_content = profile.content if profile else None
                
                new_content = await self.updater.update_profile(current_content, recent_messages, force=True)
                
                if new_content:
                    if profile:
                        profile.content = new_content
                        profile.reset_counter()
                    else:
                        profile = UserProfile(
                            user_id=user_id,
                            content=new_content,
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                            messages_since_update=0
                        )
                    
                    self._save_profile(profile)
                    logger.info(f"Profil MAJ immédiate pour user {user_id} (IA)")
                    return True
                else:
                    logger.debug(f"Aucune nouvelle info pour user {user_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Erreur force update: {e}")
            return False
    
    def delete_profile(self, user_id: int) -> bool:
        """Supprime le profil d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur Discord
            
        Returns:
            True si supprimé, False sinon
        """
        try:
            self.conn.execute(
                'DELETE FROM user_profiles WHERE user_id = ?',
                (user_id,)
            )
            self.conn.commit()
            self._profiles.pop(user_id, None)
            logger.info(f"Profil supprimé pour user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur suppression profil: {e}")
            return False
    
    def _save_profile(self, profile: UserProfile):
        """Sauvegarde un profil en DB et cache.
        
        Args:
            profile: Profil à sauvegarder
        """
        try:
            self.conn.execute(
                '''INSERT OR REPLACE INTO user_profiles 
                   (user_id, content, created_at, updated_at, messages_since_update)
                   VALUES (?, ?, ?, ?, ?)''',
                (
                    profile.user_id,
                    profile.content,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                    profile.messages_since_update
                )
            )
            self.conn.commit()
            self._profiles[profile.user_id] = profile
        except Exception as e:
            logger.error(f"Erreur sauvegarde profil: {e}")
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du système de mémoire.
        
        Returns:
            Dictionnaire de stats
        """
        cursor = self.conn.execute('SELECT COUNT(*) as count FROM user_profiles')
        total_profiles = cursor.fetchone()['count']
        
        return {
            'total_profiles': total_profiles,
            'cached_profiles': len(self._profiles),
            'pending_updates': self._update_queue.qsize()
        }
    
    async def close(self):
        """Ferme le gestionnaire."""
        if self._update_task:
            self._update_task.cancel()
        await self.updater.close()
        self.conn.close()
        logger.info("MemoryManager fermé")

