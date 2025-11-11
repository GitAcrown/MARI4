"""### Chat > Scheduler
Système de tâches planifiées pour l'exécution autonome."""

import asyncio
import logging
import sqlite3
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Callable

logger = logging.getLogger(f'MARI4.chat.scheduler')

# MODELS ----------------------------------------------------------

@dataclass
class ScheduledTask:
    """Tâche planifiée par l'IA."""
    id: int
    channel_id: int
    user_id: int
    task_description: str
    execute_at: datetime
    created_at: datetime
    status: str = 'pending'  # pending, completed, failed, cancelled
    message_id: int = 0  # ID du message d'origine pour reply

# DATABASE --------------------------------------------------------

class TaskDatabase:
    """Gestionnaire de base de données pour les tâches planifiées."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialise la base de données."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        
        # Activer le mode WAL pour permettre les lectures concurrentes
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                task_description TEXT NOT NULL,
                execute_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                message_id INTEGER DEFAULT 0
            )
        ''')
        
        # Migration : Ajouter la colonne message_id si elle n'existe pas
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(scheduled_tasks)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'message_id' not in columns:
                logger.info("Migration: Ajout de la colonne message_id")
                conn.execute('ALTER TABLE scheduled_tasks ADD COLUMN message_id INTEGER DEFAULT 0')
                conn.commit()
        except Exception as e:
            logger.warning(f"Erreur lors de la migration: {e}")
        
        # Créer des index pour optimiser les requêtes
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_status_execute 
            ON scheduled_tasks(status, execute_at)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_status 
            ON scheduled_tasks(user_id, status)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Base de données scheduler initialisée: {self.db_path}")
    
    def add_task(self, channel_id: int, user_id: int, task_description: str, execute_at: datetime, message_id: int = 0) -> int:
        """Ajoute une nouvelle tâche planifiée."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scheduled_tasks (channel_id, user_id, task_description, execute_at, created_at, status, message_id)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
        ''', (
            channel_id,
            user_id,
            task_description,
            execute_at.isoformat(),
            datetime.now(timezone.utc).isoformat(),
            message_id
        ))
        
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Tâche planifiée #{task_id}: '{task_description}' pour user {user_id} à {execute_at}")
        return task_id
    
    def get_pending_tasks(self) -> list[ScheduledTask]:
        """Récupère toutes les tâches en attente dont l'heure est arrivée."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute('''
            SELECT * FROM scheduled_tasks
            WHERE status = 'pending' AND execute_at <= ?
            ORDER BY execute_at ASC
        ''', (now,))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append(ScheduledTask(
                id=row['id'],
                channel_id=row['channel_id'],
                user_id=row['user_id'],
                task_description=row['task_description'],
                execute_at=datetime.fromisoformat(row['execute_at']),
                created_at=datetime.fromisoformat(row['created_at']),
                status=row['status'],
                message_id=row['message_id'] if 'message_id' in row.keys() else 0
            ))
        
        conn.close()
        return tasks
    
    def update_task_status(self, task_id: int, status: str):
        """Met à jour le statut d'une tâche."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute('UPDATE scheduled_tasks SET status = ? WHERE id = ?', (status, task_id))
        conn.commit()
        conn.close()
    
    def get_all_tasks(self, limit: int = 50) -> list[ScheduledTask]:
        """Récupère toutes les tâches (pour affichage admin)."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM scheduled_tasks
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append(ScheduledTask(
                id=row['id'],
                channel_id=row['channel_id'],
                user_id=row['user_id'],
                task_description=row['task_description'],
                execute_at=datetime.fromisoformat(row['execute_at']),
                created_at=datetime.fromisoformat(row['created_at']),
                status=row['status'],
                message_id=row['message_id'] if 'message_id' in row.keys() else 0
            ))
        
        conn.close()
        return tasks
    
    def cancel_task(self, task_id: int, user_id: Optional[int] = None) -> bool:
        """Annule une tâche.
        
        Args:
            task_id: ID de la tâche à annuler
            user_id: Si fourni, vérifie que la tâche appartient à cet utilisateur
        
        Returns:
            True si la tâche a été annulée
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Vérifier que la tâche appartient à l'utilisateur si spécifié
        if user_id is not None:
            cursor.execute('UPDATE scheduled_tasks SET status = ? WHERE id = ? AND user_id = ? AND status = ?', 
                          ('cancelled', task_id, user_id, 'pending'))
        else:
            # Admin : pas de vérification user_id
            cursor.execute('UPDATE scheduled_tasks SET status = ? WHERE id = ? AND status = ?', 
                          ('cancelled', task_id, 'pending'))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0
    
    def get_user_tasks(self, user_id: int, limit: int = 20) -> list['ScheduledTask']:
        """Récupère les tâches en attente d'un utilisateur spécifique.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre max de tâches à retourner
        
        Returns:
            Liste des tâches en attente de l'utilisateur
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM scheduled_tasks
            WHERE user_id = ? AND status = 'pending'
            ORDER BY execute_at ASC
            LIMIT ?
        ''', (user_id, limit))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append(ScheduledTask(
                id=row['id'],
                channel_id=row['channel_id'],
                user_id=row['user_id'],
                task_description=row['task_description'],
                execute_at=datetime.fromisoformat(row['execute_at']),
                created_at=datetime.fromisoformat(row['created_at']),
                status=row['status'],
                message_id=row['message_id'] if 'message_id' in row.keys() else 0
            ))
        
        conn.close()
        return tasks
    
    def count_pending_user_tasks(self, user_id: int) -> int:
        """Compte le nombre de tâches en attente pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
        
        Returns:
            Nombre de tâches en attente
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as count FROM scheduled_tasks
            WHERE user_id = ? AND status = 'pending'
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def cleanup_old_tasks(self, days: int = 1):
        """Supprime les tâches terminées/annulées de plus de X jours."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = (cutoff - timedelta(days=days)).isoformat()
        
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM scheduled_tasks
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND created_at < ?
        ''', (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Nettoyage: {deleted} tâche(s) supprimée(s)")

# SCHEDULER -------------------------------------------------------

class TaskScheduler:
    """Orchestrateur de tâches planifiées avec worker en arrière-plan."""
    
    def __init__(self, db_path: str, executor: Callable):
        """
        Args:
            db_path: Chemin vers la base de données SQLite
            executor: Fonction async pour exécuter les tâches (channel_id, user_id, task_description)
        """
        self.db = TaskDatabase(db_path)
        self.executor = executor
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("TaskScheduler initialisé")
    
    def schedule_task(self, channel_id: int, user_id: int, task_description: str, execute_at: datetime, message_id: int = 0) -> int:
        """Programme une nouvelle tâche.
        
        Returns:
            ID de la tâche créée
        """
        return self.db.add_task(channel_id, user_id, task_description, execute_at, message_id)
    
    def cancel_task(self, task_id: int, user_id: Optional[int] = None) -> bool:
        """Annule une tâche planifiée.
        
        Args:
            task_id: ID de la tâche
            user_id: Si fourni, vérifie que la tâche appartient à cet utilisateur
        """
        return self.db.cancel_task(task_id, user_id)
    
    def get_user_tasks(self, user_id: int, limit: int = 20) -> list[ScheduledTask]:
        """Récupère les tâches d'un utilisateur."""
        return self.db.get_user_tasks(user_id, limit)
    
    def count_pending_user_tasks(self, user_id: int) -> int:
        """Compte le nombre de tâches en attente pour un utilisateur."""
        return self.db.count_pending_user_tasks(user_id)
    
    def get_all_tasks(self, limit: int = 50) -> list[ScheduledTask]:
        """Récupère toutes les tâches."""
        return self.db.get_all_tasks(limit)
    
    async def start_worker(self):
        """Démarre le worker d'exécution des tâches."""
        if self._running:
            logger.warning("Worker déjà démarré")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Worker scheduler démarré")
    
    async def stop_worker(self):
        """Arrête le worker proprement."""
        if not self._running:
            return
        
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Worker scheduler arrêté")
    
    async def _worker_loop(self):
        """Boucle principale du worker."""
        last_cleanup = datetime.now(timezone.utc)
        
        while self._running:
            try:
                # Récupérer les tâches à exécuter
                pending_tasks = self.db.get_pending_tasks()
                
                for task in pending_tasks:
                    try:
                        logger.info(f"Exécution tâche #{task.id}: '{task.task_description}'")
                        
                        # Exécuter la tâche via l'executor fourni (avec message_id pour reply)
                        await self.executor(task.channel_id, task.user_id, task.task_description, task.message_id)
                        
                        # Marquer comme complétée
                        self.db.update_task_status(task.id, 'completed')
                        logger.info(f"Tâche #{task.id} terminée avec succès")
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de l'exécution de la tâche #{task.id}: {e}", exc_info=True)
                        self.db.update_task_status(task.id, 'failed')
                
                # Nettoyage quotidien (à 3h du matin UTC)
                now = datetime.now(timezone.utc)
                if (now - last_cleanup).total_seconds() > 86400 and now.hour == 3:
                    self.db.cleanup_old_tasks()
                    last_cleanup = now
                
            except Exception as e:
                logger.error(f"Erreur dans le worker scheduler: {e}", exc_info=True)
            
            # Attendre 30 secondes avant la prochaine vérification
            await asyncio.sleep(30)

