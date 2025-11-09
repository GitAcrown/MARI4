"""### Memory > Profile
Gestion des profils utilisateur."""

import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger('MARI4.memory.profile')

@dataclass
class UserProfile:
    """Profil utilisateur avec compteur de messages pour mises à jour automatiques."""
    user_id: int
    content: str
    created_at: datetime
    updated_at: datetime
    messages_since_update: int = 0
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire."""
        return {
            'user_id': self.user_id,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages_since_update': self.messages_since_update
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Crée depuis un dictionnaire."""
        return cls(
            user_id=data['user_id'],
            content=data['content'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            messages_since_update=data.get('messages_since_update', 0)
        )
    
    def should_update(self) -> bool:
        """Détermine si le profil doit être mis à jour automatiquement.
        
        Règles (moins fréquentes car l'IA peut déclencher manuellement):
        - Au moins 30 messages depuis dernière MAJ
        - ET au moins 12h écoulées
        """
        hours_elapsed = (datetime.now(timezone.utc) - self.updated_at).total_seconds() / 3600
        return self.messages_since_update >= 30 and hours_elapsed >= 12
    
    def increment_messages(self) -> None:
        """Incrémente le compteur de messages."""
        self.messages_since_update += 1
    
    def reset_counter(self) -> None:
        """Réinitialise le compteur après mise à jour."""
        self.messages_since_update = 0
        self.updated_at = datetime.now(timezone.utc)

