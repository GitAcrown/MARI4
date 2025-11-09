# MARIA v4 (MARI4)

Bot Discord conversationnel propulsé par GPT-5. Conversation naturelle, recherche web, transcription audio, mémoire long terme.

## Fonctionnalités

- **Conversation** : Contexte complet, ton naturel, se souvient des utilisateurs
- **Recherche web** : Infos en temps réel avec sources
- **Transcription** : Messages vocaux automatiques
- **Calculs** : Expressions mathématiques
- **Analyse** : Images et vidéos

## Installation

```bash
pip install -r requirements.txt
```

Créer `.env` :
```env
TOKEN=votre_token_discord
APP_ID=votre_app_id
OPENAI_API_KEY=votre_clé_openai
```

```bash
python bot.py
```

## Commandes

- `/info` - Statistiques du bot
- `/chatbot mode` - Mode de réponse (off/strict/greedy)
- `/chatbot forget` - Efface l'historique
- `/memory show` - Affiche votre profil
- `/memory reset` - Efface votre profil

## Technologies

Discord.py • OpenAI GPT-5 • DDGS • SQLite