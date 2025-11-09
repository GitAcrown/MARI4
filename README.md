# MARIA v4 (MARI4)

Bot Discord conversationnel intelligent propulsé par GPT.

## 🤖 Présentation

MARIA est un bot Discord capable de :
- Converser naturellement avec les utilisateurs
- Rechercher des informations sur internet en temps réel
- Transcrire des messages vocaux automatiquement
- Effectuer des calculs mathématiques
- Analyser des images et vidéos

## ✨ Fonctionnalités principales

### 💬 Conversation intelligente
- Comprend le contexte complet des conversations
- Ton décontracté et naturel adapté aux jeunes adultes
- Gestion robuste des mentions multiples simultanées
- **Mémoire long terme** : Se souvient des utilisateurs entre les sessions

### 🔍 Recherche web
- Recherche automatique d'informations récentes
- Lecture et analyse de pages web
- Résultats avec sources cliquables

### 🎙️ Transcription audio
- Transcription automatique des messages vocaux
- Réaction 💡 pour transcrire à la demande

### 🧮 Calculs
- Évaluation d'expressions mathématiques
- Conversions d'unités

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/votre-repo/MARI4.git
cd MARI4

# Installer les dépendances
pip install -r requirements.txt

# Configurer le .env
cp .env.example .env
# Éditer .env avec vos clés API

# Lancer le bot
python bot.py
```

## ⚙️ Configuration

Créer un fichier `.env` avec :
```env
TOKEN=votre_token_discord
APP_ID=votre_app_id
OPENAI_API_KEY=votre_clé_openai
```

## 📝 Commandes

- `/info` - Affiche les informations et statistiques du bot
- `/chatbot mode` - Configure le mode de réponse (off/strict/greedy)
- `/chatbot forget` - Efface l'historique de conversation du salon
- `/auto transcription` - Active/désactive la transcription automatique
- `/memory show` - Affiche votre carte d'identité enregistrée
- `/memory reset` - Efface toutes vos informations enregistrées

## 🏗️ Architecture

```
MARI4/
├── bot.py              # Point d'entrée
├── common/
│   ├── dataio.py       # Gestion base de données
│   ├── llm/            # API GPT modulaire
│   └── memory/         # Système de mémoire long terme
├── cogs/
│   ├── chat/           # Conversation principale
│   ├── web/            # Outils de recherche web
│   ├── auto/           # Fonctionnalités automatiques
│   ├── utils/          # Outils utilitaires
│   ├── status/         # Mise à jour du statut
│   └── core/           # Commandes administratives
└── requirements.txt
```

## 🔧 Technologies

- **Discord.py** - Framework Discord
- **OpenAI API** - Modèles GPT
- **DDGS** - Recherche web
- **BeautifulSoup** - Extraction de contenu web
- **SQLite** - Stockage de données

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails