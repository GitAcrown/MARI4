# MARIA v4 (MARI4)

Bot Discord conversationnel propulsé par GPT-5. MARIA tient des discussions naturelles, mémorise ce que les membres lui disent et peut gérer des rappels toute seule, sans intervention humaine.

## Fonctionnalités

### Conversation
- Comprend le fil de la discussion et adapte son ton.
- Retient les préférences clairement exprimées par chaque membre.

### Rappels autonomes
- Comprend les demandes du type : “Rappelle-moi dans 2h…”.
- Programme et exécute les rappels en arrière-plan, puis répond sur le message d’origine.
- Permet à chacun de consulter/annuler ses rappels via la commande `/tasks`.
- Limite volontairement les programmations en chaîne (max 5 dans une même réponse).

### Outils pratiques
- Recherche d’informations sur le web (sources fournies).
- Transcription automatique des messages vocaux.
- Analyse de visuels (selon les capacités du modèle).
- Calculs et petites aides de productivité.

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
- `/memory` - Affiche votre profil mémorisé
- `/tasks` - Vue interactive de vos tâches planifiées (annulation via bouton)

## Technologies

Discord.py · OpenAI GPT-5 · Recherche web DDGS · SQLite · LayoutView Discord UI