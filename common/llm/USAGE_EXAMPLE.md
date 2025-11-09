
# Exemples d'utilisation de l'API GPT MARI4

## Initialisation de base

```python
from common.llm import MariaGptApi, Tool, ToolCallRecord, ToolResponseRecord
from datetime import datetime
import zoneinfo

# Fonction pour générer le prompt système
def get_developer_prompt():
    paris_tz = zoneinfo.ZoneInfo("Europe/Paris")
    now = datetime.now(paris_tz)
    
    return f"""Tu es MARIA, assistante IA sur Discord.
[META]
Date actuelle: {now.strftime('%A %Y-%m-%d %H:%M:%S')} (Paris)
[STYLE]
Être concise, directe et informelle.
"""

# Initialisation de l'API
api = MariaGptApi(
    api_key="sk-...",
    developer_prompt_template=get_developer_prompt,
    completion_model='gpt-5-mini',
    context_window=16384,
    context_age_hours=6
)
```

## Enregistrement d'outils

```python
# Outil simple
def tool_get_time(tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    
    return ToolResponseRecord(
        tool_call_id=tool_call.id,
        response_data={'time': now.isoformat()},
        created_at=now
    )

time_tool = Tool(
    name='get_current_time',
    description='Obtenir l\'heure actuelle',
    properties={},
    function=tool_get_time
)

# Outil async
async def tool_search_web(tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
    query = tool_call.arguments.get('query')
    # ... logique de recherche ...
    from datetime import datetime, timezone
    
    return ToolResponseRecord(
        tool_call_id=tool_call.id,
        response_data={'results': ['...']},
        created_at=datetime.now(timezone.utc)
    )

search_tool = Tool(
    name='search_web',
    description='Rechercher sur le web',
    properties={
        'query': {
            'type': 'string',
            'description': 'Requête de recherche'
        }
    },
    function=tool_search_web
)

# Enregistrement
api.add_tools(time_tool, search_tool)
```

## Utilisation dans un Cog Discord

```python
import discord
from discord.ext import commands

class ChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.gpt_api = MariaGptApi(...)  # Initialisation
        
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignorer les bots
        if message.author.bot:
            return
        
        # Ingérer TOUS les messages (pour le contexte)
        await self.gpt_api.ingest_message(message.channel, message)
        
        # Décider si on répond (logique métier)
        if not self.should_respond(message):
            return
        
        # Lancer la complétion
        async with message.channel.typing():
            response = await self.gpt_api.run_completion(
                message.channel,
                trigger_message=message
            )
            
            # Envoyer la réponse
            await message.reply(response.text, mention_author=False)
    
    def should_respond(self, message: discord.Message) -> bool:
        # Logique de détection (mentions, etc.)
        return self.bot.user.mentioned_in(message)
```

## Gestion de session

```python
# Obtenir un handle de session
handle = await api.ensure_session(channel)

# Statistiques
stats = handle.get_stats()
print(f"Messages: {stats['context_stats']['total_messages']}")
print(f"Tokens: {stats['context_stats']['total_tokens']}")

# Messages récents
recent = handle.get_recent_messages(count=5)
for msg in recent:
    print(f"{msg.role}: {msg.full_text[:50]}...")

# Vider l'historique
await api.forget(channel)
```

## Gestion des outils dynamiques

```python
# Ajouter des outils depuis d'autres cogs
class WebCog(commands.Cog):
    def __init__(self, bot, gpt_api):
        self.bot = bot
        self.gpt_api = gpt_api
        
        # Enregistrer nos outils
        self.register_tools()
    
    def register_tools(self):
        search_tool = Tool(...)
        self.gpt_api.add_tools(search_tool)
    
    async def cog_unload(self):
        # Retirer nos outils
        self.gpt_api.remove_tool('search_web')
```

## Statistiques globales

```python
stats = api.get_stats()

print(f"Client: {stats['client_stats']}")
# {'completions': 42, 'transcriptions': 5, 'errors': 0}

print(f"Sessions: {stats['session_stats']}")
# {'active_sessions': 3, 'cache_stats': {...}}

print(f"Outils: {stats['tools_count']}")
# 5
```

## Fermeture propre

```python
async def shutdown():
    await api.close()
```

## Thread-safety

L'API gère automatiquement la concurrence :

```python
# Ces deux appels simultanés seront exécutés séquentiellement
# grâce au lock interne de la session
task1 = asyncio.create_task(api.run_completion(channel, msg1))
task2 = asyncio.create_task(api.run_completion(channel, msg2))

response1, response2 = await asyncio.gather(task1, task2)
# Pas de race condition ! ✓
```

## Composants personnalisés

```python
from common.llm import TextComponent, ImageComponent, MetadataComponent

# Créer un message custom (usage avancé)
components = [
    TextComponent("Voici une image :"),
    ImageComponent("https://example.com/image.jpg", detail='high'),
    MetadataComponent('CUSTOM', key='value', other='data')
]

# Ajouter directement au contexte (usage avancé)
session = api.session_manager.get_or_create_session(channel)
session.context.add_user_message(components, name="custom_user")
```

