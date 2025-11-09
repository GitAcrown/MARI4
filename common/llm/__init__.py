"""### LLM Package
API GPT pour MARI4 - Interface simplifiée et élégante."""

# Façade principale
from .api import MariaGptApi, MariaResponse, MariaSessionHandle

# Tools
from .tools import Tool, ToolRegistry, create_simple_tool

# Context components (pour construire des messages custom)
from .context import (
    TextComponent,
    ImageComponent,
    MetadataComponent,
    MessageRecord,
    AssistantRecord,
    ToolCallRecord,
    ToolResponseRecord
)

# Client (si besoin d'accès direct)
from .client import MariaLLMClient, MariaLLMError, MariaOpenAIError

# Exports publics
__all__ = [
    # API principale
    'MariaGptApi',
    'MariaResponse',
    'MariaSessionHandle',
    
    # Tools
    'Tool',
    'ToolRegistry',
    'create_simple_tool',
    
    # Components
    'TextComponent',
    'ImageComponent',
    'MetadataComponent',
    'MessageRecord',
    'AssistantRecord',
    'ToolCallRecord',
    'ToolResponseRecord',
    
    # Client
    'MariaLLMClient',
    'MariaLLMError',
    'MariaOpenAIError',
]

__version__ = '4.0.0'
__author__ = 'MARI4'

