"""### LLM > Tools
Gestion des outils (tools/functions) disponibles pour l'IA."""

import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Union, Awaitable, Any
from dataclasses import dataclass

from .context import ToolCallRecord, ToolResponseRecord

logger = logging.getLogger(f'MARI4.llm.tools')

# TOOL DEFINITION -------------------------------------------------

@dataclass
class Tool:
    """Définition d'un outil disponible pour l'IA.
    
    Suit le format OpenAI function calling avec strict mode.
    """
    name: str
    description: str
    properties: dict  # JSON Schema des paramètres
    function: Union[Callable, Callable[..., Awaitable]]
    extras: dict = None
    
    def __post_init__(self):
        if self.extras is None:
            self.extras = {}
        self._required = list(self.properties.keys())
    
    async def execute(self, tool_call: ToolCallRecord, context_data: Any = None) -> ToolResponseRecord:
        """Exécute la fonction de l'outil.
        
        Args:
            tool_call: Enregistrement de l'appel d'outil
            context_data: Données contextuelles optionnelles
            
        Returns:
            ToolResponseRecord avec la réponse
        """
        try:
            # Appel sync ou async
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(tool_call, context_data)
            else:
                result = self.function(tool_call, context_data)
            
            # Si la fonction retourne déjà un ToolResponseRecord, on le retourne
            if isinstance(result, ToolResponseRecord):
                return result
            
            # Sinon on wrappe dans un ToolResponseRecord
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data=result if isinstance(result, dict) else {'result': result},
                created_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'outil '{self.name}': {e}")
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': str(e)},
                created_at=datetime.now(timezone.utc)
            )
    
    def to_openai_dict(self) -> dict:
        """Convertit en format OpenAI function calling."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'strict': True,
                'parameters': {
                    'type': 'object',
                    'properties': self.properties,
                    'required': self._required,
                    'additionalProperties': False
                }
            }
        }

# TOOL REGISTRY ---------------------------------------------------

class ToolRegistry:
    """Registre central des outils disponibles."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._compiled_cache: list[dict] | None = None
        logger.info("ToolRegistry initialisé")
    
    def register(self, tool: Tool) -> None:
        """Enregistre un outil."""
        if tool.name in self._tools:
            logger.warning(f"Outil '{tool.name}' déjà enregistré, écrasement")
        
        self._tools[tool.name] = tool
        self._compiled_cache = None  # Invalider le cache
        logger.info(f"Outil '{tool.name}' enregistré")
    
    def register_multiple(self, *tools: Tool) -> None:
        """Enregistre plusieurs outils."""
        for tool in tools:
            self.register(tool)
    
    def unregister(self, name: str) -> None:
        """Désenregistre un outil."""
        if name in self._tools:
            del self._tools[name]
            self._compiled_cache = None
            logger.info(f"Outil '{name}' désenregistré")
    
    def get(self, name: str) -> Tool | None:
        """Récupère un outil par son nom."""
        return self._tools.get(name)
    
    def get_all(self) -> list[Tool]:
        """Récupère tous les outils."""
        return list(self._tools.values())
    
    def get_compiled(self) -> list[dict]:
        """Retourne les outils compilés au format OpenAI (avec cache)."""
        if self._compiled_cache is None:
            self._compiled_cache = [t.to_openai_dict() for t in self._tools.values()]
        return self._compiled_cache
    
    def clear(self) -> None:
        """Vide le registre."""
        self._tools.clear()
        self._compiled_cache = None
        logger.info("ToolRegistry vidé")
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools

# HELPER FUNCTIONS ------------------------------------------------

def create_simple_tool(name: str, 
                      description: str, 
                      properties: dict,
                      function: Callable,
                      **extras) -> Tool:
    """Helper pour créer rapidement un outil simple."""
    return Tool(
        name=name,
        description=description,
        properties=properties,
        function=function,
        extras=extras
    )

