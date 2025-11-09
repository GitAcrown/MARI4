"""### Utils Cog
Outils utilitaires pour l'IA (calculs mathématiques, conversions, etc.)."""

import logging
from discord.ext import commands
import numexpr as ne

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

class Utils(commands.Cog):
    """Cog fournissant des outils utilitaires pour l'IA."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        
        # Outils exportés pour l'API GPT
        self.GLOBAL_TOOLS = [
            Tool(
                name='math_eval',
                description='Évalue des expressions mathématiques. Utilise la syntaxe Python standard (opérateurs: +, -, *, /, **, %, //, etc.).',
                properties={
                    'expression': {
                        'type': 'string',
                        'description': "L'expression mathématique à évaluer (ex: '2 + 2', '3.14 * 10**2', 'sqrt(16)')"
                    }
                },
                function=self._tool_math_eval
            )
        ]
        
        logger.info("Utils cog initialisé avec 1 outil")
    
    def _tool_math_eval(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Outil pour évaluer des expressions mathématiques."""
        from datetime import datetime, timezone
        
        expression = tool_call.arguments.get('expression')
        if not expression:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'Aucune expression mathématique fournie.'},
                created_at=datetime.now(timezone.utc)
            )
        
        try:
            # Évaluation sécurisée avec numexpr
            result = float(ne.evaluate(expression))
            
            # Convertir en int si c'est un nombre entier
            if result.is_integer():
                result = int(result)
            
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={
                    'expression': expression,
                    'result': result
                },
                created_at=datetime.now(timezone.utc),
                metadata={'header': f"Calcul de `{expression}`"}
            )
            
        except Exception as e:
            logger.error(f"Erreur évaluation math '{expression}': {e}")
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={
                    'expression': expression,
                    'error': f"Erreur de calcul: {str(e)}"
                },
                created_at=datetime.now(timezone.utc)
            )

async def setup(bot):
    await bot.add_cog(Utils(bot))

