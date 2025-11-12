"""### Web Cog
Outils de recherche et navigation web pour l'IA."""

import logging
import re
import time
import html
import os
from datetime import datetime, timezone
from typing import List, Dict
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from discord.ext import commands
from ddgs import DDGS

from common.llm import Tool, ToolCallRecord, ToolResponseRecord

logger = logging.getLogger(f'MARI4.{__name__.split(".")[-1]}')

# CONSTANTES ------------------------------------------------------

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_NUM_RESULTS = 4
DEFAULT_TIMEOUT = 15
CACHE_EXPIRY_HOURS = 12
SEARCH_CACHE_EXPIRY_SECONDS = 300
REQUEST_TIMEOUT = (5, 12)  # (connect, read)
MAX_REDIRECTS = 5

# Headers pour éviter les blocages
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Formats multimédias supportés
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v'}
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}

# Patterns pour nettoyer le contenu
NOISE_PATTERNS = [
    r'En poursuivant votre navigation.*?cookies.*?\.',
    r'This site uses cookies.*?privacy.*?\.',
    r'Nous utilisons des cookies.*?confidentialité.*?\.',
    r'Accept [aA]ll cookies',
    r'cookie policy',
    r'privacy policy',
    r'terms of service',
    r'\${[^}]+}',
    r'function\s*\([^)]*\)\s*\{[^}]*\}',
    r'var\s+\w+\s*=.*?;',
]

# Sélecteurs pour détecter le contenu principal
MAIN_CONTENT_SELECTORS = [
    'main', 'article', '.content', '.post', '.post-content', '.entry-content',
    '#content', '#main', '.article', '[role="main"]'
]

class Web(commands.Cog):
    """Cog pour les outils de recherche et navigation web utilisés par l'IA."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.page_chunks_cache: Dict[str, Dict] = {}
        self.search_cache: Dict[str, Dict] = {}
        
        # Outils globaux exportés
        self.GLOBAL_TOOLS = [
            Tool(
                name='search_web',
                description='Recherche sur le web et retourne des resultats avec extraits. Utilise pour toute info recente ou inconnue. Si besoin de plus de details, utilise ensuite read_web_page sur une URL specifique.',
                properties={
                    'query': {'type': 'string', 'description': 'Requete de recherche concise'},
                    'lang': {'type': 'string', 'description': 'Code langue (fr, en, es, etc.). Par defaut: fr', 'default': 'fr'}
                },
                function=self._tool_search_web
            ),
            Tool(
                name='read_web_page',
                description='Lit le contenu complet d\'une URL specifique. Utilise si: 1) l\'utilisateur donne une URL, 2) les extraits de search_web sont insuffisants et tu veux approfondir.',
                properties={
                    'url': {'type': 'string', 'description': 'URL complete de la page'}
                },
                function=self._tool_read_web_page
            )
        ]
        
        logger.info("Web cog initialisé avec 2 outils")
    
    # MÉTHODES UTILITAIRES --------------------------------------------
    
    def clean_text_content(self, text: str) -> str:
        """Nettoie le contenu textuel des artefacts web."""
        if not text:
            return ""
        
        # Décoder les entités HTML
        text = html.unescape(text)
        
        # Supprimer les patterns de bruit
        for pattern in NOISE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Nettoyer les espaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Supprimer les lignes vides répétées
        lines = text.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines).strip()
    
    def _is_low_quality_chunk(self, chunk: str) -> bool:
        """Détecte si un chunk est de basse qualité."""
        chunk = chunk.strip()
        
        if len(chunk) < 50:
            return True
        
        # Trop de caractères spéciaux
        special_char_ratio = len(re.findall(r'[^\w\s\-.,;:!?()"\']', chunk)) / len(chunk)
        if special_char_ratio > 0.3:
            return True
        
        # Détection de contenu JavaScript
        js_indicators = ['function', 'var ', 'const ', 'let ', 'return', '${', '};', 'console.log']
        js_count = sum(1 for indicator in js_indicators if indicator in chunk.lower())
        if js_count > 2:
            return True
        
        return False
    
    def search_web_pages(self, query: str, lang: str = 'fr', num_results: int = DEFAULT_NUM_RESULTS) -> List[Dict]:
        """Effectue une recherche web avec DDGS."""
        cache_key = f"{lang}:{query.strip().lower()}"
        cache_entry = self.search_cache.get(cache_key)
        now = time.time()
        if cache_entry and now - cache_entry['timestamp'] < SEARCH_CACHE_EXPIRY_SECONDS:
            return cache_entry['results'][:num_results]
        
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query=query, 
                    region=f'{lang}-{lang}',
                    max_results=min(num_results, 10)
                )
                parsed_results = [
                    {
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'description': r.get('body', '')
                    }
                    for r in results
                ]
                self.search_cache[cache_key] = {
                    'results': parsed_results,
                    'timestamp': now
                }
                return parsed_results
        except Exception as e:
            logger.error(f"Erreur recherche web: {e}")
            return []
    
    def fetch_page_chunks(self, url: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Récupère et divise le contenu d'une page web."""
        # Vérifier le cache
        cache_key = f"{url}_{chunk_size}"
        if cache_key in self.page_chunks_cache:
            cache_entry = self.page_chunks_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < CACHE_EXPIRY_HOURS * 3600:
                return cache_entry['chunks']
        
        try:
            # Récupération de la page
            session = requests.Session()
            session.max_redirects = MAX_REDIRECTS
            response = session.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True
            )
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Suppression des éléments non pertinents
            for tag in soup.find_all(["script", "style", "header", "footer", "nav", "aside", "iframe",
                                      "noscript", "form", "button", "svg", "select", "input", "textarea", "label"]):
                tag.decompose()
            for selector in [".ad", ".ads", ".cookie", ".popup", ".banner", ".sidebar", ".menu",
                             ".comments", ".navigation", ".breadcrumb", ".social", ".share",
                             ".related", ".recommended", ".widget"]:
                for element in soup.select(selector):
                    element.decompose()
            
            # Détection du contenu principal
            main_content = None
            for selector in MAIN_CONTENT_SELECTORS:
                content = soup.select(selector)
                if content and len(str(content[0])) > 500:
                    main_content = content[0]
                    break
            
            text_container = main_content or soup.find('body') or soup
            
            # Extraction du texte
            text = ""
            for elem in text_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                content = elem.get_text(strip=True)
                if len(content) > 10:
                    prefix = f"## " if elem.name.startswith('h') else ""
                    text += f"\n{prefix}{content}\n"
            
            # Nettoyage
            text = self.clean_text_content(text)
            
            # Division en chunks
            chunks = []
            paragraphs = [p for p in re.split(r'\n\n+', text) if p.strip()]
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(paragraph.strip()) < 30:
                    continue
                    
                if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Filtrer les chunks de qualité
            chunks = [c for c in chunks if len(c.strip()) > 100 and not self._is_low_quality_chunk(c)]
            
            # Mise en cache
            self.page_chunks_cache[cache_key] = {
                'chunks': chunks, 
                'timestamp': time.time()
            }
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lecture page {url}: {e}")
            return []
    
    # OUTILS ----------------------------------------------------------
    
    def _tool_search_web(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Outil de recherche web avec extraits enrichis."""
        query = tool_call.arguments.get('query')
        lang = tool_call.arguments.get('lang', 'fr')
        
        if not query:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'Requete manquante'},
                created_at=datetime.now(timezone.utc)
            )
        
        # Recherche
        results = self.search_web_pages(query, lang, DEFAULT_NUM_RESULTS)
        
        if not results:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'Aucun resultat trouve'},
                created_at=datetime.now(timezone.utc)
            )
        
        # Construire la réponse avec les résultats enrichis
        pruned_results = []
        seen_urls = set()
        for item in results:
            url = item.get('url', '')
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            pruned_results.append(item)
            if len(pruned_results) >= DEFAULT_NUM_RESULTS:
                break
        
        response_data = {
            'query': query,
            'results': pruned_results,
            'total': len(pruned_results),
            'note': 'Si les extraits sont insuffisants, utilise read_web_page sur une URL specifique pour plus de details.'
        }
        
        header = f'Recherche de "{query}"'
        
        return ToolResponseRecord(
            tool_call_id=tool_call.id,
            response_data=response_data,
            created_at=datetime.now(timezone.utc),
            metadata={'header': header}
        )
    
    def _tool_read_web_page(self, tool_call: ToolCallRecord, context_data) -> ToolResponseRecord:
        """Outil pour lire le contenu d'une page web."""
        url = tool_call.arguments.get('url')
        if not url:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'URL manquante'},
                created_at=datetime.now(timezone.utc)
            )
        
        # Validation URL
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https'] or not parsed.netloc:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'URL invalide'},
                created_at=datetime.now(timezone.utc)
            )
        
        chunks = self.fetch_page_chunks(url)
        
        if not chunks:
            return ToolResponseRecord(
                tool_call_id=tool_call.id,
                response_data={'error': 'Impossible de lire cette page'},
                created_at=datetime.now(timezone.utc)
            )
        
        # Premier chunk (le plus important)
        content = chunks[0]
        
        domain = url.split("//")[-1].split("/")[0]
        return ToolResponseRecord(
            tool_call_id=tool_call.id,
            response_data={
                'url': url,
                'content': content,
                'total_chunks_available': len(chunks)
            },
            created_at=datetime.now(timezone.utc),
            metadata={'header': f'Lecture de [{domain}](<{url}>)'}
        )

async def setup(bot):
    await bot.add_cog(Web(bot))

