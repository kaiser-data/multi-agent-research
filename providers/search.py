"""Search provider abstraction layer supporting multiple backends."""

import os
import time
from typing import Optional

import requests


def search(
    query: str,
    provider: Optional[str] = None,
    num_results: int = 5,
) -> list[dict]:
    """
    Execute search across multiple providers with unified interface.

    Args:
        query: Search query string
        provider: Search provider ('duckduckgo', 'brave', 'serper')
                 Defaults to SEARCH_PROVIDER env var or 'duckduckgo'
        num_results: Number of results to return (1-10)

    Returns:
        List of search results: [{"title": str, "url": str, "snippet": str}, ...]

    Examples:
        >>> results = search("quantum computing", "duckduckgo", 5)
        >>> results = search("AI safety", "brave", 10)
    """
    if provider is None:
        provider = os.getenv("SEARCH_PROVIDER", "duckduckgo")

    num_results = max(1, min(num_results, 10))

    if provider == "duckduckgo":
        return _search_duckduckgo(query, num_results)
    elif provider == "brave":
        return _search_brave(query, num_results)
    elif provider == "serper":
        return _search_serper(query, num_results)
    else:
        raise ValueError(
            f"Unknown search provider: {provider}. "
            f"Supported: duckduckgo, brave, serper"
        )


def _search_duckduckgo(query: str, num_results: int) -> list[dict]:
    """
    Search using DuckDuckGo (free, no API key required).

    Uses duckduckgo-search library for instant answers and web results.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            for item in search_results:
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                })

        return results

    except ImportError:
        raise ImportError(
            "duckduckgo-search required for DuckDuckGo provider. "
            "Install with: pip install duckduckgo-search"
        )
    except Exception as e:
        print(f"⚠️  DuckDuckGo search failed: {e}")
        return []


def _search_brave(query: str, num_results: int) -> list[dict]:
    """
    Search using Brave Search API (2,000 free queries/month).

    Requires BRAVE_API_KEY environment variable.
    Get API key at: https://brave.com/search/api/
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError(
            "BRAVE_API_KEY required for Brave Search. "
            "Get your API key at https://brave.com/search/api/"
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            }
            params = {
                "q": query,
                "count": num_results,
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })

            return results

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"⚠️  Brave search failed after {max_retries} attempts: {e}")
                return []
            time.sleep(2 ** attempt)

    return []


def _search_serper(query: str, num_results: int) -> list[dict]:
    """
    Search using Serper.dev API (2,500 free queries/month).

    Requires SERPER_API_KEY environment variable.
    Get API key at: https://serper.dev
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError(
            "SERPER_API_KEY required for Serper. "
            "Get your API key at https://serper.dev"
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": query,
                "num": num_results,
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })

            return results

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"⚠️  Serper search failed after {max_retries} attempts: {e}")
                return []
            time.sleep(2 ** attempt)

    return []


def get_available_providers() -> list[dict]:
    """
    Get list of available search providers with metadata.

    Returns:
        List of provider info dicts with name, cost, and requirements
    """
    return [
        {
            "name": "duckduckgo",
            "display_name": "DuckDuckGo (Free)",
            "cost": "Free",
            "monthly_limit": "Unlimited",
            "requires_api_key": False,
            "description": "No API key needed, unlimited searches",
        },
        {
            "name": "brave",
            "display_name": "Brave Search (2K free/month)",
            "cost": "Free tier: 2,000/month",
            "monthly_limit": "2,000",
            "requires_api_key": True,
            "description": "High quality results, 2K free queries/month",
        },
        {
            "name": "serper",
            "display_name": "Serper.dev (2.5K free/month)",
            "cost": "Free tier: 2,500/month",
            "monthly_limit": "2,500",
            "requires_api_key": True,
            "description": "Google-powered results, 2.5K free queries/month",
        },
    ]
