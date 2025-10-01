"""Researcher agent - executes searches and collects results."""

import time
from typing import Any

from langchain.schema import AIMessage

from providers import search


def researcher_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute searches for each research step and collect results.

    Performs search queries for each step in the plan, normalizes results,
    and handles errors gracefully. Deduplicates URLs across all results.

    Args:
        state: Research state containing:
            - plan: List of research steps
            - search_provider: Search provider to use
            - num_results: Results per step

    Returns:
        State updates with:
            - results: List of search result dicts with keys:
                      title, url, snippet, step_index
            - messages: List with researcher's message

    Example:
        >>> state = {"plan": ["AI adoption rates", "Job displacement"]}
        >>> result = researcher_node(state)
        >>> len(result["results"])
        10  # 5 results per step * 2 steps
    """
    plan = state.get("plan", [])
    search_provider = state.get("search_provider", "duckduckgo")
    num_results = state.get("num_results", 5)

    if not plan:
        return {
            "results": [],
            "messages": [AIMessage(content="⚠️  No research plan to execute")],
        }

    all_results = []
    seen_urls = set()

    for idx, step in enumerate(plan):
        try:
            # Execute search for this step
            step_results = search(
                query=step,
                provider=search_provider,
                num_results=num_results,
            )

            # Add step index and deduplicate
            for result in step_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    result["step_index"] = idx
                    all_results.append(result)
                    seen_urls.add(url)

            # Rate limiting - be nice to search providers
            if idx < len(plan) - 1:  # Don't sleep after last step
                time.sleep(0.5)

        except ValueError as e:
            # Provider configuration error
            return {
                "results": [],
                "messages": [AIMessage(content=f"⚠️  Search configuration error: {str(e)}")],
            }

        except Exception as e:
            # Individual search failure - continue with other steps
            print(f"⚠️  Search failed for step '{step}': {e}")
            continue

    if not all_results:
        return {
            "results": [],
            "messages": [AIMessage(content="⚠️  No search results found for any step")],
        }

    return {
        "results": all_results,
        "messages": [
            AIMessage(
                content=f"Found {len(all_results)} unique results across {len(plan)} research steps"
            )
        ],
    }
