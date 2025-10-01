"""Writer agent - synthesizes research findings into brief with citations."""

from typing import Any

from langchain.schema import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from providers import get_llm


def writer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Synthesize research findings into concise brief with inline citations.

    Creates a ≤200-word summary with inline citations [1][2][3] for all
    factual claims. Generates deduplicated references list.

    Args:
        state: Research state containing:
            - query: Original research question
            - results: List of search results
            - llm_provider: LLM provider to use
            - model_name: Model identifier

    Returns:
        State updates with:
            - draft: Brief text with inline citations
            - references: List of unique URLs in citation order
            - messages: List with writer's message

    Example:
        >>> state = {"query": "AI impact", "results": [...]}
        >>> result = writer_node(state)
        >>> "[1]" in result["draft"]
        True
    """
    query = state.get("query", "")
    results = state.get("results", [])

    if not results:
        return {
            "draft": "No research results available to synthesize.",
            "references": [],
            "messages": [AIMessage(content="⚠️  No results to write about")],
        }

    try:
        llm = get_llm(
            provider=state.get("llm_provider"),
            model_name=state.get("model_name"),
        )
    except ValueError as e:
        return {
            "draft": f"Error: Could not initialize LLM - {str(e)}",
            "references": [],
            "messages": [AIMessage(content=f"⚠️  Writer error: {str(e)}")],
        }

    # Format research results for LLM
    results_text = "\n\n".join([
        f"[{i+1}] {r['title']}\n"
        f"URL: {r['url']}\n"
        f"Content: {r['snippet']}"
        for i, r in enumerate(results[:20])  # Limit to top 20 for context
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research writer specializing in synthesizing information. "
            "Your task is to write a concise research brief (≤200 words) that answers "
            "the research question using the provided sources.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Include inline citations [1][2][3] for ALL factual claims\n"
            "2. Stay under 200 words - be concise and focused\n"
            "3. Answer the research question directly\n"
            "4. Use clear, professional language\n"
            "5. Only cite sources that are actually provided\n\n"
            "Write ONLY the brief, no preamble or meta-commentary."
        ),
        (
            "human",
            "Research Question: {query}\n\n"
            "Sources:\n{results}\n\n"
            "Write a ≤200-word research brief with inline citations:"
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        draft = chain.invoke({
            "query": query,
            "results": results_text,
        })

        # Extract unique references in order
        references = []
        seen_urls = set()

        for result in results[:20]:
            url = result.get("url", "")
            if url and url not in seen_urls:
                references.append(url)
                seen_urls.add(url)

        return {
            "draft": draft.strip(),
            "references": references,
            "messages": [
                AIMessage(content=f"Draft complete with {len(references)} references")
            ],
        }

    except Exception as e:
        return {
            "draft": f"Error generating brief: {str(e)}",
            "references": [],
            "messages": [AIMessage(content=f"⚠️  Writing failed: {str(e)}")],
        }
