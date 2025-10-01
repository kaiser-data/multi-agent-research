"""Planner agent - decomposes research queries into focused steps."""

import json
from typing import Any

from langchain.schema import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from providers import get_llm


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate 2-5 focused research steps from user query.

    Uses LLM to break down complex queries into actionable research steps.
    Extended thinking is automatically used for Claude 3.7+/Sonnet 4+ models.

    Args:
        state: Research state containing:
            - query: User's research question
            - llm_provider: LLM provider to use
            - model_name: Model identifier

    Returns:
        State updates with:
            - plan: List of research step strings
            - messages: List with planner's message

    Example:
        >>> state = {"query": "impact of AI on jobs", "llm_provider": "anthropic"}
        >>> result = planner_node(state)
        >>> result["plan"]
        ['Current AI adoption rates by industry', 'Jobs displaced by automation', ...]
    """
    try:
        llm = get_llm(
            provider=state.get("llm_provider"),
            model_name=state.get("model_name"),
        )
    except ValueError as e:
        # Return error state if LLM initialization fails
        return {
            "plan": [state["query"]],  # Fallback to original query
            "messages": [AIMessage(content=f"⚠️  Planner error: {str(e)}")],
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research planner. Given a research query, decompose it into "
            "2-5 focused, actionable research steps that will help answer the question. "
            "Each step should be a specific aspect to investigate.\n\n"
            "Return ONLY a JSON array of strings, nothing else. No explanations.\n\n"
            "Example: [\"Step 1 description\", \"Step 2 description\", \"Step 3 description\"]"
        ),
        (
            "human",
            "Research query: {query}\n\n"
            "Generate 2-5 research steps as a JSON array:"
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"query": state["query"]})

        # Try to parse JSON response
        try:
            plan = json.loads(response)

            # Validate plan structure
            if not isinstance(plan, list) or len(plan) < 2 or len(plan) > 5:
                plan = [state["query"]]  # Fallback to original query

            # Ensure all items are strings
            plan = [str(step) for step in plan]

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract steps from text
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            plan = lines[:5] if lines else [state["query"]]

        return {
            "plan": plan,
            "messages": [AIMessage(content=f"Created research plan with {len(plan)} steps")],
        }

    except Exception as e:
        # Fallback to original query if anything goes wrong
        return {
            "plan": [state["query"]],
            "messages": [AIMessage(content=f"⚠️  Planning failed, using direct query: {str(e)}")],
        }
