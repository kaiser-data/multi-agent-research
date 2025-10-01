"""Reviewer agent - validates citations and triggers revisions."""

from typing import Any

from langchain.schema import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from providers import get_llm


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that all factual claims have proper citations.

    Reviews the draft to ensure all claims are properly cited.
    Triggers one revision if issues are found, then finalizes output.

    Args:
        state: Research state containing:
            - draft: Written brief with citations
            - revised_once: Whether revision has been done
            - llm_provider: LLM provider to use
            - model_name: Model identifier

    Returns:
        State updates with either:
            - revised_once: True (triggers revision loop)
            - final: Approved draft text
            - messages: List with reviewer's message

    Example:
        >>> state = {"draft": "AI is growing [1].", "revised_once": False}
        >>> result = reviewer_node(state)
        >>> "final" in result or "revised_once" in result
        True
    """
    draft = state.get("draft", "")
    revised_once = state.get("revised_once", False)

    if not draft:
        return {
            "final": "No draft available.",
            "messages": [AIMessage(content="⚠️  No draft to review")],
        }

    try:
        llm = get_llm(
            provider=state.get("llm_provider"),
            model_name=state.get("model_name"),
        )
    except ValueError as e:
        # If we can't get LLM, approve the draft as-is
        return {
            "final": draft,
            "messages": [AIMessage(content=f"⚠️  Reviewer unavailable, auto-approving: {str(e)}")],
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a fact-checker reviewing research briefs. Your job is to verify "
            "that all factual claims have proper inline citations [1][2][3].\n\n"
            "Review the brief and respond with EXACTLY ONE of:\n"
            "- 'APPROVED' if all claims are properly cited\n"
            "- 'NEEDS_REVISION: <specific reason>' if citations are missing or inadequate\n\n"
            "Be strict but fair. Every factual statement needs a citation."
        ),
        (
            "human",
            "Review this research brief:\n\n{draft}\n\n"
            "Provide your assessment (APPROVED or NEEDS_REVISION):"
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        review = chain.invoke({"draft": draft})

        # Check if revision is needed and allowed
        needs_revision = "NEEDS_REVISION" in review.upper()

        if needs_revision and not revised_once:
            # Trigger one revision
            return {
                "revised_once": True,
                "messages": [
                    AIMessage(content=f"Requesting revision: {review}")
                ],
            }

        # Either approved or already revised once - finalize
        return {
            "final": draft,
            "messages": [
                AIMessage(
                    content="Research complete" if not needs_revision
                    else "Revision limit reached, finalizing"
                )
            ],
        }

    except Exception as e:
        # On error, approve draft to prevent blocking
        return {
            "final": draft,
            "messages": [
                AIMessage(content=f"⚠️  Review failed, auto-approving: {str(e)}")
            ],
        }
