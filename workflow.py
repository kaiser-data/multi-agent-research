"""LangGraph workflow orchestration for multi-agent research."""

import operator
from typing import Annotated, Any

from langchain.schema import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents import planner_node, researcher_node, reviewer_node, writer_node


class ResearchState(TypedDict):
    """
    State schema for research workflow.

    Attributes:
        messages: Accumulated messages from agents
        query: User's research question
        plan: List of research steps
        results: Search results with metadata
        draft: Written brief with citations
        final: Approved final output
        references: Deduplicated URL list
        revised_once: Whether revision has occurred
        llm_provider: LLM provider name
        model_name: Model identifier
        search_provider: Search provider name
        num_results: Results per search step
    """
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    plan: list[str]
    results: list[dict]
    draft: str
    final: str
    references: list[str]
    revised_once: bool
    llm_provider: str
    model_name: str
    search_provider: str
    num_results: int


def _should_revise(state: ResearchState) -> str:
    """
    Conditional edge: determine if draft needs revision.

    Args:
        state: Current workflow state

    Returns:
        "writer" if revision needed and allowed, else END
    """
    # If final is set, we're done
    if state.get("final"):
        return END

    # If draft exists but not final, and not revised yet, go to writer
    if state.get("draft") and not state.get("revised_once", False):
        return "writer"

    return END


def build_workflow() -> StateGraph:
    """
    Construct LangGraph workflow with agent nodes and edges.

    Workflow structure:
        Entry → Planner → Researcher → Writer → Reviewer → END
                                         ↑          |
                                         └──────────┘
                                      (revision loop)

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> graph = build_workflow()
        >>> result = graph.invoke({"query": "AI safety", ...})
    """
    workflow = StateGraph(ResearchState)

    # Add agent nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    # Conditional edge for revision loop
    workflow.add_conditional_edges(
        "reviewer",
        _should_revise,
        {
            "writer": "writer",
            END: END,
        },
    )

    # Compile with checkpointer
    return workflow.compile(checkpointer=MemorySaver())


def run_research(
    query: str,
    llm_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-5-20250929",
    search_provider: str = "duckduckgo",
    num_results: int = 5,
) -> dict[str, Any]:
    """
    Execute complete research workflow for a query.

    Args:
        query: Research question to investigate
        llm_provider: LLM provider ('anthropic', 'openai', 'ollama', 'custom')
        model_name: Model identifier
        search_provider: Search provider ('duckduckgo', 'brave', 'serper')
        num_results: Results per search step (1-10)

    Returns:
        Final state dict containing:
            - plan: Research steps taken
            - results: Search results found
            - final: Final brief with citations
            - references: List of source URLs
            - messages: Agent messages

    Raises:
        ValueError: If configuration is invalid
        Exception: If workflow execution fails

    Example:
        >>> result = run_research(
        ...     "impact of quantum computing",
        ...     llm_provider="ollama",
        ...     model_name="llama3.1:70b"
        ... )
        >>> print(result["final"])
    """
    # Build graph
    graph = build_workflow()

    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "plan": [],
        "results": [],
        "draft": "",
        "final": "",
        "references": [],
        "revised_once": False,
        "llm_provider": llm_provider,
        "model_name": model_name,
        "search_provider": search_provider,
        "num_results": max(1, min(num_results, 10)),
    }

    # Execute workflow
    config = {"configurable": {"thread_id": f"research_{hash(query)}"}}

    final_state = None
    for state_update in graph.stream(initial_state, config):
        final_state = list(state_update.values())[0]

    if not final_state:
        raise RuntimeError("Workflow produced no output")

    return final_state
