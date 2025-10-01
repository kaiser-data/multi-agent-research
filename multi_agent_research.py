#!/usr/bin/env python3
"""
Multi-Agent Research Script using LangGraph + LangChain + Claude + SerpAPI

PURPOSE:
    Orchestrates a multi-agent workflow to research any query:
    1. Planner decomposes the query into research steps
    2. Researcher executes Google searches for each step
    3. Writer synthesizes findings into a brief with citations
    4. Reviewer validates citations and triggers one revision if needed

INSTALL:
    pip install 'langchain>=0.3.27,<0.4' 'langgraph>=0.6.8,<0.7' \
                'langchain-anthropic>=0.3.21,<0.4' python-dotenv \
                google-search-results requests

.ENV EXAMPLE:
    ANTHROPIC_API_KEY=sk-ant-...
    SERPAPI_API_KEY=your_serpapi_key
    CLAUDE_MODEL=claude-sonnet-4-5-20250929  # optional

RUN EXAMPLES:
    python multi_agent_research.py --query "impact of quantum computing on cryptography"
    python multi_agent_research.py --query "latest AI safety research" --results 8
    python multi_agent_research.py --query "climate change solutions" --model claude-3-7-sonnet-20250219
"""

import argparse
import json
import operator
import os
import sys
import time
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from serpapi import GoogleSearch


# ─── Configuration ───────────────────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-sonnet-4-5-20250929"
)

MODEL_FALLBACKS = [
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
]

if not ANTHROPIC_API_KEY:
    print("❌ ERROR: ANTHROPIC_API_KEY not found in environment", file=sys.stderr)
    sys.exit(1)
if not SERPAPI_API_KEY:
    print("❌ ERROR: SERPAPI_API_KEY not found in environment", file=sys.stderr)
    sys.exit(1)


# ─── State Definition ────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    query: str
    plan: List[str]
    results: List[dict]
    draft: str
    final: str
    references: List[str]
    revised_once: bool


# ─── SerpAPI Tool ────────────────────────────────────────────────────────────
def google_search(query: str, num_results: int = 5) -> List[dict]:
    """Execute Google search via SerpAPI with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            params = {
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "num": min(num_results, 10),
                "engine": "google",
            }
            search = GoogleSearch(params)
            raw = search.get_dict()
            organic = raw.get("organic_results", [])

            normalized = []
            for item in organic[:num_results]:
                normalized.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })
            return normalized
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️  Search failed after {max_retries} attempts: {e}", file=sys.stderr)
                return []
            time.sleep(2 ** attempt)
    return []


# ─── LLM Initialization ──────────────────────────────────────────────────────
def get_llm(model_name: str) -> ChatAnthropic:
    """Initialize Claude LLM with model selection."""
    selected = model_name if model_name in MODEL_FALLBACKS else MODEL_FALLBACKS[0]
    return ChatAnthropic(model=selected, temperature=0, api_key=ANTHROPIC_API_KEY)


# ─── Agent Nodes ─────────────────────────────────────────────────────────────
def planner_node(state: ResearchState) -> dict:
    """Generate 2-5 research steps from query."""
    llm = get_llm(CLAUDE_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research planner. Given a query, decompose it into 2-5 focused research steps. Return ONLY a JSON array of strings."),
        ("human", "Query: {query}\n\nReturn research steps as JSON array.")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": state["query"]})

    try:
        plan = json.loads(response)
        if not isinstance(plan, list) or len(plan) < 2 or len(plan) > 5:
            plan = [state["query"]]
    except json.JSONDecodeError:
        plan = [state["query"]]

    return {
        "plan": plan,
        "messages": [AIMessage(content=f"Research plan: {json.dumps(plan)}")],
    }


def researcher_node(state: ResearchState) -> dict:
    """Execute searches for each plan step."""
    results = []
    num_results = 5

    for idx, step in enumerate(state["plan"]):
        search_results = google_search(step, num_results)
        for res in search_results:
            res["step_index"] = idx
            results.append(res)
        time.sleep(0.5)

    return {
        "results": results,
        "messages": [AIMessage(content=f"Found {len(results)} total results")],
    }


def writer_node(state: ResearchState) -> dict:
    """Synthesize findings into brief with citations."""
    llm = get_llm(CLAUDE_MODEL)

    results_text = "\n\n".join([
        f"[{i+1}] {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
        for i, r in enumerate(state["results"][:20])
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research writer. Synthesize findings into a ≤200-word brief with inline citations [1], [2], etc. Be factual and concise."),
        ("human", "Query: {query}\n\nResearch Results:\n{results}\n\nWrite brief with citations:")
    ])

    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"query": state["query"], "results": results_text})

    references = []
    seen_urls = set()
    for r in state["results"][:20]:
        if r["url"] and r["url"] not in seen_urls:
            references.append(r["url"])
            seen_urls.add(r["url"])

    return {
        "draft": draft,
        "references": references,
        "messages": [AIMessage(content="Draft complete")],
    }


def reviewer_node(state: ResearchState) -> dict:
    """Validate citations and decide if revision needed."""
    llm = get_llm(CLAUDE_MODEL)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fact-checker. Review if all claims have proper citations [n]. Reply with 'APPROVED' or 'NEEDS_REVISION: <reason>'."),
        ("human", "Draft:\n{draft}\n\nReview:")
    ])

    chain = prompt | llm | StrOutputParser()
    review = chain.invoke({"draft": state["draft"]})

    if "NEEDS_REVISION" in review and not state.get("revised_once", False):
        return {
            "revised_once": True,
            "messages": [AIMessage(content=f"Revision requested: {review}")],
        }

    return {
        "final": state["draft"],
        "messages": [AIMessage(content="Research complete")],
    }


# ─── Routing Logic ───────────────────────────────────────────────────────────
def should_revise(state: ResearchState) -> str:
    """Route to writer for revision or end."""
    if state.get("final"):
        return END
    if state.get("revised_once", False):
        return "writer"
    return "writer" if not state.get("draft") else END


# ─── Graph Construction ──────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """Construct LangGraph workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        lambda s: "writer" if (not s.get("final") and not s.get("revised_once", False)) else END,
        {"writer": "writer", END: END}
    )

    return workflow.compile(checkpointer=MemorySaver())


# ─── CLI & Execution ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-agent research with Claude + SerpAPI")
    parser.add_argument("--query", required=True, help="Research query")
    parser.add_argument("--results", type=int, default=5, help="Results per step (1-8)")
    parser.add_argument("--model", help="Override Claude model")
    args = parser.parse_args()

    if args.model:
        global CLAUDE_MODEL
        CLAUDE_MODEL = args.model

    num_results = max(1, min(args.results, 8))

    print(f"\n{'='*70}")
    print(f"🔍 RESEARCH QUERY: {args.query}")
    print(f"{'='*70}\n")

    graph = build_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.query)],
        "query": args.query,
        "plan": [],
        "results": [],
        "draft": "",
        "final": "",
        "references": [],
        "revised_once": False,
    }

    config = {"configurable": {"thread_id": "research_001"}}

    try:
        final_state = None
        for state in graph.stream(initial_state, config):
            final_state = list(state.values())[0]

        if not final_state:
            print("❌ ERROR: Workflow produced no output", file=sys.stderr)
            sys.exit(1)

        print("📋 RESEARCH PLAN:")
        for i, step in enumerate(final_state.get("plan", []), 1):
            print(f"  {i}. {step}")
        print()

        print("🔗 TOP RESULTS:")
        for i, res in enumerate(final_state.get("results", [])[:10], 1):
            print(f"  [{i}] {res['title']}")
            print(f"      {res['url']}")
        print()

        print("📝 FINAL BRIEF:")
        print(final_state.get("final", final_state.get("draft", "No output generated")))
        print()

        print("📚 REFERENCES:")
        for i, url in enumerate(final_state.get("references", []), 1):
            print(f"  [{i}] {url}")
        print()

    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
