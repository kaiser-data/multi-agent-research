#!/usr/bin/env python3
"""Gradio web interface for multi-agent research system."""

import os
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from providers.llm import get_available_models
from providers.search import get_available_providers
from workflow import run_research

# Load environment variables
load_dotenv()


def format_output(state: dict) -> tuple[str, str, str]:
    """
    Format research results for display.

    Args:
        state: Final workflow state

    Returns:
        Tuple of (plan_md, brief_md, references_md)
    """
    # Format research plan
    plan = state.get("plan", [])
    plan_md = "## 📋 Research Plan\n\n"
    if plan:
        for i, step in enumerate(plan, 1):
            plan_md += f"{i}. {step}\n"
    else:
        plan_md += "_No plan generated_\n"

    # Format brief
    brief = state.get("final", state.get("draft", ""))
    brief_md = "## 📝 Research Brief\n\n"
    if brief:
        brief_md += brief
    else:
        brief_md += "_No brief generated_\n"

    # Format references
    references = state.get("references", [])
    references_md = "## 📚 References\n\n"
    if references:
        for i, url in enumerate(references, 1):
            references_md += f"[{i}] {url}\n\n"
    else:
        references_md += "_No references found_\n"

    return plan_md, brief_md, references_md


def research_interface(
    query: str,
    llm_provider: str,
    model_name: str,
    search_provider: str,
    num_results: int,
) -> tuple[str, str, str, str]:
    """
    Execute research and return formatted results.

    Args:
        query: Research question
        llm_provider: Selected LLM provider
        model_name: Selected model
        search_provider: Selected search provider
        num_results: Results per step

    Returns:
        Tuple of (status, plan, brief, references)
    """
    if not query.strip():
        return (
            "❌ Please enter a research query",
            "",
            "",
            "",
        )

    try:
        # Execute research workflow
        state = run_research(
            query=query,
            llm_provider=llm_provider.lower(),
            model_name=model_name,
            search_provider=search_provider.lower(),
            num_results=num_results,
        )

        # Format output
        plan_md, brief_md, references_md = format_output(state)

        status = f"✅ Research completed successfully for: *{query}*"

        return status, plan_md, brief_md, references_md

    except ValueError as e:
        # Configuration errors
        error_msg = f"❌ Configuration Error: {str(e)}"
        return error_msg, "", "", ""

    except Exception as e:
        # Unexpected errors
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", "", ""


def update_model_choices(provider: str) -> gr.Dropdown:
    """Update available models based on selected provider."""
    models = get_available_models(provider.lower())
    return gr.Dropdown(choices=models, value=models[0] if models else None)


def create_interface() -> gr.Blocks:
    """
    Create Gradio interface for multi-agent research.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Multi-Agent Research System") as interface:
        gr.Markdown(
            """
            # 🔍 Multi-Agent Research System

            Enter a research question and the system will:
            1. **Plan** - Break down your query into research steps
            2. **Research** - Search for relevant information
            3. **Write** - Synthesize findings into a brief with citations
            4. **Review** - Validate citations and quality
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Query input
                query_input = gr.Textbox(
                    label="Research Query",
                    placeholder="e.g., What is the impact of quantum computing on cryptography?",
                    lines=3,
                )

                # Example queries
                gr.Examples(
                    examples=[
                        "What are the latest developments in AI safety research?",
                        "How is quantum computing affecting cybersecurity?",
                        "What are the most promising renewable energy technologies?",
                        "What is the current state of fusion energy research?",
                        "How are large language models changing software development?",
                    ],
                    inputs=query_input,
                )

            with gr.Column(scale=1):
                # LLM provider selection
                llm_provider = gr.Dropdown(
                    choices=["Anthropic", "OpenAI", "Ollama", "Custom"],
                    value="Anthropic",
                    label="LLM Provider",
                    info="Select your language model provider",
                )

                # Model selection (dynamic based on provider)
                model_name = gr.Dropdown(
                    choices=get_available_models("anthropic"),
                    value="claude-sonnet-4-5-20250929",
                    label="Model",
                    info="Select specific model",
                )

                # Update models when provider changes
                llm_provider.change(
                    fn=update_model_choices,
                    inputs=[llm_provider],
                    outputs=[model_name],
                )

                # Search provider selection
                providers_info = get_available_providers()
                search_choices = [p["display_name"] for p in providers_info]
                search_values = [p["name"] for p in providers_info]

                search_provider = gr.Dropdown(
                    choices=search_choices,
                    value=search_choices[0],  # DuckDuckGo by default
                    label="Search Provider",
                    info="Free options available",
                )

                # Number of results
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Results per Step",
                    info="Number of search results per research step",
                )

                # Submit button
                submit_btn = gr.Button("🚀 Start Research", variant="primary", size="lg")

        # Status message
        status_output = gr.Markdown(label="Status")

        # Output sections
        with gr.Row():
            plan_output = gr.Markdown(label="Research Plan")

        with gr.Row():
            brief_output = gr.Markdown(label="Research Brief")

        with gr.Row():
            references_output = gr.Markdown(label="References")

        # Wire up the interface
        submit_btn.click(
            fn=research_interface,
            inputs=[
                query_input,
                llm_provider,
                model_name,
                search_provider,
                num_results,
            ],
            outputs=[
                status_output,
                plan_output,
                brief_output,
                references_output,
            ],
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Note:** Ensure required API keys are set in your `.env` file.
            See README for setup instructions.
            """
        )

    return interface


def main():
    """Launch Gradio interface."""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
