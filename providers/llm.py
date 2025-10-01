"""LLM provider abstraction layer supporting multiple backends."""

import os
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0,
):
    """
    Factory function to create LLM instance based on provider.

    Args:
        provider: LLM provider ('anthropic', 'openai', 'ollama', 'custom')
                 Defaults to LLM_PROVIDER env var or 'anthropic'
        model_name: Model identifier. Defaults to MODEL_NAME env var
        temperature: Sampling temperature (0-1)

    Returns:
        Configured LLM instance (ChatAnthropic or ChatOpenAI)

    Raises:
        ValueError: If required API keys are missing

    Examples:
        >>> llm = get_llm("anthropic", "claude-sonnet-4-5-20250929")
        >>> llm = get_llm("ollama", "llama3.1:70b")
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "anthropic")

    if model_name is None:
        model_name = os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929")

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required for provider 'anthropic'. "
                "Get your API key at https://console.anthropic.com"
            )

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_retries=3,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY required for provider 'openai'. "
                "Get your API key at https://platform.openai.com"
            )

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            max_retries=3,
        )

    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key="not-needed",  # Ollama doesn't require API key
            openai_api_base=base_url,
            max_retries=3,
        )

    elif provider == "custom":
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL required for provider 'custom'. "
                "Set it to your self-hosted endpoint (e.g., http://your-server:8000/v1)"
            )

        api_key = os.getenv("OPENAI_API_KEY", "not-needed")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_retries=3,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: anthropic, openai, ollama, custom"
        )


def get_available_models(provider: str) -> list[str]:
    """
    Get list of available models for a provider.

    Args:
        provider: LLM provider name

    Returns:
        List of model identifiers
    """
    models = {
        "anthropic": [
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
        "ollama": [
            "llama3.1:70b",
            "llama3.1:8b",
            "mistral:latest",
            "mixtral:latest",
            "phi3:medium",
            "gemma2:27b",
        ],
        "custom": [
            "custom-model",  # Placeholder for user-defined models
        ],
    }

    return models.get(provider, [])
