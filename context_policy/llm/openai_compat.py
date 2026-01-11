"""OpenAI-compatible HTTP client for model inference."""
from __future__ import annotations

import os
import time

import requests


def get_base_url() -> str:
    """Get the OpenAI-compatible API base URL from environment."""
    return os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")


def get_api_key() -> str:
    """Get the API key from environment."""
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    timeout_s: int = 120,
) -> str:
    """Call OpenAI-compatible chat completion endpoint.

    Args:
        model: Model name/path.
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (default 0.0 for deterministic).
        top_p: Top-p sampling parameter (default 1.0).
        max_tokens: Maximum tokens to generate.
        timeout_s: Request timeout in seconds.

    Returns:
        The assistant's response content string.

    Raises:
        RuntimeError: If request fails after retries.
    """
    base_url = get_base_url().rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(0.5)

    raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_error}")
