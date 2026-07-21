"""Tests for src.llm.registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from src.config import ModelConfig
from src.llm import registry
from src.llm.backends.codex import CodexResponsesBackend
from src.llm.codex_oauth import CodexOAuthCredentials
from src.llm.registry import _default_headers_for  # pyright: ignore[reportPrivateUsage]
from src.llm.types import ProviderClient


def test_default_headers_for_openrouter_base_url() -> None:
    """OpenRouter base URLs get the app-attribution headers."""
    headers = _default_headers_for("https://openrouter.ai/api/v1")
    assert headers["HTTP-Referer"] == "https://honcho.dev"
    assert headers["X-Openrouter-Title"] == "Honcho"


def test_default_headers_for_non_openrouter_base_url() -> None:
    """Other OpenAI-compatible providers get no extra headers."""
    assert _default_headers_for("https://api.openai.com/v1") == {}


def test_default_headers_for_none_base_url() -> None:
    """A missing base URL (default OpenAI) gets no extra headers."""
    assert _default_headers_for(None) == {}

def test_codex_oauth_client_bypasses_default_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_client = object()
    codex_client = object()

    def fake_resolve(**kwargs: Any) -> CodexOAuthCredentials:
        assert kwargs["auth_path"] == "/tmp/auth.json"
        return CodexOAuthCredentials(
            access_token="oauth-access-token",
            base_url="https://chatgpt.com/backend-api/codex",
            default_headers={"originator": "codex_cli_rs"},
            auth_path=Path("/tmp/auth.json"),
        )

    def fake_client(
        base_url: str,
        api_key: str,
        default_headers: tuple[tuple[str, str], ...],
    ) -> object:
        assert base_url == "https://chatgpt.com/backend-api/codex"
        assert api_key == "oauth-access-token"
        assert default_headers == (("originator", "codex_cli_rs"),)
        return codex_client

    monkeypatch.setattr(registry, "resolve_codex_oauth_credentials", fake_resolve)
    monkeypatch.setattr(registry, "get_codex_oauth_client", fake_client)
    monkeypatch.setitem(registry.CLIENTS, "openai", default_client)

    client = registry.client_for_model_config(
        "openai",
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
            codex_auth_path="/tmp/auth.json",
        ),
    )

    assert client is codex_client


@pytest.mark.asyncio
async def test_async_codex_oauth_client_resolves_credentials_off_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_client = object()
    calls: list[dict[str, Any]] = []

    def fake_resolve(**kwargs: Any) -> CodexOAuthCredentials:
        calls.append(kwargs)
        return CodexOAuthCredentials(
            access_token="oauth-access-token",
            base_url="https://chatgpt.com/backend-api/codex",
            default_headers={"originator": "codex_cli_rs"},
            auth_path=Path("/tmp/auth.json"),
        )

    def fake_client(
        base_url: str,
        api_key: str,
        default_headers: tuple[tuple[str, str], ...],
    ) -> object:
        assert base_url == "https://chatgpt.com/backend-api/codex"
        assert api_key == "oauth-access-token"
        assert default_headers == (("originator", "codex_cli_rs"),)
        return codex_client

    monkeypatch.setattr(registry, "resolve_codex_oauth_credentials", fake_resolve)
    monkeypatch.setattr(registry, "get_codex_oauth_client", fake_client)

    client = await registry.aclient_for_model_config(
        "openai",
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
            codex_auth_path="/tmp/auth.json",
        ),
    )

    assert client is codex_client
    assert calls[0]["auth_path"] == "/tmp/auth.json"


def test_codex_oauth_uses_responses_backend() -> None:
    backend = registry.backend_for_provider(
        "openai",
        cast(ProviderClient, object()),
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
        ),
    )

    assert isinstance(backend, CodexResponsesBackend)
