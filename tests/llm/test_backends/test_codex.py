from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from src.llm.backend import StreamChunk
from src.llm.backends.codex import (
    DEFAULT_CODEX_INSTRUCTIONS,
    CodexResponsesBackend,
)


class StructuredCodexResponse(BaseModel):
    answer: str


class FakeResponsesStream:
    def __init__(
        self,
        response: SimpleNamespace,
        events: list[SimpleNamespace] | None = None,
        final_exception: Exception | None = None,
        iteration_exception: Exception | None = None,
    ) -> None:
        self.response: SimpleNamespace = response
        self.events: list[SimpleNamespace] = list(events or [])
        self.final_exception: Exception | None = final_exception
        self.iteration_exception: Exception | None = iteration_exception

    async def __aenter__(self) -> FakeResponsesStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> FakeResponsesStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if not self.events:
            if self.iteration_exception is not None:
                exc = self.iteration_exception
                self.iteration_exception = None
                raise exc
            raise StopAsyncIteration
        return self.events.pop(0)

    async def get_final_response(self) -> SimpleNamespace:
        if self.final_exception is not None:
            raise self.final_exception
        return self.response


@pytest.mark.asyncio
async def test_codex_backend_uses_responses_stream_and_normalizes_text() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(
                status="completed",
                output_text="Hello from Codex",
                output=[],
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    input_tokens_details=SimpleNamespace(cached_tokens=4),
                ),
            )
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=100,
        temperature=0.2,
        thinking_effort="minimal",
    )

    assert result.content == "Hello from Codex"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cache_read_input_tokens == 4

    call = client.responses.stream.call_args.kwargs
    assert call["model"] == "gpt-5.5"
    assert call["instructions"] == "Be terse."
    assert call["input"] == [{"role": "user", "content": "Hello"}]
    assert "max_output_tokens" not in call
    assert "temperature" not in call
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.asyncio
async def test_codex_backend_uses_stream_text_when_final_response_is_empty() -> None:
    client = Mock()
    final_response = SimpleNamespace(
        status="completed",
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            final_response,
            events=[
                SimpleNamespace(type="response.output_text.delta", delta="HONCHO"),
                SimpleNamespace(type="response.output_text.delta", delta="_OK"),
                SimpleNamespace(
                    type="response.completed",
                    response=final_response,
                ),
            ],
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Reply exactly HONCHO_OK"}],
        max_tokens=100,
    )

    assert result.content == "HONCHO_OK"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.cache_read_input_tokens == 3
    assert result.raw_response == {"codex_stream_fallback": True}



@pytest.mark.asyncio
async def test_codex_backend_uses_stream_text_for_structured_output_when_final_response_is_empty() -> None:
    client = Mock()
    final_response = SimpleNamespace(
        status="completed",
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            input_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
    )
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            final_response,
            events=[
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta='{\"answer\":\"HONCHO_OK\"}',
                ),
                SimpleNamespace(type="response.completed", response=final_response),
            ],
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Reply with structured JSON"}],
        max_tokens=100,
        response_format=StructuredCodexResponse,
    )

    assert isinstance(result.content, StructuredCodexResponse)
    assert result.content.answer == "HONCHO_OK"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.cache_read_input_tokens == 3
    assert result.raw_response == {"codex_stream_fallback": True}

@pytest.mark.asyncio
async def test_codex_backend_falls_back_to_collected_events_on_null_output_parse_error() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(status="completed", output_text="", output=[], usage=None),
            events=[
                SimpleNamespace(type="response.output_text.delta", delta="Hello"),
                SimpleNamespace(type="response.output_text.delta", delta=" from fallback"),
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="function_call",
                        call_id="call_weather",
                        name="get_weather",
                        arguments='{"city":"Miami"}',
                    ),
                ),
            ],
            iteration_exception=TypeError("'NoneType' object is not iterable"),
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    )

    assert result.content == "Hello from fallback"
    assert result.tool_calls[0].id == "call_weather"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "Miami"}
    assert result.raw_response == {"codex_stream_fallback": True}


@pytest.mark.asyncio
async def test_codex_backend_reraises_null_output_parse_error_without_stream_parts() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(status="completed", output_text="", output=[], usage=None),
            iteration_exception=TypeError("'NoneType' object is not iterable"),
        )
    )

    backend = CodexResponsesBackend(client)
    with pytest.raises(TypeError, match="NoneType"):
        await backend.complete(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )


@pytest.mark.asyncio
async def test_codex_backend_final_empty_response_uses_streamed_tool_items() -> None:
    client = Mock()
    final_response = SimpleNamespace(status="completed", output_text="", output=[], usage=None)
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            final_response,
            events=[
                SimpleNamespace(
                    type="response.output_item.done",
                    item=SimpleNamespace(
                        type="function_call",
                        call_id="call_weather",
                        name="get_weather",
                        arguments='{"city":"Miami"}',
                    ),
                ),
                SimpleNamespace(type="response.completed", response=final_response),
            ],
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Call a tool"}],
        max_tokens=100,
    )

    assert result.content == ""
    assert result.tool_calls[0].id == "call_weather"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "Miami"}
    assert result.raw_response == {"codex_stream_fallback": True}


@pytest.mark.asyncio
async def test_codex_backend_stream_reraises_null_output_parse_error_without_signal() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(status="completed", output_text="", output=[], usage=None),
            iteration_exception=TypeError("'NoneType' object is not iterable"),
        )
    )

    backend = CodexResponsesBackend(client)
    result: AsyncIterator[StreamChunk] = backend.stream(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    with pytest.raises(TypeError, match="NoneType"):
        _ = [chunk async for chunk in result]


@pytest.mark.asyncio
async def test_codex_backend_streams_and_normalizes_text() -> None:
    client = Mock()
    final_response = SimpleNamespace(
        status="completed",
        output_text="Hello from Codex",
        output=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
        ),
    )
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            final_response,
            events=[
                SimpleNamespace(type="response.output_text.delta", delta="Hello"),
                SimpleNamespace(
                    type="response.completed",
                    response=final_response,
                ),
            ],
        )
    )

    backend = CodexResponsesBackend(client)
    result: AsyncIterator[StreamChunk] = backend.stream(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=100,
        thinking_effort="minimal",
    )
    chunks: list[StreamChunk] = [chunk async for chunk in result]

    assert chunks[0].content == "Hello"
    assert chunks[-1].is_done is True
    assert chunks[-1].output_tokens == 5

    call = client.responses.stream.call_args.kwargs
    assert call["model"] == "gpt-5.5"
    assert call["instructions"] == "Be terse."
    assert call["input"] == [{"role": "user", "content": "Hello"}]
    assert "max_output_tokens" not in call
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.asyncio
async def test_codex_backend_logs_final_usage_openai_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(status="completed", output_text="", output=[], usage=None),
            final_exception=OpenAIError("final response unavailable"),
        )
    )

    backend = CodexResponsesBackend(client)
    with caplog.at_level(logging.DEBUG, logger="src.llm.backends.codex"):
        result: AsyncIterator[StreamChunk] = backend.stream(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        chunks: list[StreamChunk] = [chunk async for chunk in result]

    assert chunks[-1].is_done is True
    assert chunks[-1].output_tokens is None
    assert "OpenAI error" in caplog.text
    assert "final response unavailable" in caplog.text


@pytest.mark.asyncio
async def test_codex_backend_converts_tools_and_tool_results() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(
                status="completed",
                output_text="",
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        id="rs_123",
                        summary=[{"text": "checking weather"}],
                        status="completed",
                        model_dump=lambda: {
                            "type": "reasoning",
                            "id": "rs_123",
                            "summary": [{"text": "checking weather"}],
                            "status": "completed",
                        },
                    ),
                    SimpleNamespace(
                        type="function_call",
                        call_id="call_weather",
                        name="get_weather",
                        arguments='{"city": "Miami"}',
                    )
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "reasoning_details": [
                    {
                        "type": "reasoning",
                        "id": "rs_prev",
                        "status": "completed",
                        "summary": [{"text": "checking weather"}],
                    }
                ],
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Miami"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "sunny",
            },
        ],
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
        tool_choice="any",
    )

    assert result.tool_calls[0].id == "call_weather"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].input == {"city": "Miami"}
    assert result.reasoning_details == [
        {
            "type": "reasoning",
            "summary": [{"text": "checking weather"}],
        }
    ]

    call = client.responses.stream.call_args.kwargs
    assert call["input"] == [
        {"type": "reasoning", "summary": [{"text": "checking weather"}]},
        {
            "type": "function_call",
            "call_id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city": "Miami"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_weather",
            "output": "sunny",
        },
    ]
    assert call["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "strict": False,
        }
    ]
    assert call["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_codex_backend_uses_responses_json_schema_format() -> None:
    client = Mock()
    client.responses.stream = Mock(
        return_value=FakeResponsesStream(
            SimpleNamespace(
                status="completed",
                output_text='{"answer": "ok"}',
                output=[],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )
        )
    )

    backend = CodexResponsesBackend(client)
    result = await backend.complete(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "Return JSON"}],
        max_tokens=100,
        response_format=StructuredCodexResponse,
    )

    assert isinstance(result.content, StructuredCodexResponse)
    assert result.content.answer == "ok"

    text_config = client.responses.stream.call_args.kwargs["text"]
    assert client.responses.stream.call_args.kwargs["instructions"] == (
        DEFAULT_CODEX_INSTRUCTIONS
    )
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["name"] == "StructuredCodexResponse"
