#!/usr/bin/env python

"""Tests for `fastmlx` package."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import the actual classes and functions
from fastmlx import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelProvider,
    Usage,
    app,
    handle_function_calls,
)


# Create mock classes that inherit from the original classes
class MockModelProvider(ModelProvider):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_model(self, model_name: str):
        if model_name not in self.models:
            model_type = "vlm" if "llava" in model_name.lower() else "lm"
            self.models[model_name] = {
                "model": MagicMock(),
                "processor": MagicMock(),
                "tokenizer": MagicMock(),
                "image_processor": MagicMock() if model_type == "vlm" else None,
                "config": {"model_type": model_type},
            }
        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False

    async def get_available_models(self):
        return list(self.models.keys())


# Mock MODELS dictionary
MODELS = {"vlm": ["llava"], "lm": ["phi"]}


# Mock functions
def mock_generate(*args, **kwargs):
    return "generated response", {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def mock_vlm_stream_generate(*args, **kwargs):
    yield "Hello"
    yield " world"
    yield "!"


def mock_lm_stream_generate(*args, **kwargs):
    yield "Testing"
    yield " stream"
    yield " generation"


@pytest.fixture(scope="module")
def client():
    # Apply patches
    with patch("fastmlx.fastmlx.model_provider", MockModelProvider()), patch(
        "fastmlx.fastmlx.vlm_generate", mock_generate
    ), patch("fastmlx.fastmlx.lm_generate", mock_generate), patch(
        "fastmlx.fastmlx.MODELS", MODELS
    ), patch(
        "fastmlx.utils.vlm_stream_generate", mock_vlm_stream_generate
    ), patch(
        "fastmlx.utils.lm_stream_generate", mock_lm_stream_generate
    ):
        yield TestClient(app)


def test_chat_completion_vlm(client):
    request = ChatCompletionRequest(
        model="test_llava_model",
        messages=[ChatMessage(role="user", content="Hello")],
        image="test_image",
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]
    assert "usage" in response.json()
    usage = response.json()["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_chat_completion_lm(client):
    request = ChatCompletionRequest(
        model="test_phi_model", messages=[ChatMessage(role="user", content="Hello")]
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]
    assert "usage" in response.json()
    usage = response.json()["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


@pytest.mark.asyncio
async def test_vlm_streaming(client):

    # Mock the vlm_stream_generate function
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test_llava_model",
            "messages": [{"role": "user", "content": "Describe this image"}],
            "image": "base64_encoded_image_data",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = list(response.iter_lines())
    assert len(chunks) == 8  # 7 content chunks + [DONE]
    for chunk in chunks[:-2]:  # Exclude the [DONE] message
        if chunk:
            chunk = chunk.split("data: ")[1]
            data = json.loads(chunk)
            assert "id" in data
            assert data["object"] == "chat.completion.chunk"
            assert "created" in data
            assert data["model"] == "test_llava_model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["index"] == 0
            assert "delta" in data["choices"][0]
            assert "role" in data["choices"][0]["delta"]
            assert "content" in data["choices"][0]["delta"]
            if "usage" in data:
                usage = data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage

    assert chunks[-2] == "data: [DONE]"


@pytest.mark.asyncio
async def test_lm_streaming(client):

    # Mock the lm_stream_generate function
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test_phi_model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    chunks = list(response.iter_lines())
    assert len(chunks) == 8  # 7 content chunks + [DONE]

    for chunk in chunks[:-2]:  # Exclude the [DONE] message
        if chunk:
            chunk = chunk.split("data: ")[1]
            data = json.loads(chunk)
            assert "id" in data
            assert data["object"] == "chat.completion.chunk"
            assert "created" in data
            assert data["model"] == "test_phi_model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["index"] == 0
            assert "delta" in data["choices"][0]
            assert "role" in data["choices"][0]["delta"]
            assert "content" in data["choices"][0]["delta"]
            if "usage" in data:
                usage = data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage

    assert chunks[-2] == "data: [DONE]"


def test_get_supported_models(client):
    response = client.get("/v1/supported_models")
    assert response.status_code == 200
    data = response.json()
    assert "vlm" in data
    assert "lm" in data
    assert data["vlm"] == ["llava"]
    assert data["lm"] == ["phi"]


def test_list_models(client):
    client.post("/v1/models?model_name=test_llava_model")
    client.post("/v1/models?model_name=test_phi_model")

    response = client.get("/v1/models")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["object"] == "list"
    model_ids = {model["id"] for model in response_data["data"]}
    assert model_ids == {"test_llava_model", "test_phi_model"}


def test_add_model(client):
    response = client.post("/v1/models?model_name=new_llava_model")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model new_llava_model added successfully",
    }


def test_remove_model(client):
    # Add a model
    response = client.post("/v1/models?model_name=test_model")
    assert response.status_code == 200

    # Verify the model is added
    response = client.get("/v1/models")
    response_data = response.json()
    model_ids = {model["id"] for model in response_data["data"]}
    assert "test_model" in model_ids

    # Remove the model
    response = client.delete("/v1/models?model_name=test_model")
    assert response.status_code == 204

    # Verify the model is removed
    response = client.get("/v1/models")
    response_data = response.json()
    model_ids = {model["id"] for model in response_data["data"]}
    assert "test_model" not in model_ids

    # Try to remove a non-existent model
    response = client.delete("/v1/models?model_name=non_existent_model")
    assert response.status_code == 404
    assert "Model 'non_existent_model' not found" in response.json()["detail"]


def test_handle_function_calls_json_format():
    output = """Here's the weather forecast:
    {"tool_calls": [{"name": "get_weather", "arguments": {"location": "New York", "date": "2023-08-15"}}]}
    """
    request = MagicMock()
    request.model = "test_model"
    token_info = MagicMock()
    token_info.prompt_tokens = 10
    token_info.completion_tokens = 20
    token_info.total_tokens = 30
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, request, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "New York",
        "date": "2023-08-15",
    }
    assert "Here's the weather forecast:" in response.choices[0]["message"]["content"]
    assert '{"tool_calls":' not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_xml_format_old():
    output = """Let me check that for you.
    <function_calls>
    <function=get_stock_price>{"symbol": "AAPL"}</function>
    </function_calls>
    """
    request = MagicMock()
    request.model = "test_model"
    token_info = MagicMock()
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, request, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_stock_price"
    assert json.loads(response.tool_calls[0].function.arguments) == {"symbol": "AAPL"}
    assert "Let me check that for you." in response.choices[0]["message"]["content"]
    assert "<function_calls>" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_xml_format_new():
    output = """I'll get that information for you.
    <function_calls>
    <invoke>
    <tool_name>search_database</tool_name>
    <query>latest smartphones</query>
    <limit>5</limit>
    </invoke>
    </function_calls>
    """
    request = MagicMock()
    request.model = "test_model"
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, request, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "search_database"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "query": "latest smartphones",
        "limit": "5",
    }
    assert (
        "I'll get that information for you."
        in response.choices[0]["message"]["content"]
    )
    assert "<function_calls>" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


def test_handle_function_calls_functools_format():
    output = """Here are the results:
    functools[{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}]
    """
    request = MagicMock()
    request.model = "test_model"
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, request, token_info)

    assert isinstance(response, ChatCompletionResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "San Francisco, CA",
        "format": "fahrenheit",
    }
    assert "Here are the results:" in response.choices[0]["message"]["content"]
    assert "functools[" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


# Add a new test for multiple function calls in functools format
def test_handle_function_calls_multiple_functools():
    output = """Here are the results:
    functools[{"name": "get_weather", "arguments": {"location": "New York"}}, {"name": "get_time", "arguments": {"timezone": "EST"}}]
    """
    request = MagicMock()
    request.model = "test_model"
    token_info = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    response = handle_function_calls(output, request, token_info)
    assert isinstance(response, ChatCompletionResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.tool_calls[0].function.arguments) == {
        "location": "New York"
    }
    assert response.tool_calls[1].function.name == "get_time"
    assert json.loads(response.tool_calls[1].function.arguments) == {"timezone": "EST"}
    assert "Here are the results:" in response.choices[0]["message"]["content"]
    assert "functools[" not in response.choices[0]["message"]["content"]
    assert response.usage
    usage = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30


if __name__ == "__main__":
    pytest.main(["-v", __file__])
