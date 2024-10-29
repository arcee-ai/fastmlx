"""Main module for FastMLX API server.

This module provides a FastAPI-based server for hosting MLX models,
including Vision Language Models (VLMs) and Language Models (LMs).
It offers an OpenAI-compatible API for chat completions and model management.
"""

import argparse
import asyncio
import os
from typing import Any, Dict, List
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .types.chat.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from .types.model import SupportedModels

try:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as apply_vlm_chat_template
    from mlx_vlm.utils import load_config

    from .utils import (
        MODEL_REMAPPING,
        MODELS,
        apply_lm_chat_template,
        get_eom_token,
        get_tool_prompt,
        handle_function_calls,
        lm_generate,
        lm_stream_generator,
        load_lm_model,
        load_vlm_model,
        vlm_stream_generator,
    )

    MLX_AVAILABLE = True
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")
    MLX_AVAILABLE = False


class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    def load_model(self, model_name: str):
        if model_name not in self.models:
            config = load_config(model_name)
            model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
            if model_type in MODELS["vlm"]:
                self.models[model_name] = load_vlm_model(model_name, config)
            else:
                self.models[model_name] = load_lm_model(model_name, config)

        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                return True
            return False

    async def get_available_models(self):
        async with self.lock:
            return list(self.models.keys())


app = FastAPI()


def int_or_float(value):

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    if num_workers_env := os.getenv("FASTMLX_NUM_WORKERS"):
        try:
            workers = int(num_workers_env)
        except ValueError:
            workers = max(1, int(os.cpu_count() * float(num_workers_env)))
    return workers


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Initialize the ModelProvider
model_provider = ModelProvider()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    Handle chat completion requests for both VLM and LM models.

    Args:
        request (ChatCompletionRequest): The chat completion request.

    Returns:
        ChatCompletionResponse (ChatCompletionResponse): The generated chat completion response.

    Raises:
        HTTPException (str): If MLX library is not available.
    """
    if not MLX_AVAILABLE:
        raise HTTPException(status_code=500, detail="MLX library not available")

    stream = request.stream
    model_data = model_provider.load_model(request.model)
    model = model_data["model"]
    config = model_data["config"]
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
    stop_words = get_eom_token(request.model)

    if model_type in MODELS["vlm"]:
        processor = model_data["processor"]
        image_processor = model_data["image_processor"]

        image_url = None
        chat_messages = []

        for msg in request.messages:
            if isinstance(msg.content, str):
                chat_messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                text_content = ""
                for content_part in msg.content:
                    if content_part.type == "text":
                        text_content += content_part.text + " "
                    elif content_part.type == "image_url":
                        image_url = content_part.image_url["url"]
                chat_messages.append(
                    {"role": msg.role, "content": text_content.strip()}
                )

        if not image_url and model_type in MODELS["vlm"]:
            raise HTTPException(
                status_code=400, detail="Image URL not provided for VLM model"
            )

        prompt = ""
        if model.config.model_type != "paligemma":
            prompt = apply_vlm_chat_template(processor, config, chat_messages)
        else:
            prompt = chat_messages[-1]["content"]

        if stream:
            return StreamingResponse(
                vlm_stream_generator(
                    model,
                    request.model,
                    processor,
                    image_url,
                    prompt,
                    image_processor,
                    request.max_tokens,
                    request.temperature,
                    stream_options=request.stream_options,
                ),
                media_type="text/event-stream",
            )
        else:
            # Generate the response
            output = vlm_generate(
                model,
                processor,
                image_url,
                prompt,
                image_processor,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                verbose=False,
            )

    else:
        # Add function calling information to the prompt
        if request.tools and "firefunction-v2" not in request.model:
            # Handle system prompt
            if request.messages and request.messages[0].role == "system":
                pass
            else:
                # Generate system prompt based on model and tools
                prompt, user_role = get_tool_prompt(
                    request.model,
                    [tool.model_dump() for tool in request.tools],
                    request.messages[-1].content,
                )

                if user_role:
                    request.messages[-1].content = prompt
                else:
                    # Insert the system prompt at the beginning of the messages
                    request.messages.insert(
                        0, ChatMessage(role="system", content=prompt)
                    )

        tokenizer = model_data["tokenizer"]

        chat_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        prompt = apply_lm_chat_template(tokenizer, chat_messages, request)

        if stream:
            return StreamingResponse(
                lm_stream_generator(
                    model,
                    request.model,
                    tokenizer,
                    prompt,
                    request.max_tokens,
                    request.temperature,
                    stop_words=stop_words,
                    stream_options=request.stream_options,
                ),
                media_type="text/event-stream",
            )
        else:
            output, token_length_info = lm_generate(
                model,
                tokenizer,
                prompt,
                request.max_tokens,
                temp=request.temperature,
                stop_words=stop_words,
            )

    # Parse the output to check for function calls
    return handle_function_calls(output, request, token_length_info)


@app.get("/v1/supported_models", response_model=SupportedModels)
async def get_supported_models():
    """
    Get a list of supported model types for VLM and LM.

    Returns:
        JSONResponse (json): A JSON response containing the supported models.
    """
    return JSONResponse(content=MODELS)


@app.get("/v1/models")
async def list_models():
    """
    List all available (loaded) models.

    Returns:
        dict (dict): A dictionary containing the list of available models.
    """
    return {"models": await model_provider.get_available_models()}


@app.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.

    Args:
        model_name (str): The name of the model to add.

    Returns:
        dict (dict): A dictionary containing the status of the operation.
    """
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
    """
    Remove a model from the API.

    Args:
        model_name (str): The name of the model to remove.

    Returns:
        Response (str): A 204 No Content response if successful.

    Raises:
        HTTPException (str): If the model is not found.
    """
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


def run():
    parser = argparse.ArgumentParser(description="FastMLX API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server. Only works when 'workers' is set to None.",
    )

    parser.add_argument(
        "--workers",
        type=int_or_float,
        default=calculate_default_workers(),
        help="""Number of workers. Overrides the `FASTMLX_NUM_WORKERS` env variable.
        Can be either an int or a float.
        If an int, it will be the number of workers to use.
        If a float, number of workers will be this fraction of the  number of CPU cores available, with a minimum of 1.
        Defaults to the `FASTMLX_NUM_WORKERS` env variable if set and to 2 if not.
        To use all available CPU cores, set it to 1.0.

        Examples:
        --workers 1 (will use 1 worker)
        --workers 1.0 (will use all available CPU cores)
        --workers 0.5 (will use half the number of CPU cores available)
        --workers 0.0 (will use 1 worker)""",
    )

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    setup_cors(app, args.allowed_origins)

    import uvicorn

    uvicorn.run(
        "fastmlx:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        loop="asyncio",
    )


if __name__ == "__main__":
    run()
