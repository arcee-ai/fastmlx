"""Main module for FastMLX API server.

This module provides a FastAPI-based server for hosting MLX models,
including Vision Language Models (VLMs) and Language Models (LMs).
It offers an OpenAI-compatible API for chat completions and model management.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Callable
from urllib.parse import unquote

import mlx
from fastapi import FastAPI, HTTPException, Response, Request, APIRouter
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import scan_cache_dir
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install

from .types.chat.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
)
from .types.model import SupportedModels

# Set up rich logging
install(show_locals=True)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console, markup=True)]
)
logger = logging.getLogger(__name__)

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
    logger.info("[green]MLX libraries successfully imported[/green]")
except ImportError as e:
    logger.error(f"[red]Failed to import MLX libraries: {str(e)}[/red]")
    logger.warning("[yellow]Some functionality will be limited[/yellow]")
    MLX_AVAILABLE = False


class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            logger.info(f"[blue]Processing request to {request.url.path}[/blue]")

            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)

            logger.info(f"[green]Request to {request.url.path} completed in {duration:.2f}s[/green]")

            # Pretty-print the JSON response
            if response.headers.get("Content-Type") == "application/json":
                try:
                    logger.debug(f"[cyan]JSON Response: {json.dumps(response.body.decode(), indent=4)}[/cyan]")
                except json.JSONDecodeError:
                    logger.warning(f"[yellow]Response is not valid JSON: {response.text()}[/yellow]")
            else:
                logger.debug(f"[cyan]Non-JSON Response Content-Type: {response.headers.get('Content-Type')}[/cyan]")

            return response

        return custom_route_handler


app = FastAPI()
router = APIRouter(route_class=TimedRoute)

class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        logger.info("[green]ModelProvider initialized[/green]")

    def load_model(self, model_name: str):
        if model_name not in self.models:
            config = load_config(model_name)
            model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])

            start_time = time.time()
            if model_type in MODELS["vlm"]:
                logger.info(f"[blue]Loading VLM model: {model_name}[/blue]")
                self.models[model_name] = load_vlm_model(model_name, config)
            else:
                logger.info(f"[blue]Loading LM model: {model_name}[/blue]")
                self.models[model_name] = load_lm_model(model_name, config)

            load_time = time.time() - start_time
            logger.info(f"[green]Model {model_name} loaded successfully in {load_time:.2f}s[/green]")

        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in self.models:
                logger.info(f"[blue]Removing model: {model_name}[/blue]")
                del self.models[model_name]
                return True
            logger.warning(f"[yellow]Attempted to remove non-existent model: {model_name}[/yellow]")
            return False

    async def get_available_models(self):
        async with self.lock:
            models = list(self.models.keys())
            logger.debug(f"[cyan]Available models: {models}[/cyan]")
            return models


def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            logger.error(f"[red]Invalid value for int_or_float conversion: {value}[/red]")
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    if num_workers_env := os.getenv("FASTMLX_NUM_WORKERS"):
        try:
            workers = int(num_workers_env)
            logger.info(f"[blue]Using {workers} workers from FASTMLX_NUM_WORKERS env variable[/blue]")
        except ValueError:
            workers = max(1, int(os.cpu_count() * float(num_workers_env)))
            logger.info(f"[blue]Calculated {workers} workers based on CPU count[/blue]")
    return workers


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    logger.info(f"[blue]Setting up CORS middleware with allowed origins: {allowed_origins}[/blue]")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Initialize the ModelProvider
model_provider = ModelProvider()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
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
    logger.info(f"[blue]Received chat completion request for model: {request.model}[/blue]")

    if not MLX_AVAILABLE:
        logger.error("[red]MLX library not available[/red]")
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
            logger.error("[red]Image URL not provided for VLM model[/red]")
            raise HTTPException(
                status_code=400, detail="Image URL not provided for VLM model"
            )

        prompt = ""
        if model.config.model_type != "paligemma":
            prompt = apply_vlm_chat_template(processor, config, chat_messages)
        else:
            prompt = chat_messages[-1]["content"]

        if stream:
            logger.info("[blue]Starting VLM streaming response[/blue]")
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
            logger.info("[blue]Generating VLM response[/blue]")
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
            logger.debug("[cyan]Processing function calling tools[/cyan]")
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
            logger.info("[blue]Starting LM streaming response[/blue]")
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
            logger.info("[blue]Generating LM response[/blue]")
            output, token_length_info = lm_generate(
                model,
                tokenizer,
                prompt,
                request.max_tokens,
                temp=request.temperature,
                stop_words=stop_words,
            )

    # Parse the output to check for function calls
    logger.info("[blue]Processing function calls in response[/blue]")
    return handle_function_calls(output, request, token_length_info)


@router.get("/v1/supported_models", response_model=SupportedModels)
async def get_supported_models():
    """
    Get a list of supported model types for VLM and LM.

    Returns:
        JSONResponse (json): A JSON response containing the supported models.
    """
    logger.info("[blue]Retrieving supported models[/blue]")
    return JSONResponse(content=MODELS)


@router.get("/v1/models")
async def list_models():
    """
    Get list of models - provided in OpenAI API compliant format.
    """
    logger.info("[blue]Retrieving list of loaded models[/blue]")
    models = await model_provider.get_available_models()
    models_data = []
    for model in models:
        models_data.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system",
            }
        )
    return {"object": "list", "data": models_data}


@router.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.

    Args:
        model_name (str): The name of the model to add.

    Returns:
        dict (dict): A dictionary containing the status of the operation.
    """
    logger.info(f"[blue]Adding new model: {model_name}[/blue]")
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@router.delete("/v1/models")
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
    logger.info(f"[blue]Attempting to remove model: {model_name}[/blue]")
    removed = await model_provider.remove_model(model_name)
    if removed:
        logger.info(f"[green]Successfully removed model: {model_name}[/green]")
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        logger.warning(f"[yellow]Failed to remove model: {model_name} (not found)[/yellow]")
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

    logger.info(f"[green]Starting FastMLX server on {args.host}:{args.port} with {args.workers} workers[/green]")
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


app.include_router(router)