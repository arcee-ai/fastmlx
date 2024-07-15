"""Main module."""

import argparse
import asyncio
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    import mlx.core as mx
    from mlx_lm import generate as lm_generate
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import get_message_json
    from mlx_vlm.utils import load_config

    from .utils import (
        MODEL_REMAPPING,
        MODELS,
        SupportedModels,
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=100)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]


app = FastAPI()


# Custom type function
def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    """Calculate the default number of workers based on environment variable."""
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
    if not MLX_AVAILABLE:
        raise HTTPException(status_code=500, detail="MLX library not available")

    stream = request.stream
    model_data = model_provider.load_model(request.model)
    model = model_data["model"]
    config = model_data["config"]
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])

    if model_type in MODELS["vlm"]:
        processor = model_data["processor"]
        image_processor = model_data["image_processor"]

        image = request.image

        chat_messages = []

        for msg in request.messages:
            if msg.role == "user":
                chat_messages.append(
                    get_message_json(config["model_type"], msg.content)
                )
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        prompt = ""
        if "chat_template" in processor.__dict__.keys():
            prompt = processor.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        elif "tokenizer" in processor.__dict__.keys():
            if model.config.model_type != "paligemma":
                prompt = processor.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = request.messages[-1].content

        if stream:
            return StreamingResponse(
                vlm_stream_generator(
                    model,
                    request.model,
                    processor,
                    request.image,
                    prompt,
                    image_processor,
                    request.max_tokens,
                    request.temperature,
                ),
                media_type="text/event-stream",
            )
        else:
            # Generate the response
            output = vlm_generate(
                model,
                processor,
                image,
                prompt,
                image_processor,
                max_tokens=request.max_tokens,
                temp=request.temperature,
                verbose=False,
            )

    else:
        tokenizer = model_data["tokenizer"]
        chat_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        if tokenizer.chat_template is not None and hasattr(
            tokenizer, "apply_chat_template"
        ):
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = request.messages[-1].content

        if stream:
            return StreamingResponse(
                lm_stream_generator(
                    model,
                    request.model,
                    tokenizer,
                    prompt,
                    request.max_tokens,
                    request.temperature,
                ),
                media_type="text/event-stream",
            )
        else:
            output = lm_generate(
                model, tokenizer, prompt, request.max_tokens, False, request.temperature
            )

    # Prepare the response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{os.urandom(4).hex()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
    )

    return response


@app.get("/v1/supported_models", response_model=SupportedModels)
async def get_supported_models():
    """
    Get a list of supported model types for VLM and LM.
    """
    return JSONResponse(content=MODELS)


@app.get("/v1/models")
async def list_models():
    return {"models": await model_provider.get_available_models()}


@app.post("/v1/models")
async def add_model(model_name: str):
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
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
        default=calculate_default_workers,
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
