import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Optional, Union

from jinja2 import Environment, FileSystemLoader

from .types.chat.chat_completion import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
)

# MLX Imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as lm_load
    from mlx_lm import models as lm_models
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import generate_step
    from mlx_lm.utils import stream_generate as lm_stream_generate
    from mlx_vlm import load as vlm_load
    from mlx_vlm import models as vlm_models
    from mlx_vlm.utils import load_image_processor
    from mlx_vlm.utils import stream_generate as vlm_stream_generate
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")


def get_model_type_list(models, type="vlm"):

    # Get the directory path of the models package
    models_dir = os.path.dirname(models.__file__)

    # List all items in the models directory
    all_items = os.listdir(models_dir)

    if type == "vlm":
        submodules = [
            item
            for item in all_items
            if os.path.isdir(os.path.join(models_dir, item))
            and not item.startswith(".")
            and item != "__pycache__"
        ]
        return submodules
    else:

        return [item for item in all_items if not item.startswith("__")]


MODELS = {
    "vlm": get_model_type_list(vlm_models),
    "lm": get_model_type_list(lm_models, "lm"),
}
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}


def load_tools_config():
    with open("./fastmlx/tools/config.json", "r") as file:
        return json.load(file)


def get_system_prompt(model_name, tools):
    tool_config = load_tools_config()
    model_config = tool_config["models"].get(
        model_name, tool_config["models"]["default"]
    )
    templates_dir = os.path.abspath("./fastmlx/tools")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(model_config["prompt_template"])

    return template.render(
        tools=tools,
        parallel_tool_calling=model_config.get("parallel_tool_calling", False),
        current_date=datetime.now().strftime("%d %b %Y"),
    )


def get_eom_token(model_type):
    tool_config = load_tools_config()
    model_config = tool_config["models"].get(
        model_type, tool_config["models"]["default"]
    )
    eom_token = model_config.get("eom_token", None)
    return eom_token


def handle_function_calls(output: str, request):
    tool_calls = []
    if "<function_calls>" in output.lower():
        try:
            # Extract all function calls
            function_calls = re.findall(r"<function=(\w+)>\s*({[^<>]+})", output)

            for i, (function_name, args_str) in enumerate(function_calls):
                args = json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        id=f"call_{os.urandom(4).hex()}",
                        function=FunctionCall(
                            name=function_name, arguments=json.dumps(args)
                        ),
                    )
                )

            # Remove the function calls from the output
            output = re.sub(
                r"<function_calls>.*</function_calls>", "", output, flags=re.DOTALL
            ).strip()
        except Exception as e:
            print(f"Error parsing function call: {e}")

    # Prepare the response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{os.urandom(4).hex()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop" if not tool_calls else "tool_call",
            }
        ],
        tool_calls=tool_calls,
    )

    return response


# Model Loading and Generation Functions
def load_vlm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, processor = vlm_load(model_name, {"trust_remote_code": True})
    image_processor = load_image_processor(model_name)
    return {
        "model": model,
        "processor": processor,
        "image_processor": image_processor,
        "config": config,
    }


def load_lm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, tokenizer = lm_load(model_name)
    return {"model": model, "tokenizer": tokenizer, "config": config}


def vlm_stream_generator(
    model,
    model_name,
    processor,
    image,
    prompt,
    image_processor,
    max_tokens,
    temperature,
):
    for token in vlm_stream_generate(
        model,
        processor,
        image,
        prompt,
        image_processor,
        max_tokens=max_tokens,
        temp=temperature,
    ):
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


def lm_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    eom_token = get_eom_token(model.model_type)

    eom_token_id = tokenizer.encode(eom_token)[0] if eom_token else None

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()

    for (token, logprobs), n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id or (eom_token_id and token == eom_token_id):
            break

        detokenizer.add_token(token)

    detokenizer.finalize()
    return detokenizer.text


def lm_stream_generator(model, model_name, tokenizer, prompt, max_tokens, temperature):
    eom_token = get_eom_token(model.model_type)

    for token in lm_stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temp=temperature
    ):
        if eom_token and token == eom_token:
            break

        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    yield "data: [DONE]\n\n"
