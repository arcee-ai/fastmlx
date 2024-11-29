from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=16000)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)
    tools: Optional[List[Function]] = Field(default=None)
    tool_choice: Optional[str] = Field(default=None)
    stream_options: Optional[Dict[str, Any]] = Field(default={ "include_usage": True })

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    usage: Usage
    choices: List[dict]
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str = Field(default="")
    max_tokens: int = Field(default=16000)
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=1.0)
    n: int = Field(default=1)
    stream: bool = Field(default=False)
    logprobs: Union[int, None] = Field(default=None)
    echo: bool = Field(default=False)
    stop: Union[str, List[str], None] = Field(default=None)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    best_of: int = Field(default=1)
    user: str = Field(default="")
    stream_options: Optional[Dict[str, Any]] = Field(default={ "include_usage": True })