from typing import Any, Dict, List, Optional

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
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=100)
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=0.2)
    tools: Optional[List[Function]] = Field(default=None)
    tool_choice: Optional[str] = Field(default=None)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
