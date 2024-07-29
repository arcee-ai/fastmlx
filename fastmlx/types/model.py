from typing import List

from pydantic import BaseModel


class SupportedModels(BaseModel):
    vlm: List[str]
    lm: List[str]
