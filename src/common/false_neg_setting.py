from typing import Union
from enum import Enum
from dataclasses import dataclass


class FalseNegMode(Enum):
    NONE = "none"
    ELIMINATION = "elimination"
    ATTRACTION = "attraction"


@dataclass
class FalseNegSettings:
    mode: Union[FalseNegMode, str]
    start_step: int
    memory_step: int

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = FalseNegMode(self.mode)

