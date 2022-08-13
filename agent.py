from dataclasses import dataclass
import numpy as np


@dataclass
class ImageAgentInfo:
    name: str
    image_rect: tuple[np.array, np.array]
    conf: float
    team: str


@dataclass
class AgentInfo:
    name: str
    team: str
    health: int
