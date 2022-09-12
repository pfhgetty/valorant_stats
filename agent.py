from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class AgentPictureInfo:
    agent_name: str
    image_rect: tuple[np.array, np.array]
    conf: float
    team: str

    def __hash__(self):
        return hash(
            (
                self.agent_name,
                self.conf,
                self.team,
                self.image_rect[0][0],
                self.image_rect[0][0],
                self.image_rect[1][0],
                self.image_rect[1][1],
            )
        )

@dataclass(frozen=True)
class ScoreboardInfo:
    username: str
    ultimate_status: tuple[int, int]
    kills: int
    deaths: int
    assists: int
    credits: int
    spike_status: bool

@dataclass(frozen=True)
class ScoreboardPosition:
    username_x: int
    ultimate_x : int
    kills_x: int
    deaths_x: int
    assists_x: int
    credits_x: int
    spike_pos: Tuple[int, int]
    y_coord: int


@dataclass(frozen=True)
class PlayerInfo:
    username: str
    ultimate_status: tuple[int, int]
    kills: int
    deaths: int
    assists: int
    credits: int
    spike_status: bool
    agent_name: str
    team: str
    health: int
