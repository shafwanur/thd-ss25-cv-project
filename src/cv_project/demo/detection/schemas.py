from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Klass(Enum):
    Chicken = 0
    Egg = 1


SrcType = Literal["camera", "video", "image", "video_url"]


@dataclass
class CmdTerminate:
    pass


@dataclass
class MsgTerminated:
    pass


@dataclass
class CmdSetModel:
    model_path: str


@dataclass
class ReplySetModel:
    ok: bool


@dataclass
class CmdSetSrc:
    src_type: SrcType
    src_value: str


@dataclass
class ReplySetSrc:
    ok: bool
    shm_names: list[str]
    width: int
    height: int


@dataclass
class CmdGetFrame:
    pass


@dataclass
class DetectedObject:
    id: int
    klass: Klass
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ReplyGetFrame:
    ok: bool
    idx: int
    objects: list[DetectedObject]


Cmd = CmdTerminate | CmdSetModel | CmdSetSrc | CmdGetFrame
CmdReply = MsgTerminated | ReplySetModel | ReplySetSrc | ReplyGetFrame


