from dataclasses import dataclass
from typing import final

from PySide6.QtCore import (
    QObject,
    QRect,
    Signal,
)

from .detection.schemas import Klass
from .utils import Img


@dataclass
class ObjectInfo:
    id: int
    klass: Klass
    confidence: float
    rect: QRect


@dataclass
class ChickenInfo:
    visible: bool
    name: str
    obj: ObjectInfo
    eggs: list["EggInfo"]


@dataclass
class EggInfo:
    visible: bool
    obj: ObjectInfo
    chicken: ChickenInfo | None


@final
class State(QObject):
    image_updated = Signal()
    was_reset = Signal()
    object_added = Signal(int)
    object_removed = Signal(int)

    def __init__(self):
        super().__init__()

        self.img: Img | None = None
        self.objects: dict[int, ObjectInfo] = {}
        self.chickens: dict[int, ChickenInfo] = {}
        self.eggs: dict[int, EggInfo] = {}
        self.all_egg_ids: set[int] = set()

    def reset(self):
        self.img = None
        self.objects = {}
        self.chickens = {}
        self.eggs = {}
        self.all_egg_ids = set()
        self.was_reset.emit()

    def remove_all_objects(self):
        old_objects = self.objects
        self.objects = {}
        self.chickens = {}
        self.eggs = {}

        for id in old_objects:
            self.object_removed.emit(id)
