from abc import ABC
from sqlite3 import Connection
from typing import Any, cast, override

import cv2
import numpy as np

from cv_project.training.recipes import make

from ..utils import Img


class Source(ABC):
    def read(self, _img: Img) -> bool:
        return False

    def close(self):
        pass

    def size(self) -> tuple[int, int]: ...


class CameraSource(Source):
    def __init__(self, spec: str):
        idx, wh = spec.split(",")
        w, h = map(int, wh.split("x"))

        self.cap: cv2.VideoCapture = cv2.VideoCapture(int(idx))
        _ = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        _ = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    @override
    def read(self, img: Img) -> bool:
        ok, _ = self.cap.read(img)
        return ok

    @override
    def close(self):
        self.cap.release()

    @override
    def size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h


class VideoSource(Source):
    def __init__(self, path: str):
        print("VideoSource from", path)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(path)

    @override
    def read(self, img: Img) -> bool:
        ok, _ = self.cap.read(img)
        return ok

    @override
    def close(self):
        self.cap.release()

    @override
    def size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h


class ImageSource(Source):
    def __init__(self, path: str):
        self.img: Img = cast(Any, cv2.imread(self.path, cv2.IMREAD_COLOR_BGR))
        self.path: str = path
        self.done: bool = False

    @override
    def read(self, img: Img) -> bool:
        if self.done:
            return False

        self.done = True
        np.copyto(img, self.img)
        return True


def mk_source(typ: str, val: str) -> Source:
    match typ:
        case "camera":
            return CameraSource(val)
        case "video":
            return VideoSource(val)
        case "image":
            return ImageSource(val)
        case "video_url":
            path = make(f"download_video_sample('{val}')/'video.mp4'")
            return VideoSource(str(path))
        case _:
            raise RuntimeError("unknown src type", typ, val)
