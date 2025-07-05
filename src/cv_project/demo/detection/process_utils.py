import time
from abc import ABC
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
        idx, wh = spec.split(":")
        w, h = map(int, wh.split("x"))

        self.cap: cv2.VideoCapture = cv2.VideoCapture(int(idx))
        _ = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        _ = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        print("CameraSource fps", self.cap.get(cv2.CAP_PROP_FPS))

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

        self.frame_count: int = 0
        self.start_time: int = -1
        self.start_ts: int = -1
        self.cur_time: int = -1
        self.cur_ts: int = -1

        print("VideoSource fps", self.cap.get(cv2.CAP_PROP_FPS))

    def upd_time(self):
        self.cur_time = int(time.time() * 1000)

    def upd_ts(self):
        self.cur_ts = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def time(self):
        return self.cur_time - self.start_time

    def ts(self):
        return self.cur_ts - self.start_ts

    @override
    def read(self, img: Img) -> bool:
        ok, _ = self.cap.read(img)
        if not ok:
            return False
        self.frame_count += 1
        if self.frame_count == 1:
            return True

        self.upd_time()
        self.upd_ts()
        if self.frame_count == 2:
            self.start_time = self.cur_time
            self.start_ts = self.cur_ts
            return True

        if self.ts() < self.time() - 100:
            # Skip late images
            while self.ts() < self.time():
                # print("skip late image", self.ts(), self.time())
                ok, _ = self.cap.read(img)
                if not ok:
                    return False
                self.upd_ts()
            self.upd_time()

        while self.time() < self.ts():
            time.sleep(0.001)
            self.upd_time()
        return True

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
