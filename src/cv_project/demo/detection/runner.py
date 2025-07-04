import multiprocessing
from dataclasses import dataclass
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, cast, final

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal

from ..utils import Img
from .schemas import (
    Cmd,
    CmdGetFrame,
    CmdReply,
    CmdSetModel,
    CmdSetSrc,
    CmdTerminate,
    DetectedObject,
    ReplyGetFrame,
    ReplySetModel,
    ReplySetSrc,
    SrcType,
)


def run_detection(conn: Connection):
    from .process import Processor

    p = Processor(conn)
    try:
        p.run()
    finally:
        p.reset_images()


@dataclass
class PipeReply:
    content: CmdReply


@final
class MsgPipe(QObject):
    response_received = Signal(PipeReply)

    def __init__(self, pipe: Connection):
        super().__init__()
        _ = self.destroyed.connect(lambda: print("MsgPipe destroyed"))

        self.pipe = pipe

    def send(self, obj: Cmd):
        QTimer.singleShot(0, self, lambda: self._send(obj))

    def _send(self, obj: Cmd):
        if self.pipe.closed:
            print("MsgPipe.send closed ", obj)
            return

        # print("PipeReader.send ", obj)
        self.pipe.send(obj)
        self._recv()

    def _recv(self):
        obj = self.pipe.recv()
        # print("PipeReader.recv ", obj)
        self.response_received.emit(PipeReply(obj))


@dataclass
class NewFrame:
    img: Img
    objects: list[DetectedObject]


@final
class DetectionRunner(QObject):
    _model_updated = Signal(bool)
    _source_updated = Signal(bool)
    _frames_stopped = Signal()
    new_frame = Signal(NewFrame)

    def __init__(self):
        super().__init__()
        _ = self.destroyed.connect(lambda: print("DetectionRunner destroyed"))

        self.shm_images: list[SharedMemory] = []
        self.images: list[Img] = []
        self.request_frames = False
        self.requesting_frames = False

        # Initialize a pipe

        self.pipe_thread = QThread(self)
        _ = self.pipe_thread.destroyed.connect(
            lambda: print("DetectionRunner.pipe_thread destroyed")
        )

        here, there = Pipe()
        self.pipe = MsgPipe(cast(Connection, cast(object, here)))
        _ = self.pipe.moveToThread(self.pipe_thread)

        _ = self.pipe.response_received.connect(
            self._on_response, Qt.ConnectionType.BlockingQueuedConnection
        )

        # END Initialize a pipe

        self.process = multiprocessing.Process(target=run_detection, args=[there])

    def set_model(self, model: str, cb: Callable[[bool], None]):
        _ = self._model_updated.connect(cb, Qt.ConnectionType.SingleShotConnection)
        self.pipe.send(CmdSetModel(model))

    def set_source(self, src_type: SrcType, src_value: str, cb: Callable[[bool], None]):
        _ = self._source_updated.connect(cb, Qt.ConnectionType.SingleShotConnection)
        self.pipe.send(CmdSetSrc(src_type, src_value))

    def start_frames(self):
        self.request_frames = True
        self.requesting_frames = True
        self.pipe.send(CmdGetFrame())

    def stop_frames(self, cb: Callable[[], None]):
        self.request_frames = False
        if not self.requesting_frames:
            cb()
        else:
            _ = self._frames_stopped.connect(cb, Qt.ConnectionType.SingleShotConnection)

    def start(self):
        self.pipe_thread.start()
        self.process.start()

    def stop(self):
        _ = self.pipe.response_received.disconnect(self._on_response)

        self._reset_shm_image()
        self.pipe.send(CmdTerminate())
        self.process.join()
        self.pipe.deleteLater()
        self.pipe_thread.quit()
        _ = self.pipe_thread.wait()
        self.deleteLater()

    def _on_response(self, msg: PipeReply):
        match msg.content:
            case ReplySetModel() as obj:
                self._model_updated.emit(obj.ok)
            case ReplySetSrc() as obj:
                self._reset_shm_image()
                shape = (obj.height, obj.width, 3)
                for name in obj.shm_names:
                    shm_image = SharedMemory(name=name)
                    image: Img = np.ndarray(shape, dtype=np.uint8, buffer=shm_image.buf)
                    self.shm_images.append(shm_image)
                    self.images.append(image)
                self._source_updated.emit(obj.ok)
            case ReplyGetFrame() as obj:
                if not obj.ok:
                    print("no frame: stopping requests")
                    self.requesting_frames = False
                    self._frames_stopped.emit()
                    return
                if not self.request_frames:
                    print("not self.request_frames: stopping requests")
                    self.requesting_frames = False
                    self._frames_stopped.emit()
                    return

                resp = NewFrame(self.images[obj.idx], obj.objects)
                self.new_frame.emit(resp)
                self.pipe.send(CmdGetFrame())

    def _reset_shm_image(self):
        while self.shm_images:
            shm = self.shm_images.pop()
            shm.close()
        self.images = []
