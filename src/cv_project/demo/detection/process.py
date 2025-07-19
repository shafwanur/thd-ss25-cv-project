import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.model import Model

from ..utils import Img
from .process_utils import Source, mk_source
from .schemas import (
    Cmd,
    CmdGetFrame,
    CmdReply,
    CmdSetModel,
    CmdSetSrc,
    CmdTerminate,
    DetectedObject,
    Klass,
    MsgTerminated,
    ReplyGetFrame,
    ReplySetModel,
    ReplySetSrc,
)


@dataclass
class ImageObj:
    shm: SharedMemory
    img: Img
    ready: bool
    prepared: ReplyGetFrame


class Processor:
    def __init__(self, conn: Connection):
        self.conn: Connection = conn

        self.model: Model | None = None
        self.src: Source | None = None
        self.images: list[ImageObj] = []

        self.prev_img: int = -1
        self.sent_img: int = -1
        self.unknown_id_count: int = -1
        self.failing: bool = False

    def next_img(self, idx: int):
        res = idx + 1
        if res == len(self.images):
            return 0
        assert res >= 0 and res < len(self.images)
        return res

    def run(self):
        while True:
            if self.conn.poll(0):
                msg = self.conn.recv()
            else:
                prepared = False
                if not self.failing:
                    for o in self.images:
                        if not o.ready:
                            # print("prepare")
                            prepared = True
                            self.prepare_ignore(o)
                            break
                if prepared:
                    continue
                # print("all frames ready")
                msg = self.conn.recv()

            try:
                resp = self.on_message(msg)
            except Exception:
                print("on_message failed", msg)
                print(traceback.format_exc())
                resp = self.on_failure(msg)
            self.conn.send(resp)

            if isinstance(resp, MsgTerminated):
                return

    def on_message(self, msg: Cmd) -> CmdReply:
        match msg:
            case CmdTerminate():
                return MsgTerminated()
            case CmdSetModel():
                self.model = YOLO(msg.model_path)
                return ReplySetModel(True)
            case CmdSetSrc():
                self.reset_source()

                self.src = mk_source(msg.src_type, msg.src_value)
                w, h = self.src.size()
                shm_size = w * h * 3
                shape = (h, w, 3)

                names = list[str]()

                for idx in range(3):
                    shm = SharedMemory(create=True, size=shm_size)
                    img: Img = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

                    obj = ImageObj(shm, img, False, ReplyGetFrame(False, idx, []))
                    self.images.append(obj)
                    names.append(shm.name)
                return ReplySetSrc(True, names, w, h)
            case CmdGetFrame():
                assert self.model is not None

                if self.prev_img != -1:
                    o = self.images[self.prev_img]
                    assert o.ready
                    o.ready = False
                if self.sent_img == -1:
                    self.sent_img = 0
                else:
                    self.prev_img = self.sent_img
                    self.sent_img = self.next_img(self.sent_img)
                o = self.images[self.sent_img]

                if not o.ready:
                    self.prepare_raise(o)

                return o.prepared

    def on_failure(self, msg: Cmd) -> CmdReply:
        match msg:
            case CmdTerminate():
                raise NotImplementedError()
            case CmdSetModel():
                return ReplySetModel(False)
            case CmdSetSrc():
                return ReplySetSrc(False, [], -1, -1)
            case CmdGetFrame():
                return ReplyGetFrame(False, -1, [])

    def reset_source(self):
        if self.src is None:
            return

        for o in self.images:
            o.shm.close()
            o.shm.unlink()
        self.images = []

        self.sent_img = -1
        self.prev_img = -1
        self.failing = False

        self.src.close()
        self.src = None

    def prepare_ignore(self, o: ImageObj):
        try:
            self.prepare_raise(o)
        except Exception as exc:
            print(exc)
            pass

    def prepare_raise(self, o: ImageObj):
        if self.failing:
            raise RuntimeError("self.failing is set")

        try:
            ok = self.prepare_optimistic(o)
        except Exception as exc:
            self.failing = True
            raise exc
        if not ok:
            self.failing = True
            raise RuntimeError()

    def prepare_optimistic(self, o: ImageObj):
        assert self.src is not None
        assert self.model is not None
        assert not o.ready

        ok = self.src.read(o.img)
        if not ok:
            print("no frame read")
            return False

        results = self.model.track(
            o.img,
            persist=True,
            show=False,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        objects = list[DetectedObject]()

        if results[0].boxes:
            for box in results[0].boxes:
                if box.id is None:
                    self.unknown_id_count += 1
                    id = -self.unknown_id_count
                else:
                    id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                klass = int(box.cls[0])
                if klass > 1:
                    continue

                obj = DetectedObject(
                    id=int(id),
                    klass=Klass(klass),
                    confidence=(float(box.conf[0])),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                objects.append(obj)

        o.ready = True
        o.prepared.ok = True
        o.prepared.objects = objects
        return True
