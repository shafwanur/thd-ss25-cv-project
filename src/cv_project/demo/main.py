import sys
import ctypes
import multiprocessing
from pathlib import Path
from itertools import chain
from ultralytics import solutions
from collections.abc import Callable
from dataclasses import dataclass, field
from names_generator import generate_name
from typing import ParamSpec, cast, final, override
from multiprocessing.sharedctypes import SynchronizedString

import cv2
from PySide6.QtCore import (
    QObject,
    QPoint,
    QRect,
    QSize,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QResizeEvent,
    Qt,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO
from ultralytics.engine.model import Model

from cv_project.demo.state_display import LayeredDisplay, LayerId

from .state import ChickenInfo, DetectedObject, EggInfo, Klass, State
from .utils import (
    Img,
    distance_between_rects,
    img_size,
)


@final
class Container[T](QObject):
    updated = Signal()

    def __init__(self):
        super().__init__()
        self._value: T | None = None

    @property
    def value(self) -> T | None:
        return self._value

    @value.setter
    def value(self, x: T | None):
        self._value = x
        self.updated.emit()


@final
class SourceThread(QThread):
    ready: Signal = Signal()

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__()
        self.cap = cap
        self.img = Img((0, 0, 3))

    @override
    def run(self, /):
        while self.cap.isOpened():
            if self.isInterruptionRequested():
                break

            success, img = self.cap.read()
            if not success:
                print("End of video or error reading frame.")
                break
            self.img = cast(Img, img)
            self.ready.emit()
        self.cap.release()


@final
class ImageSource(QObject):
    finished: Signal = Signal()

    def __init__(self, cap: cv2.VideoCapture, container: Container[Img]):
        super().__init__()
        self.container = container
        self.fs_thread = SourceThread(cap)

        _ = self.fs_thread.ready.connect(
            self._on_img, Qt.ConnectionType.BlockingQueuedConnection
        )
        _ = self.fs_thread.finished.connect(self.finished)

    def start(self):
        self.fs_thread.start()

    def _on_img(self):
        self.container.value = self.fs_thread.img

    def interrupt(self):
        print("ImageSource.interrupt")
        self.fs_thread.requestInterruption()

    def wait(self):
        print("ImageSource.wait")
        ok = self.fs_thread.wait()
        print(f"ImageSource.waited {ok}")


@dataclass
class BoxedImage:
    image: Img = field(default_factory=lambda: Img((0, 0, 3)))
    objects: list[DetectedObject] = field(default_factory=list)


@final
class Boxer(QObject):
    def __init__(
        self, model_path: Path, img: Container[Img], boxed: Container[BoxedImage]
    ):
        super().__init__()
        self.model_path: Path = model_path
        self.model: Model = YOLO(model_path)
        self.unknown_id_count: int = 0
        self.img = img
        self.boxed = boxed

        _ = self.img.updated.connect(self.on_updated)

    def on_updated(self):
        image = self.img.value
        if image is None:
            self.boxed.value = None
            return

        results = self.model.track(
            image, persist=True, show=False, verbose=False, tracker="bytetrack.yaml"
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
                    rect=QRect(QPoint(x1, y1), QPoint(x2, y2)),
                )
                objects.append(obj)

        self.boxed.value = BoxedImage(
            image=image,
            objects=objects,
        )


@final
class BoxerFilter(QObject):
    def __init__(self, input: Container[BoxedImage], state: State):
        super().__init__()
        self.input = input
        self.state = state
        self.add_fake_eggs = False
        self.f = False

        _ = self.input.updated.connect(self.on_updated)

    def set_f(self, f):
        self.f = f

    def on_updated(self):
        # TODO: keep track of ids, for new ids - strict confidence level check, for old - lax?
        # TODO: naming, here or in the BoxesLayer?
        # TODO: keep box for id for some time, then delete

        boxed = self.input.value
        if boxed is None:
            self.state.reset()
            return

        self.state.img = boxed.image
        self.state.image_updated.emit()

        self.state.remove_all_objects()

        fake_eggs = list[DetectedObject]()
        if self.add_fake_eggs:
            size = img_size(self.state.img)
            EGGS = 5
            for idx in range(EGGS):
                rect = QRect(0, 0, 20, 30)

                rect.moveCenter(
                    QPoint(int(20 + size.width() / 6 * idx), int(size.height() / 2))
                )

                fake_eggs.append(
                    DetectedObject(
                        1000000 + idx,
                        Klass.Egg,
                        0.9,
                        rect,
                    )
                )

        if not self.f:
            self.state.all_egg_ids = set()

        for obj in chain(boxed.objects, fake_eggs):
            if obj.confidence < 0.75:
                continue
            
            self.state.objects[obj.id] = obj
            if obj.klass == Klass.Chicken:
                self.state.chickens[obj.id] = ChickenInfo(
                    visible=True, name=generate_name(style="capital", seed=obj.id), obj=obj, eggs=[]
                )
            else:
                self.state.eggs[obj.id] = EggInfo(visible=True, obj=obj, chicken=None)

        for egg in self.state.eggs.values():
            self.state.all_egg_ids.add(egg.obj.id)
            dists = [
                (chicken, distance_between_rects(egg.obj.rect, chicken.obj.rect))
                for chicken in self.state.chickens.values()
            ]
            dists = sorted(dists, key=lambda x: x[1])
            if len(dists) > 0:
                chicken = dists[0][0]
                chicken.eggs.append(egg)
                egg.chicken = chicken

        for id in self.state.objects:
            self.state.object_added.emit(id)


@final
class MainWindow(QWidget):
    resized = Signal(QSize)

    def __init__(self):
        super().__init__()
        self.state = State()
        self.source_container = Container[Img]()
        self.boxed_container = Container[BoxedImage]()

        # Video widget

        video_hor_widget = QWidget()
        # video_hor_widget.setSizePolicy(
        #     QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        # )
        video_hor = QHBoxLayout(video_hor_widget)
        video_hor.setContentsMargins(0, 0, 0, 0)
        video_hor.addStretch()

        self.display = LayeredDisplay(self.state)
        video_hor.addWidget(self.display)

        video_hor.addStretch()

        video_widget = QWidget()
        # video_widget.setSizePolicy(
        #     QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        # )
        video_vert = QVBoxLayout(video_widget)
        video_vert.setContentsMargins(0, 0, 0, 0)
        video_vert.addWidget(video_hor_widget)

        video_vert.addStretch()

        # END Video widget

        first_row_widget = QWidget()
        first_row = QHBoxLayout(first_row_widget)
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.addWidget(video_widget)

        side_column_widget = QWidget()
        side_column = QVBoxLayout(side_column_widget)
        side_column.setContentsMargins(0, 0, 0, 0)
        self.info = DisplayInfo(self.state)
        self.info.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        side_column.addWidget(self.info)

        self.layer_options = DisplayOptions()
        self.layer_options.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )
        side_column.addWidget(self.layer_options)

        first_row.addWidget(side_column_widget)

        self.l: QVBoxLayout = QVBoxLayout(self)
        self.l.addWidget(first_row_widget)

        self.options = Options()
        self.options.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.l.addWidget(self.options)
        _ = self.options.changed.connect(self.restart)

        # Connect layer options
        def mk_upd(layer: LayerId):
            def upd(on: bool):
                self.display.set_visible(layer, on)

            return upd

        for layer in LayerId:
            checkbox = self.layer_options.layers[layer.value]

            upd = mk_upd(layer)
            _ = checkbox.toggled.connect(upd)
            upd(checkbox.isChecked())

        # Connect scaling
        # TODO
        self.display.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        # if True:

        #     def upd(on: bool):
        #         self.display.set_scale(on)
        #         if on:
        #             self.display.setSizePolicy(
        #                 QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        #             )
        #         else:
        #             self.display.setSizePolicy(
        #                 QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        #             )

        #     self.layer_options.scale.toggled.connect(upd)
        #     upd(self.layer_options.scale.isChecked())

        self.is_new = True
        self.is_restarting = False
        self.is_closing = False
        self.source: ImageSource
        self.boxer: Boxer
        self.state_updater: BoxerFilter

    @override
    def closeEvent(self, event: QCloseEvent, /) -> None:
        if not self.is_new and not self.is_restarting and not self.is_closing:
            self.is_closing = True
            event.ignore()
            _ = self.source.finished.connect(lambda: self.close())
            self.source.interrupt()
        else:
            super().closeEvent(event)

    @override
    def resizeEvent(self, event: QResizeEvent, /) -> None:
        self.resized.emit(event.size())
        return super().resizeEvent(event)

    def restart(self):
        if self.is_restarting:
            return
        self.is_restarting = True

        if self.is_new:
            self._restart_prepare()
        else:
            self._restart_stop()

    def _restart_stop(self):
        _ = self.source.finished.connect(
            self._restart_ensure_stopped, Qt.ConnectionType.SingleShotConnection
        )
        self.source.interrupt()

    def _restart_ensure_stopped(self):
        self.source.wait()

        QTimer.singleShot(0, self._restart_prepare)

    def _restart_prepare(self):
        if self.options.source_camera.isChecked():
            self._restart_start("camera", self.options.source_value.text())
            return

        if self.options.source_file.isChecked():
            self._restart_start("file", self.options.source_value.text())
            return

        if not self.options.source_url.isChecked():
            raise RuntimeError()

        cap_path = multiprocessing.Array(ctypes.c_char, 4096)
        cap_path.value = b""

        url = self.options.source_value.text()

        cap_proc = WaitProcess.make(
            self,
            cap_path,
            f"download_video_sample('{url}')/'video.mp4'",
            dry=False,
        )
        _ = cap_proc.done.connect(
            lambda: self._restart_start("file", cap_path.value.decode())
        )
        cap_proc.start()

    def _restart_start(self, cap_type: str, cap_value: str):
        model = self.options.model.currentText()

        if model == "" or cap_value == "":
            print("aborted")
            self.is_restarting = False
            return

        if cap_type == "camera":
            cap = cv2.VideoCapture(int(cap_value))
            _ = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _ = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        else:
            cap = cv2.VideoCapture(cap_value)
            assert cap.isOpened(), f"Error reading video file at: {cap_value}"

        self.is_new = False

        self.source = ImageSource(cap, self.source_container)

        self.boxer = Boxer(
            Path(model),
            self.source_container,
            self.boxed_container,
        )

        self.state_updater = BoxerFilter(self.boxed_container, self.state)
        self.state_updater.add_fake_eggs = self.options.fake_eggs.isChecked()
        self.state_updater.f = self.layer_options.conv.isChecked()
        self.layer_options.conv.toggled.connect(self.state_updater.set_f)
        

        self.is_restarting = False
        self.source.start()


@final
class DisplayInfo(QFrame):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        _ = self.state.object_added.connect(self.update_count)
        _ = self.state.object_removed.connect(self.update_count)

        self.setStyleSheet("background-color: lightgray; color: black;")
        layout = QGridLayout(self)

        layout.addWidget(
            QLabel("**Chickens:**", textFormat=Qt.TextFormat.MarkdownText), 0, 0
        )
        self.chicken_count = QLabel("0")
        layout.addWidget(self.chicken_count, 0, 1)

        layout.addWidget(
            QLabel("**Eggs:**", textFormat=Qt.TextFormat.MarkdownText), 1, 0
        )
        self.egg_count = QLabel("0")
        layout.addWidget(self.egg_count, 1, 1)

    def update_count(self):
        self.chicken_count.setText(str(len(self.state.chickens)))
        self.egg_count.setText(str(len(self.state.all_egg_ids)))


@final
class DisplayOptions(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: lightgray; color: black;")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("**Layers**", textFormat=Qt.TextFormat.MarkdownText))

        self.layers = [
            QCheckBox("Show boxes"),
            QCheckBox("Show labels"),
            QCheckBox("Show connections"),
            QCheckBox("Show image"),
        ]
        for layer in self.layers:
            layer.setChecked(True)
            layout.addWidget(layer)

        layout.addSpacing(20)

        self.conv = QCheckBox("Conveyor Belt")
        self.conv.setChecked(False)
        layout.addWidget(self.conv)

        # self.scale = QCheckBox("Scale video")
        # layout.addWidget(self.scale)

        layout.addStretch()


@final
class Options(QFrame):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: lightgray; color: black;")

        layout = QVBoxLayout(self)

        # Model choice

        layout.addWidget(QLabel("**Model**", textFormat=Qt.TextFormat.MarkdownText))
        self.model = QComboBox()
        found_models = list(Path("models").glob("**/*.pt"))
        if len(found_models) == 0:
            layout.addWidget(QLabel("!!!! No models found at ./models !!!!"))
        else:
            for model in sorted(found_models):
                self.model.addItem(str(model))
            layout.addWidget(self.model)
            self.model.setCurrentIndex(0)

        # END Model choice

        layout.addWidget(QLabel("**Source**", textFormat=Qt.TextFormat.MarkdownText))
        self.default_camera = "0"
        self.source_camera = QRadioButton("camera")
        _ = self.source_camera.toggled.connect(
            lambda on: self.source_value.setText(self.default_camera) if on else None
        )
        layout.addWidget(self.source_camera)

        self.default_url = "https://www.youtube.com/watch?v=70IqKloH-mw&pp=ygUNY2hpY2tlbiB2aWRlbw%3D%3D"
        self.source_url = QRadioButton("url")
        _ = self.source_url.toggled.connect(
            lambda on: self.source_value.setText(self.default_url) if on else None
        )
        layout.addWidget(self.source_url)

        self.default_file = "./video.mp4"
        self.source_file = QRadioButton("file")
        _ = self.source_file.toggled.connect(
            lambda on: self.source_value.setText(self.default_file) if on else None
        )
        layout.addWidget(self.source_file)

        self.source_value = QLineEdit()
        layout.addWidget(self.source_value)

        self.fake_eggs = QCheckBox("Fake eggs")
        layout.addWidget(self.fake_eggs)

        self.restart = QPushButton("Restart")
        layout.addWidget(self.restart)
        _ = self.restart.pressed.connect(self.on_restart_pressed)

        self.source_camera.toggle()

    def on_restart_pressed(self):
        self.changed.emit()


P = ParamSpec("P")


@final
class WaitProcess(QFrame):
    done = Signal()

    @staticmethod
    def _make(ret: SynchronizedString, request: str, dry: bool):
        from cv_project.training.recipes import make

        path = make(request, dry=dry)
        ret.value = str(path).encode()

    @staticmethod
    def make(parent: MainWindow, ret: SynchronizedString, request: str, dry: bool):
        return WaitProcess(
            parent, f"make({request})", WaitProcess._make, ret, request, dry
        )

    def __init__(
        self,
        parent: MainWindow,
        info: str,
        target: Callable[P, None],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        super().__init__(parent=parent)

        self.move(0, 0)
        self.resize(parent.size())
        _ = parent.resized.connect(self.resize)

        self.setWindowOpacity(0.5)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 128); color: black;")
        self.show()

        layout = QVBoxLayout(self)
        layout.addStretch()

        self.status = QLabel(f"Waiting for {info}")
        self.status.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        layout.addWidget(self.status)
        self.stop = QPushButton("Stop")
        self.stop.setStyleSheet("background-color: rgba(255, 255, 255, 255);")
        layout.addWidget(self.stop)

        layout.addStretch()

        _ = self.stop.pressed.connect(lambda: self.process.kill())

        self.process = multiprocessing.Process(target=target, args=args, kwargs=kwargs)

    def start(self):
        self.show()
        self.process.start()
        QTimer.singleShot(100, self.check)

    def check(self):
        if self.process.is_alive():
            QTimer.singleShot(100, self.check)
        else:
            self.process.join()
            self.done.emit()
            self.deleteLater()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())
