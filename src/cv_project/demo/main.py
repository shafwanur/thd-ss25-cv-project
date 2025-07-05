import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import time
from typing import final, override

from names_generator import generate_name
from PySide6.QtCore import (
    QObject,
    QPoint,
    QRect,
    QSize,
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
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cv_project.demo.detection.runner import DetectionRunner, NewFrame
from cv_project.demo.detection.schemas import (
    DetectedObject,
    Klass,
    SrcType,
)

from .state import ChickenInfo, EggInfo, ObjectInfo, State
from .state_display import LayeredDisplay, LayerId
from .utils import (
    distance_between_rects,
    img_size,
)


@final
class BoxerFilter(QObject):
    def __init__(self, state: State):
        super().__init__()
        self.input = input
        self.state = state
        self.add_fake_eggs = False
        self.f = False
        self.chickens = False
        self.confidence_threashold: int = 50

        self.last = time()
        self.cnt = 0

    def set_add_fake_eggs(self, f: bool):
        self.add_fake_eggs = f

    def set_f(self, f: bool):
        self.f = f

    def set_chickens(self, chickens: bool):
        self.chickens = chickens

    def set_min_confidence(self, x: int):
        self.confidence_threashold = x

    def on_updated(self, new_frame: NewFrame):
        # TODO: keep track of ids, for new ids - strict confidence level check, for old - lax?
        # TODO: naming, here or in the BoxesLayer?
        # TODO: keep box for id for some time, then delete
        self.cnt += 1
        if self.cnt == 100:
            self.cnt = 0
            cur = time()
            delta = cur - self.last
            self.last = cur
            print(1 / delta * 100)

        self.state.img = new_frame.img
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
                        rect.left(),
                        rect.top(),
                        rect.right(),
                        rect.bottom(),
                    )
                )

        if not self.f:
            self.state.all_egg_ids = set()

        for obj in chain(new_frame.objects, fake_eggs):
            if int(obj.confidence * 100) < self.confidence_threashold:
                continue

            if self.chickens and obj.klass == Klass.Chicken:
                continue
            info = ObjectInfo(
                obj.id,
                obj.klass,
                obj.confidence,
                QRect(QPoint(obj.x1, obj.y1), QPoint(obj.x2, obj.y2)),
            )

            self.state.objects[obj.id] = info
            if obj.klass == Klass.Chicken:
                self.state.chickens[obj.id] = ChickenInfo(
                    visible=True,
                    name=generate_name(style="capital", seed=obj.id),
                    obj=info,
                    eggs=[],
                )
            else:
                self.state.eggs[obj.id] = EggInfo(visible=True, obj=info, chicken=None)

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
        self.runner = DetectionRunner()
        self.runner.start()

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

        self.filter = BoxerFilter(self.state)
        self.filter.add_fake_eggs = self.options.fake_eggs.isChecked()
        _ = self.options.fake_eggs.toggled.connect(self.filter.set_add_fake_eggs)
        self.filter.f = self.layer_options.conv.isChecked()
        _ = self.layer_options.conv.toggled.connect(self.filter.set_f)
        self.filter.chickens = self.layer_options.hide_chickens.isChecked()
        _ = self.layer_options.hide_chickens.toggled.connect(self.filter.set_chickens)
        self.filter.set_min_confidence(self.layer_options.confidence.value())
        _ = self.layer_options.confidence.valueChanged.connect(
            self.filter.set_min_confidence
        )

        _ = self.runner.new_frame.connect(self.filter.on_updated)

    @override
    def closeEvent(self, event: QCloseEvent, /) -> None:
        self.runner.stop()
        super().closeEvent(event)

    @override
    def resizeEvent(self, event: QResizeEvent, /) -> None:
        self.resized.emit(event.size())
        return super().resizeEvent(event)

    def restart(self):
        self.runner.stop_frames(self._on_stopped)

    def _on_stopped(self):
        self.state.reset()
        self.runner.set_model(self.options.model.currentText(), self._on_model)

    def _on_model(self, ok: bool):
        if not ok:
            _ = QMessageBox.warning(self, "Nope", "set_model failed")
            return
        self.runner.set_source(
            self.options.src_type, self.options.source_value.text(), self._on_source
        )

    def _on_source(self, ok: bool):
        if not ok:
            _ = QMessageBox.warning(self, "Nope", "set_source failed")
            return
        self.state.reset()
        self.runner.start_frames()


@dataclass
class StreamOptions:
    model: str
    src_type: SrcType
    src: str
    fake_eggs: bool


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

        self.confidence = QSlider(Qt.Orientation.Horizontal)
        self.confidence.setMinimum(0)
        self.confidence.setMaximum(100)
        self.confidence.setValue(50)
        layout.addWidget(self.confidence)

        layout.addSpacing(20)

        self.conv = QCheckBox("Conveyor Belt")
        self.conv.setChecked(False)
        layout.addWidget(self.conv)

        self.hide_chickens = QCheckBox("Hide Chickens")
        self.hide_chickens.setChecked(False)
        layout.addWidget(self.hide_chickens)

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
        self.default_camera = "0:640x360"
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

    @property
    def src_type(self) -> SrcType:
        if self.source_camera.isChecked():
            return "camera"
        if self.source_file.isChecked():
            return "video"
        return "video_url"


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()

    def cleanup():
        pass

    _ = app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())
