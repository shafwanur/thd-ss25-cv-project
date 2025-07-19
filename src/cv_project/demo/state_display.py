from dataclasses import dataclass
from enum import Enum
from typing import final, override

from PySide6.QtCore import (
    QPoint,
    QRect,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
    Qt,
    QTransform,
)
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cv_project.demo.detection.schemas import Klass

from .state import State
from .utils import (
    img_size,
    img_to_pixmap,
    img_width,
)


class LayerId(Enum):
    LABELS = 1
    BOXES = 0
    CONNECTIONS = 2
    IMAGE = 3


@final
class LayeredDisplay(QWidget):
    def __init__(self, state: State):
        super().__init__()

        self.setFixedSize(640, 360)

        self.state = state
        _ = self.state.image_updated.connect(self.on_image)
        _ = self.state.was_reset.connect(self.on_reset)
        # self.scale = False

        self.layers: list[StateDisplayLayer] = []
        self.layers_visibility: list[bool] = [True] * len(LayerId)

        self.init()

    def set_visible(self, layer_id: LayerId, visible: bool):
        self.layers_visibility[layer_id.value] = visible
        self.layers[layer_id.value].setVisible(visible)

    # def set_scale(self, on: bool):
    #     self.scale = on
    #     if on or self.state.img is None:
    #         self.setMinimumSize(640, 360)
    #     else:
    #         self.setMinimumSize(img_size(self.state.img))

    def init(self):
        self.layers.append(BoxesLayer(self.state))
        self.layers.append(LabelsLayer(self.state))
        self.layers.append(ConnectionLayer(self.state))
        self.layers.append(ImageLayer(self.state))

        for layer in self.layers:
            layer.setParent(self)
            layer.setGeometry(QRect(QPoint(0, 0), self.size()))
        for idx in range(1, len(self.layers)):
            self.layers[idx].stackUnder(self.layers[idx - 1])
        for idx in range(len(self.layers)):
            self.layers[idx].setVisible(self.layers_visibility[idx])

    def reset(self):
        print("LayeredDisplay.reset")
        for layer in self.layers:
            layer.hide()
            layer.deleteLater()
        self.layers = []
        self.init()

    # @override
    # def heightForWidth(self, width: int, /) -> int:
    #     if self.state.img is None:
    #         size = QSize(640, 360)
    #     else:
    #         size = img_size(self.state.img)

    #     fraction = width / size.width()
    #     return int(size.height() * fraction)

    @override
    def resizeEvent(self, event: QResizeEvent, /) -> None:
        for layer in self.layers:
            layer.resize(self.size())
        return super().resizeEvent(event)

    def on_image(self):
        assert self.state.img is not None
        self.setFixedSize(img_size(self.state.img))

    def on_reset(self):
        self.reset()


class StateDisplayLayer(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state: State = state
        self.transform: QTransform = QTransform()
        self.last_img_width: int = -1

        _ = self.state.image_updated.connect(self.update_transform)

    def obj(self, id: int):
        return self.state.objects[id]

    def on_transform_updated(self):
        pass

    @final
    def update_transform(self, resized: bool = False):
        if self.state.img is None:
            width = -1
        else:
            width = img_width(self.state.img)

        if not resized and self.last_img_width == width:
            return

        self.last_img_width = width
        if width == -1:
            self.transform.reset()
        else:
            factor = self.width() / width
            self.transform = QTransform.fromScale(factor, factor)
        self.on_transform_updated()
        self.update()

    @override
    def resizeEvent(self, event: QResizeEvent, /) -> None:
        self.update_transform(resized=True)
        return super().resizeEvent(event)


@final
class ImageLayer(StateDisplayLayer):
    def __init__(self, state: State):
        super().__init__(state)

        _ = self.state.image_updated.connect(self.on_image)

    def on_image(self):
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent, /) -> None:
        painter = QPainter(self)

        img = self.state.img
        if img is None:
            painter.fillRect(self.geometry(), QColor(0, 0, 0))
            return

        assert self.transform.isIdentity()
        assert img_size(img) == self.size(), f"{img_size(img)} != {self.size()}"

        painter.setTransform(self.transform)
        painter.drawPixmap(QPoint(0, 0), img_to_pixmap(img))


@dataclass
class BoxLabel:
    frame: QFrame
    info1: QLabel
    info2: QLabel


class LabelsLayer(StateDisplayLayer):
    def __init__(self, state: State):
        super().__init__(state)

        self.labels: dict[int, BoxLabel] = {}

        _ = self.state.object_removed.connect(self.on_removed)
        _ = self.state.object_added.connect(self.on_added)

    @override
    def on_transform_updated(self):
        self.update_positions()

    def update_positions(self):
        for id, label in self.labels.items():
            obj = self.obj(id)
            pos = self.transform.map(obj.rect.topLeft())
            label.frame.move(pos)

    def on_removed(self, id: int):
        old = self.labels.pop(id)
        if not old:
            return
        old.frame.hide()
        old.frame.deleteLater()

    def on_added(self, id: int):
        frame = QFrame(self)
        obj = self.state.objects[id]
        pos = self.transform.map(obj.rect.topLeft())
        frame.move(pos)
        frame.setStyleSheet(
            "background-color: rgba(255, 255, 255, 128); color: black; font-size: 12px"
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)

        if obj.klass == Klass.Chicken:
            info1 = QLabel(
                f"**Confidence**: {round(obj.confidence, 2)}",
                textFormat=Qt.TextFormat.MarkdownText,
            )
        else:
            info1 = QLabel(f"**ID**: {id}", textFormat=Qt.TextFormat.MarkdownText)
        info1.setStyleSheet("background-color: transparent;")
        layout.addWidget(info1)

        info2 = QLabel(textFormat=Qt.TextFormat.MarkdownText)
        info2.setStyleSheet("background-color: transparent;")
        layout.addWidget(info2)

        info3 = QLabel(textFormat=Qt.TextFormat.MarkdownText)
        info3.setStyleSheet("background-color: transparent;")
        layout.addWidget(info3)

        if obj.klass == Klass.Chicken:
            info2.show()
            info2.setText(f"**Eggs**: {len(self.state.chickens[id].eggs)}")

            info3.show()
            info3.setText(f"**Name**: {self.state.chickens[id].name}")
        else:
            info2.hide()
            info3.hide()

        frame.adjustSize()
        frame.show()

        self.labels[id] = BoxLabel(frame=frame, info1=info1, info2=info2)


@dataclass
class BoxDisplayInfo:
    rect: QRect
    color: QColor


@final
class BoxesLayer(StateDisplayLayer):
    def __init__(self, state: State):
        super().__init__(state)
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        _ = self.state.object_removed.connect(self.on_removed)
        _ = self.state.object_added.connect(self.on_added)

        self.boxes: dict[int, BoxDisplayInfo] = {}
        self.box_width = 4

    def on_removed(self, id: int):
        _ = self.boxes.pop(id, None)
        self.update()

    def on_added(self, id: int):
        obj = self.obj(id)
        if obj.klass is Klass.Chicken:
            color = QColor(50, 255, 50)
        else:
            color = QColor(50, 50, 255)

        self.boxes[id] = BoxDisplayInfo(obj.rect, color)
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setTransform(self.transform)

        for box in self.boxes.values():
            pen = QPen(box.color)
            pen.setWidth(self.box_width)
            painter.setPen(pen)
            painter.setBrush(Qt.GlobalColor.transparent)
            painter.drawRect(box.rect)


@dataclass
class ConnectionDisplayInfo:
    p1: QPoint
    p2: QPoint
    color: QColor


@final
class ConnectionLayer(StateDisplayLayer):
    def __init__(self, state: State):
        super().__init__(state)

        _ = self.state.object_removed.connect(self.on_removed)
        _ = self.state.object_added.connect(self.on_added)

        self.links: dict[int, ConnectionDisplayInfo] = {}
        self.line_width = 3

    def on_removed(self, id: int):
        _ = self.links.pop(id, None)
        self.update()

    def on_added(self, id: int):
        obj = self.obj(id)
        if obj.klass is Klass.Chicken:
            return
        egg = self.state.eggs[id]
        if egg.chicken is None:
            return

        color = QColor(50, 50, 255)

        self.links[id] = ConnectionDisplayInfo(
            obj.rect.center(), egg.chicken.obj.rect.center(), color
        )
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setTransform(self.transform)

        for link in self.links.values():
            pen = QPen(link.color)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(self.line_width)
            painter.setPen(pen)
            painter.drawLine(link.p1, link.p2)
