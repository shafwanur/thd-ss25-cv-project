import math

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QImage, QPixmap

Img = NDArray[np.uint8]


def img_width(img: Img):
    if len(img.shape) != 3:
        raise ValueError(f"Invalid img shape: {img.shape}")

    return img.shape[1]


def img_size(img: Img):
    if len(img.shape) != 3:
        raise ValueError(f"Invalid img shape: {img.shape}")

    return QSize(img.shape[1], img.shape[0])


def img_to_pixmap(img: Img):
    if len(img.shape) != 3:
        raise ValueError(f"Invalid img shape: {img.shape}")

    height, width, channels = img.shape
    if channels != 3:
        raise ValueError(f"Unsupported number of channels: {channels}")

    qimage = QImage(
        img.data, width, height, width * channels, QImage.Format.Format_BGR888
    )

    return QPixmap.fromImage(qimage, Qt.ImageConversionFlag.NoFormatConversion)


def distance_between_rects(rect1: QRect, rect2: QRect) -> float:
    if rect1.intersects(rect2):
        return 0

    closest_x1 = max(rect1.left(), min(rect2.left(), rect1.right()))
    closest_y1 = max(rect1.top(), min(rect2.top(), rect1.bottom()))

    closest_x2 = max(rect2.left(), min(rect1.left(), rect2.right()))
    closest_y2 = max(rect2.top(), min(rect1.top(), rect2.bottom()))

    distance = math.sqrt(
        (closest_x1 - closest_x2) ** 2 + (closest_y1 - closest_y2) ** 2
    )
    return distance
