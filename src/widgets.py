"""Small reusable Qt widgets and helpers."""
from pathlib import Path

import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLineEdit

from audio_io import (
    default_devices,
    device_label,
    input_devices,
    output_devices,
    refresh_devices,
)
from runtime import IMAGE_SUFFIXES


class ChatInput(QLineEdit):
    """Line edit that also accepts an image file dropped onto it."""

    image_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in IMAGE_SUFFIXES:
                self.image_dropped.emit(path)
                event.acceptProposedAction()
                return


def show_pixmap(label, arr, height=None):
    """Blit an HxWx3 uint8 array onto a QLabel, resizing the label to fit."""
    h, w, _ = arr.shape
    image = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image)
    if height is not None:
        pixmap = pixmap.scaledToHeight(height, Qt.SmoothTransformation)
    label.setPixmap(pixmap)
    if label.width() != pixmap.width() or label.height() != pixmap.height():
        label.setFixedSize(pixmap.width(), pixmap.height())


def populate_device_combos(in_combo, out_combo, refresh=True):
    """Refill the input/output device combos, keeping sensible selections."""
    if refresh:
        refresh_devices()
    current_in = in_combo.currentData()
    current_out = out_combo.currentData()
    inputs = input_devices()
    outputs = output_devices()
    defaults = default_devices()
    default_in = defaults[0] if defaults and defaults[0] is not None else -1
    default_out = defaults[1] if defaults and defaults[1] is not None else -1
    in_combo.clear()
    for index, name in inputs:
        in_combo.addItem(device_label(index), index)
    out_combo.clear()
    for index, name in outputs:
        out_combo.addItem(device_label(index), index)
    if not inputs:
        logger.warning(
            "No WDM-KS input devices found; connect a microphone or check "
            "the audio drivers."
        )
    if not outputs:
        logger.warning(
            "No WDM-KS output devices found; check the audio drivers."
        )
    for combo, current, default in (
        (in_combo, current_in, default_in),
        (out_combo, current_out, default_out),
    ):
        position = combo.findData(current)
        if position < 0:
            position = combo.findData(default)
        if position < 0 and combo.count():
            position = 0
        if position >= 0:
            combo.setCurrentIndex(position)
