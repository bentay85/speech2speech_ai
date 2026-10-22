"""Desktop / webcam capture that feeds one still frame to the LLM per turn.

Frames are grabbed continuously at their native resolution and the latest one
is displayed below the avatar at 25 fps. On VAD end-of-speech the VoiceLoop
takes a snapshot which is sent to the multimodal LLM at the original capture
resolution; the display then shows one white frame to mark that moment.
"""
import queue
import threading
import time

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

DISPLAY_FPS = 25
DESKTOP_FPS = 10
DESKTOP_DWELL_S = 2.0


def qimage_to_array(image):
    image = image.convertToFormat(QImage.Format_RGB888)
    width = image.width()
    height = image.height()
    stride = image.bytesPerLine()
    raw = image.constBits()
    if hasattr(raw, "setsize"):
        raw.setsize(height * stride)
    arr = np.frombuffer(raw, np.uint8)
    need = height * stride
    if arr.size < need:
        padded = np.zeros(need, dtype=np.uint8)
        padded[: arr.size] = arr
        arr = padded
    arr = arr[:need].reshape((height, stride))
    arr = arr[:, : width * 3]
    return np.array(arr.reshape((height, width, 3)), dtype=np.uint8, copy=True)


class CaptureManager:
    """Owns the live capture; display frames and LLM snapshots come from here."""

    def __init__(self):
        self.mode = "off"
        self.region = None
        self.webcam_index = 0
        self._lock = threading.Lock()
        self._frame = None
        self._flash = False
        self._stop = threading.Event()
        self._thread = None
        self._worker = None
        self._img_q = queue.Queue(maxsize=1)
        self._timer = None
        self._dwell_until = 0.0
        self._last_grab_error = 0.0

    def configure(self, mode, region=None, webcam_index=0):
        self.stop()
        self.mode = mode
        self.region = region
        self.webcam_index = webcam_index
        with self._lock:
            self._frame = None
        self._flash = False
        if mode == "off":
            return True
        if mode == "desktop":
            self._stop.clear()
            self._worker = threading.Thread(target=self._convert_loop, daemon=True)
            self._worker.start()
            self._timer = QTimer()
            self._timer.setTimerType(Qt.PreciseTimer)
            self._timer.setInterval(int(round(1000.0 / DESKTOP_FPS)))
            self._timer.timeout.connect(self._grab_desktop)
            self._timer.start()
            if self.region is None:
                logger.info("Desktop capture started (whole desktop).")
            else:
                logger.info(
                    f"Desktop capture started at "
                    f"{self.region.width()}x{self.region.height()}."
                )
            return True
        if mode == "webcam":
            self._stop.clear()
            self._thread = threading.Thread(target=self._webcam_loop, daemon=True)
            self._thread.start()
            logger.info(f"Webcam capture started on device {self.webcam_index}.")
            return True
        self.mode = "off"
        return False

    def stop(self):
        self._stop.set()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        while True:
            try:
                self._img_q.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop.clear()

    def snapshot(self):
        """Grab the current frame for the LLM and mark the white flash."""
        with self._lock:
            frame = self._frame
            self._flash = True
        return frame.copy() if frame is not None else None

    def display_frame(self):
        """Frame for the next display tick; one white frame right after a snapshot."""
        with self._lock:
            frame = self._frame
            if self._flash:
                self._flash = False
                if frame is not None:
                    return np.full_like(frame, 255)
                return np.full((120, 160, 3), 255, np.uint8)
        return frame

    def _grab_desktop(self):
        try:
            self._grab_desktop_inner()
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_grab_error > 5.0:
                self._last_grab_error = now
                logger.error(f"Desktop capture failed: {exc}")

    def _grab_desktop_inner(self):
        if time.perf_counter() < self._dwell_until:
            return
        rect = self.region
        if rect is None:
            screen = QApplication.primaryScreen()
            shot = screen.grabWindow(0)
        else:
            center = rect.center()
            screen = QApplication.primaryScreen()
            for candidate in QApplication.screens():
                if candidate.geometry().contains(center.x(), center.y()):
                    screen = candidate
                    break
            origin = screen.geometry().topLeft()
            shot = screen.grabWindow(
                0,
                rect.x() - origin.x(),
                rect.y() - origin.y(),
                rect.width(),
                rect.height(),
            )
        if shot.isNull():
            return
        img = shot.toImage()
        try:
            self._img_q.put_nowait(img)
        except queue.Full:
            try:
                self._img_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._img_q.put_nowait(img)
            except queue.Full:
                pass

    def _convert_loop(self):
        while not self._stop.is_set():
            try:
                img = self._img_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                arr = qimage_to_array(img)
            except Exception as exc:
                now = time.perf_counter()
                if now - self._last_grab_error > 5.0:
                    self._last_grab_error = now
                    logger.error(f"Desktop frame conversion failed: {exc}")
                continue
            with self._lock:
                self._frame = arr

    def _webcam_loop(self):
        cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            logger.error(f"Webcam device {self.webcam_index} could not be opened.")
            return
        interval = 1.0 / DISPLAY_FPS
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                logger.warning("Webcam read failed; retrying.")
                time.sleep(0.2)
                continue
            arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = arr
            sleep = interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
        cap.release()
