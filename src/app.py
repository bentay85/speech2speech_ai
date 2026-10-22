"""Native desktop app: live voice conversation with the FlashHead avatar.

On launch the app starts llama-server (Gemma 4 with audio projector) and preloads
the avatar, Silero VAD and Kokoro TTS. A status panel lights each component up as
it becomes ready. Press "Start conversation" to open the microphone and speakers.

Pipeline: microphone -> Silero VAD -> Gemma 4 (audio in) -> Kokoro TTS ->
speakers + avatar. Speaking while the assistant talks barges in.

The avatar is rendered at a fixed size and is never rescaled; resizing the
window does not change the image.
"""
import argparse
import shlex
import subprocess
import sys
import threading
import time
from html import escape
from pathlib import Path

# runtime must be imported before any heavy stack: it truncates the log file,
# configures std logging, sets cache env vars used by torch et al.
from runtime import (
    COMPONENTS,
    HEAVY_IMPORT_LOCK,
    IMAGE_FILTER,
    NAME_TOKEN,
    ROOT,
    SETTINGS_FILE,
    STATE_TEXT,
)

import numpy as np
from PIL import Image
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import logging

logger = logging.getLogger(__name__)

import llm as llm_client
from capture import (
    DESKTOP_DWELL_S,
    CaptureManager,
)
from audio_io import (
    AudioEngine,
    device_label,
    device_name,
)
from llama_server import LlamaServer
from settings import load_settings, write_settings
from ui.main_tab import build_main_tab
from ui.model_tab import build_model_tab
from ui.status_panel import build_status_panel, refresh_status
from widgets import ChatInput, populate_device_combos, show_pixmap


class MainWindow(QMainWindow):
    shut_done = Signal()
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.avatar = None
        self.avatar_cond = None
        self.engine = None
        self.loop = None
        self.server = LlamaServer(
            url=args.server,
            bin_path=str(ROOT / "bin" / "llama-server.exe"),
            model=str(ROOT / "models" / "Gemma4-E4B" / "gemma-4-E4B-it-UD-Q4_K_XL.gguf"),
            mmproj=str(ROOT / "models" / "Gemma4-E4B" / "mmproj-BF16.gguf"),
            log_path=str(ROOT / ".tmp" / "llama-server.log"),
        )
        self.capture = CaptureManager()
        self._capture_region = None
        self._pending = {"image": None}
        self._sent = {"image": None}
        self._closing = False
        self.shut_done.connect(self._quit)
        self._vad = None
        self._tts = None
        self._load_states = {key: ("pending", "") for key, _ in COMPONENTS}
        self._state_lock = threading.Lock()
        self._start_pending = False
        self.running = False
        self.frames = None
        self.frame_i = 0
        self.last_pixmap = None
        self._settings = load_settings()
        s = self._settings

        if isinstance(s.get("llama_opts"), dict):
            self.server.opts.update(
                {k: v for k, v in s["llama_opts"].items() if k in self.server.opts}
            )
        try:
            self._image_megapixels = float(s.get("image_megapixels", 0.35))
        except (TypeError, ValueError):
            self._image_megapixels = 0.35
        self._image_megapixels = max(0.0, self._image_megapixels)

        def pick(arg_name, setting_name, hard_default):
            value = getattr(self.args, arg_name)
            if value is not None:
                return value
            value = s.get(setting_name)
            return value if value is not None else hard_default

        self.args.cond_image = pick("cond_image", "cond_image", str(Path("avatar_images") / "female.jpg"))
        self.args.base_seed = int(pick("base_seed", "seed", 42))
        self.args.size = int(pick("size", "resolution", 384))
        self.args.voice = pick("voice", "voice", "af_heart")
        self.avatar_size = self.args.size
        self.base_seed = self.args.base_seed
        self.avatar_started_size = None
        self.avatar_note = ""
        self._avatar_name = str(s.get("avatar_name", "") or "").strip()
        saved_prompt = str(s.get("system_prompt", "") or "").strip()
        self._prompt_template = saved_prompt or llm_client.SYSTEM_PROMPT
        self._system_prompt = self._apply_name(self._prompt_template)
        self.setWindowTitle("Speech to Speech Avatar")
        self.setMinimumSize(1280, 760)
        self._build_ui()
        self._populate_devices(refresh=True)
        self._restore_audio()
        self._restore_capture()
        hint = self.sizeHint()
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(
            max(self.minimumWidth(), min(hint.width(), avail.width())),
            max(self.minimumHeight(), min(hint.height(), avail.height())),
        )
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._load_timer = QTimer(self)
        self._load_timer.setInterval(200)
        self._load_timer.timeout.connect(self._refresh_status)
        self._capture_timer = QTimer(self)
        self._capture_timer.setTimerType(Qt.PreciseTimer)
        self._capture_timer.setInterval(67)
        self._capture_timer.timeout.connect(self._capture_tick)
        self._capture_timer.start()
        self._capture_cleared = False
        self._last_capture_arr = None
        self._last_llm_arr = None
        self._attach_note = False
        QTimer.singleShot(0, self._preload)

    def _build_ui(self):
        size = self.avatar_size
        self.video = QLabel("No output yet")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setFixedSize(size, size)
        self.video.setStyleSheet("background:#000;color:#888;")

        self.capture_view = QLabel("No capture")
        self.capture_view.setAlignment(Qt.AlignCenter)
        self.capture_view.setFixedSize(512, 384)
        self.capture_view.setStyleSheet("background:#000;color:#888;")
        self.llm_view = QLabel("")
        self.llm_view.setAlignment(Qt.AlignCenter)
        self.llm_view.setFixedSize(384, 384)
        self.llm_view.setStyleSheet("background:#000;")

        status_box = build_status_panel(self)
        main_tab = build_main_tab(self)
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setFrameShape(QFrame.NoFrame)
        model_scroll.setWidget(build_model_tab(self))

        self.start_btn = QPushButton("Start conversation")
        self.start_btn.clicked.connect(self._start)
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        start_stop_row = QHBoxLayout()
        start_stop_row.addWidget(self.start_btn)
        start_stop_row.addWidget(self.stop_btn)
        start_stop_row.addStretch(1)

        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(10)

        self.chat = QPlainTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Conversation transcript")
        self.chat.setStyleSheet("font-size:12px;")

        self.chat_input = ChatInput()
        self.chat_input.setPlaceholderText(
            "Type a message or drop an image here, Enter to send"
        )
        self.chat_input.setEnabled(False)
        self.chat_input.returnPressed.connect(self._send_chat)
        self.chat_input.image_dropped.connect(self._on_image_dropped)
        self.send_btn = QPushButton("Send")
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send_chat)
        chat_row = QHBoxLayout()
        chat_row.addWidget(self.chat_input, 1)
        chat_row.addWidget(self.send_btn)
        self.attach_label = QLabel("")
        self.attach_label.setStyleSheet("color:#666;")

        self.status = QLabel("Idle")
        self.status.setStyleSheet("color:#666;")

        tabs = QTabWidget()
        tabs.addTab(main_tab, "Main")
        tabs.addTab(model_scroll, "Model Settings")

        # Middle column: model status + status label on top, settings in tabs.
        panel = QVBoxLayout()
        panel.addLayout(status_box)
        panel.addWidget(self.status)
        panel.addSpacing(8)
        panel.addWidget(tabs, 1)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(430)

        # Left column: avatar video above the desktop/webcam capture preview.
        video_column = QVBoxLayout()
        video_column.addWidget(QLabel("Avatar"))
        video_column.addWidget(self.video, 0, Qt.AlignTop | Qt.AlignHCenter)
        video_column.addSpacing(8)
        video_column.addWidget(QLabel("Desktop / Webcam Capture"))
        video_column.addWidget(self.capture_view, 0, Qt.AlignTop | Qt.AlignHCenter)
        video_column.addSpacing(8)
        video_column.addWidget(QLabel("Image Queued for the LLM"))
        video_column.addWidget(self.llm_view, 0, Qt.AlignTop | Qt.AlignHCenter)
        video_column.addStretch(1)
        video_widget = QWidget()
        video_widget.setLayout(video_column)

        # Right column: start/stop, then mic level, then the transcript.
        chat_column = QVBoxLayout()
        chat_column.addLayout(start_stop_row)
        chat_column.addWidget(self.level)
        chat_column.addWidget(QLabel("Conversation"))
        chat_column.addWidget(self.chat, 1)
        chat_column.addLayout(chat_row)
        chat_column.addWidget(self.attach_label)
        chat_widget = QWidget()
        chat_widget.setLayout(chat_column)
        chat_widget.setFixedWidth(430)

        row = QHBoxLayout()
        row.addWidget(video_widget, 1)
        row.addWidget(panel_widget)
        row.addWidget(chat_widget)

        root = QWidget()
        root.setLayout(row)
        self.setCentralWidget(root)
        self._update_capture_visibility()
        self._show_cond_preview()

    def _populate_devices(self, refresh=True):
        populate_device_combos(self.in_combo, self.out_combo, refresh=refresh)

    def _refresh_devices(self):
        self._populate_devices(refresh=True)
        self.status.setText("Audio devices refreshed")

    def _collect_settings(self):
        region = self._capture_region
        return {
            "system_prompt": self.prompt_edit.toPlainText().strip(),
            "avatar_name": self._avatar_name,
            "cond_image": self.cond_image.text().strip(),
            "voice": self.voice_combo.currentData(),
            "resolution": self.avatar_size,
            "seed": self.base_seed,
            "speed": self._speed,
            "sampling": dict(self._sampling),
            "vad": dict(self._vad_opts),
            "audio_input": self._audio_missing.get(
                "audio_input", self.in_combo.currentText()
            ),
            "audio_output": self._audio_missing.get(
                "audio_output", self.out_combo.currentText()
            ),
            "capture_mode": self.capture_combo.currentData(),
            "webcam_index": self.capture_index.value(),
            "image_megapixels": self._image_megapixels,
            "llama_opts": dict(self.server.opts),
            "region": None
            if region is None
            else [
                region.x(),
                region.y(),
                region.x() + region.width(),
                region.y() + region.height(),
            ],
            "llama_command": self.server_cmd.toPlainText().strip(),
        }

    def _write_settings(self):
        write_settings(self._collect_settings())

    def _restore_audio(self):
        # If a saved device is currently absent from its combo, remember it so
        # _collect_settings keeps persisting it instead of the fallback that
        # the combo defaulted to while the device was offline.
        self._audio_missing = {}
        for combo, key in (
            (self.in_combo, "audio_input"),
            (self.out_combo, "audio_output"),
        ):
            wanted = str(self._settings.get(key, "") or "")
            if not wanted:
                continue
            found = False
            for i in range(combo.count()):
                if combo.itemText(i) == wanted:
                    combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self._audio_missing[key] = wanted
                combo.currentIndexChanged.connect(
                    lambda _index, k=key: self._audio_missing.pop(k, None)
                )

    def _restore_capture(self):
        s = self._settings
        try:
            self.capture_index.setValue(int(s.get("webcam_index", 0)))
        except (TypeError, ValueError):
            pass
        region = s.get("region")
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                x1, y1, x2, y2 = (int(v) for v in region)
            except ValueError:
                return
            if x2 > x1 and y2 > y1:
                for edit, value in zip(
                    (self.region_x1, self.region_y1, self.region_x2, self.region_y2),
                    (x1, y1, x2, y2),
                ):
                    edit.setText(str(value))
                self._capture_region = QRect(x1, y1, x2 - x1, y2 - y1)
                self.capture_region_label.setText(
                    f"{x2 - x1} x {y2 - y1} at ({x1}, {y1})"
                )
        mode = s.get("capture_mode")
        if mode in ("desktop", "webcam"):
            position = self.capture_combo.findData(mode)
            if position >= 0:
                self.capture_combo.setCurrentIndex(position)

    def _show_cond_preview(self):
        edit = getattr(self, "cond_image", None)
        raw = edit.text().strip() if edit is not None else ""
        if not raw:
            raw = self.args.cond_image or ""
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / raw
        if not path.exists():
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self.video.setPixmap(
            pixmap.scaled(
                self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _pick_cond_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Avatar image", "", IMAGE_FILTER
        )
        if path:
            self.cond_image.setText(path)
            self._show_cond_preview()

    def _update_capture_visibility(self):
        mode = self.capture_combo.currentData()
        for key, widgets in self._video_rows.items():
            for widget in widgets:
                widget.setVisible(mode == key)

    def _on_capture_source(self, index):
        self._update_capture_visibility()
        self._apply_capture()

    def _on_capture_index(self, value):
        if self.capture_combo.currentData() == "webcam":
            self._apply_capture()

    def _apply_capture(self):
        mode = self.capture_combo.currentData()
        ok = self.capture.configure(
            mode,
            region=self._capture_region,
            webcam_index=self.capture_index.value(),
        )
        if not ok:
            self.capture_combo.blockSignals(True)
            self.capture_combo.setCurrentIndex(self.capture_combo.findData("off"))
            self.capture_combo.blockSignals(False)
            self._update_capture_visibility()
            self.status.setText("Capture source failed to start")
        else:
            self._capture_cleared = False

    def _set_capture_region_coords(self):
        fields = (self.region_x1, self.region_y1, self.region_x2, self.region_y2)
        texts = [edit.text().strip() for edit in fields]
        if not any(texts):
            self._commit_capture_region(None)
            return
        if not all(texts):
            self.status.setText(
                "Enter all four pixel coordinates, or leave all empty for the whole desktop"
            )
            return
        x1, y1, x2, y2 = (int(text) for text in texts)
        if x2 <= x1 or y2 <= y1:
            self.status.setText("Capture region needs X2 > X1 and Y2 > Y1")
            return
        self._commit_capture_region(QRect(x1, y1, x2 - x1, y2 - y1))

    def _commit_capture_region(self, rect):
        self._capture_region = rect
        self.capture._dwell_until = time.perf_counter() + DESKTOP_DWELL_S
        if rect is None:
            self.capture_region_label.setText("whole desktop (default)")
            logger.info("Capture region cleared; sending the whole desktop.")
        else:
            self.capture_region_label.setText(
                f"{rect.width()}x{rect.height()} at ({rect.x()}, {rect.y()})"
            )
            logger.info(
                f"Capture region set: {rect.width()}x{rect.height()} "
                f"at ({rect.x()}, {rect.y()})"
            )
        if self.capture_combo.currentData() == "desktop":
            self._apply_capture()

    def _apply_name(self, template):
        return template.replace(NAME_TOKEN, self._avatar_name or "Assistant")

    def _on_set_name(self):
        name = self.name_edit.text().strip()
        if not name:
            self.status.setText("Enter an avatar name first")
            return
        self._avatar_name = name
        self._settings["avatar_name"] = name
        self._system_prompt = self._apply_name(self.prompt_edit.toPlainText())
        if self.loop is not None:
            self.loop.system_prompt = self._system_prompt
        self.status.setText(f"Avatar name set to {name}")

    def _on_prompt_text(self):
        self._system_prompt = self._apply_name(self.prompt_edit.toPlainText())
        if self.loop is not None:
            self.loop.system_prompt = self._system_prompt

    def _save_prompt(self):
        text = self.prompt_edit.toPlainText().strip()
        if not text:
            self.status.setText("System prompt is empty - not saved")
            return
        self._settings["system_prompt"] = text
        self._write_settings()
        self.status.setText(f"System prompt saved to {SETTINGS_FILE.name}")

    def _on_sampling(self):
        self._sampling.update({
            "temperature": round(self.temp_spin.value(), 3),
            "top_p": round(self.top_p_spin.value(), 3),
            "top_k": int(self.top_k_spin.value()),
            "min_p": round(self.min_p_spin.value(), 3),
            "repeat_penalty": round(self.repeat_pen_spin.value(), 3),
            "max_tokens": int(self.max_tokens_spin.value()),
        })

    def _on_set_megapixels(self):
        mp = round(self.megapixel_spin.value(), 2)
        self._image_megapixels = mp
        if self.loop is not None:
            self.loop.image_megapixels = mp
        self._settings["image_megapixels"] = mp
        self.status.setText(
            f"Image megapixels set to {mp} MP"
            + (" (no downscale)" if mp <= 0 else "")
        )

    def _apply_llama_opts(self):
        self.server.opts.update({
            "context": int(self.ctx_spin.value()),
            "gpu_layers": int(self.ngl_spin.value()),
            "flash_attn": self.fa_combo.currentText(),
            "kv_cache": self.kv_combo.currentText(),
            "batch": int(self.batch_spin.value()),
        })

    def _build_llama_cmd(self):
        self._apply_llama_opts()
        self.server_cmd.setPlainText(
            subprocess.list2cmdline(self.server.opts_args())
        )
        self.status.setText(
            "Command rebuilt from options - press 'Restart llama.cpp server' "
            "to apply it now, or 'Save settings for next run'."
        )

    def _save_llama_settings(self):
        self._apply_llama_opts()
        self.server_cmd.setPlainText(
            subprocess.list2cmdline(self.server.opts_args())
        )
        self._write_settings()
        self.status.setText(
            "Saved. Restart the application for these settings to take effect "
            "on launch."
        )

    def _on_vad_opts(self):
        self._vad_opts.update({
            "threshold": round(self.vad_thresh_spin.value(), 3),
            "min_silence_ms": int(self.vad_silence_spin.value()),
            "speech_pad_ms": int(self.vad_pad_spin.value()),
        })

    def _on_speed(self):
        self._speed = round(self.speed_spin.value(), 3)
        if self._tts is not None:
            self._tts.speed = self._speed

    def _on_seed(self, value):
        self.base_seed = int(value)

    def _restart_llama(self):
        if self.running:
            self._teardown_audio()
        parts = []
        try:
            parts = shlex.split(self.server_cmd.toPlainText().strip(), posix=False)
        except ValueError as exc:
            self.status.setText(f"Invalid command: {exc}")
            return
        parts = [part.strip('"') for part in parts if part.strip('"')]
        self.server.set_command(parts)
        self.llama_restart_btn.setEnabled(False)
        self.status.setText("Restarting llama.cpp server...")
        self._set_load("llm", "loading", "restarting")
        threading.Thread(target=self._restart_llama_worker, daemon=True).start()

    def _restart_llama_worker(self):
        ok = False
        try:
            ok = self.server.restart()
            if ok:
                self._set_load("llm", "ready")
            else:
                self._set_load("llm", "failed", "server not healthy")
        except Exception as exc:
            logger.error(f"llama-server restart failed: {exc}")
            self._set_load("llm", "failed", str(exc)[:120])

        def _done():
            self.llama_restart_btn.setEnabled(True)
            self.status.setText(
                "llama.cpp server restarted" if ok else "llama.cpp server restart failed"
            )

        QTimer.singleShot(0, _done)

    def _on_image_dropped(self, path):
        try:
            with Image.open(path) as handle:
                image = np.ascontiguousarray(np.asarray(handle.convert("RGB")))
        except Exception as exc:
            self.status.setText(f"Could not read image: {exc}")
            return
        self._pending["image"] = image
        self._last_llm_arr = image
        self._attach_note = True
        self.attach_label.setText(
            f"Attached: {Path(path).name} - will be sent with the next request"
        )
        self._show_pixmap(self.llm_view, image, height=384)

    def _send_chat(self):
        text = self.chat_input.text().strip()
        if not text and self._pending["image"] is None:
            return
        if not self.running or self.loop is None:
            self.status.setText("Conversation not running")
            return
        if not self.loop.submit_text(text):
            self.status.setText("Conversation stopped, cannot send")
            return
        self.chat_input.clear()

    def _show_pixmap(self, label, arr, height=None):
        show_pixmap(label, arr, height=height)

    def _capture_tick(self):
        pending = self._pending["image"]
        if pending is None and self._attach_note:
            self._attach_note = False
            self.attach_label.setText("")
        queued = pending if pending is not None else self._sent["image"]
        if queued is not None and queued is not self._last_llm_arr:
            self._last_llm_arr = queued
            self._show_pixmap(self.llm_view, queued, height=384)
        elif queued is None and self._last_llm_arr is not None:
            self._last_llm_arr = None
            self.llm_view.clear()
            self.llm_view.setFixedSize(384, 384)
        if self.capture.mode == "off":
            if not self._capture_cleared:
                self._capture_cleared = True
                self.capture_view.setText("No capture")
                self.capture_view.setFixedSize(512, 384)
        else:
            arr = self.capture.display_frame()
            if arr is None:
                if not self._capture_cleared:
                    self._capture_cleared = True
                    self.capture_view.setText("Waiting for frames...")
                    self.capture_view.setFixedSize(512, 384)
            elif arr is not self._last_capture_arr:
                self._capture_cleared = False
                self._last_capture_arr = arr
                self._show_pixmap(self.capture_view, arr, height=384)

    def _set_load(self, key, state, detail=""):
        with self._state_lock:
            self._load_states[key] = (state, detail)

    def _get_loads(self):
        with self._state_lock:
            return dict(self._load_states)

    def _avatar_state(self):
        if self.avatar is None:
            return "pending", ""
        if self.avatar.failed:
            return "failed", self.avatar.failed
        if self.avatar.ready.is_set():
            return "ready", ""
        return "loading", self.avatar_note

    def _all_ready(self):
        states = self._get_loads()
        avatar_state, _ = self._avatar_state()
        return (
            all(states[key][0] == "ready" for key, _ in COMPONENTS if key != "avatar")
            and avatar_state == "ready"
        )

    def _preload(self):
        self._load_timer.start()
        voice = self.voice_combo.currentData()
        threading.Thread(target=self._load_llm, daemon=True).start()
        threading.Thread(target=self._load_vad, daemon=True).start()
        threading.Thread(target=self._load_tts, args=(voice,), daemon=True).start()
        threading.Thread(
            target=self._start_avatar,
            args=(self.cond_image.text().strip(), self.avatar_size),
            daemon=True,
        ).start()

    def _start_avatar(self, cond, size=None):
        with HEAVY_IMPORT_LOCK:
            from avatar import AvatarSession

        size = size or self.avatar_size
        if self.avatar is not None:
            self.avatar.stop()
        self.avatar_note = f"compiling {size}px"
        self.frames = None
        self.frame_i = 0
        self.last_pixmap = None
        logger.info(
            f"Starting avatar at {size}x{size} "
            "(first use of a resolution recompiles and may take several minutes)..."
        )
        self.avatar = AvatarSession(
            ckpt_dir=self.args.ckpt_dir,
            wav2vec_dir=self.args.wav2vec_dir,
            cond_image=cond,
            size=size,
            base_seed=self.base_seed,
        )
        self.avatar_cond = cond
        self.avatar_started_size = size
        self.avatar.start()

    def _load_llm(self):
        self._set_load("llm", "loading", "starting")
        try:
            if self.server.ensure_started():
                self._set_load("llm", "ready")
            else:
                self._set_load("llm", "failed", "cancelled")
        except Exception as exc:
            logger.error(f"LLM server failed: {exc}")
            self._set_load("llm", "failed", str(exc)[:120])

    def _load_vad(self):
        with HEAVY_IMPORT_LOCK:
            from silero_vad.utils_vad import OnnxWrapper

        self._set_load("vad", "loading")
        try:
            model_path = ROOT / "models" / "silero-vad" / "silero_vad.onnx"
            self._vad = OnnxWrapper(str(model_path), force_onnx_cpu=True)
            self._set_load("vad", "ready")
        except Exception as exc:
            logger.error(f"VAD failed: {exc}")
            self._set_load("vad", "failed", str(exc)[:120])

    def _load_tts(self, voice):
        with HEAVY_IMPORT_LOCK:
            from tts import TTS

        self._set_load("tts", "loading", "synthesizing")
        try:
            self._tts = TTS(voice=voice, speed=self._speed)
            self._tts.warmup()
            self._set_load("tts", "ready")
        except Exception as exc:
            logger.error(f"TTS failed: {exc}")
            self._set_load("tts", "failed", str(exc)[:120])

    def _refresh_status(self):
        refresh_status(self)

    def _on_voice(self, index):
        voice = self.voice_combo.itemData(index)
        if voice is None:
            return
        if self._tts is not None:
            self._tts.voice = voice
        if self.loop is not None:
            self.loop.set_voice(voice)

    def _on_size(self, index):
        size = self.size_combo.itemData(index)
        if size is None or size == self.avatar_size:
            return
        self.avatar_size = size
        self.video.setFixedSize(size, size)
        self.frames = None
        self.frame_i = 0
        self.last_pixmap = None
        was_running = self.running
        if was_running:
            self._teardown_audio()
        self._start_pending = was_running
        self.status.setText(
            f"Recompiling avatar at {size}x{size} (first use may take minutes)..."
        )
        threading.Thread(
            target=self._start_avatar,
            args=(self.cond_image.text().strip(), size),
            daemon=True,
        ).start()

    def _start(self):
        if self.running:
            return
        if not self._all_ready():
            self._start_pending = True
            self.status.setText("Waiting for models...")
            return
        cond = self.cond_image.text().strip()
        if (
            cond != self.avatar_cond
            or self.avatar_size != self.avatar_started_size
        ):
            self._start_pending = True
            self.status.setText("Reloading avatar...")
            self._start_avatar(cond, self.avatar_size)
            return
        self._open_audio()

    def _open_audio(self):
        if self._tts is None or self._vad is None:
            self.status.setText("Models not ready")
            return
        in_index = self.in_combo.currentData()
        out_index = self.out_combo.currentData()
        in_name = device_name(in_index)
        logger.info(
            f"Audio input: {device_label(in_index)}; "
            f"output: {device_label(out_index)}"
        )
        if self.args.input_wav:
            logger.info(f"Debug input: {self.args.input_wav} (microphone bypassed)")
        with HEAVY_IMPORT_LOCK:
            from voice_loop import VoiceLoop
        if not self.args.input_wav and any(
            key in in_name.lower()
            for key in ("virtual", "stereo mix", "steam streaming")
        ):
            logger.warning(
                "Selected input is a virtual/loopback device and is usually "
                "silent; choose a physical microphone if the level bar stays empty."
            )
        self.engine = AudioEngine(in_device=in_index, out_device=out_index)
        try:
            self.engine.open_output()
            self.engine.open_input(wav_path=self.args.input_wav or None)
            delay_ms = 1000 * (
                self.engine.out_latency + 0.03 + self.args.av_delay_ms / 1000.0
            )
            logger.info(f"Video delay: {delay_ms:.0f} ms.")
            self.loop = VoiceLoop(
                self.engine,
                self.avatar,
                self.server.url,
                voice=self.voice_combo.currentData(),
                tts=self._tts,
                vad_model=self._vad,
                input_name=in_name,
                av_delay=self.args.av_delay_ms / 1000.0,
                capture=self.capture,
                system_prompt=self._system_prompt,
                sampling=self._sampling,
                vad_opts=self._vad_opts,
                pending=self._pending,
                sent=self._sent,
                image_megapixels=self._image_megapixels,
            )
            self.loop.level.connect(self._on_level)
            self.loop.state_changed.connect(self._on_state)
            self.loop.user_text.connect(self._on_user_text)
            self.loop.reply_text.connect(self._on_reply_text)
            self.loop.interrupted.connect(self._on_interrupted)
            self.loop.failed.connect(self._on_failed)
            self.loop.start()
        except Exception as exc:
            logger.error(f"Failed to start conversation: {exc}")
            self.status.setText(f"Failed: {exc}")
            self._teardown_audio()
            return
        self.frames = None
        self.frame_i = 0
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.in_combo.setEnabled(False)
        self.out_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.chat_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self._timer.start()

    def _stop(self):
        if not self.running:
            return
        self._teardown_audio()
        self.status.setText("Idle")

    def _teardown_audio(self):
        self.running = False
        self._timer.stop()
        if self.loop is not None:
            self.loop.stop()
            self.loop = None
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        self.start_btn.setEnabled(self._all_ready())
        self.stop_btn.setEnabled(False)
        self.in_combo.setEnabled(True)
        self.out_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.chat_input.setEnabled(False)
        self.send_btn.setEnabled(False)

    def _tick(self):
        if self.avatar is None:
            return
        if self.avatar.failed:
            self.status.setText(f"Avatar failed: {self.avatar.failed}")
            self._teardown_audio()
            return
        if self.frames is None or self.frame_i >= len(self.frames):
            frames = self.avatar.next_frames()
            if frames is None:
                return
            self.frames = frames
            self.frame_i = 0
        arr = self.frames[self.frame_i]
        self.frame_i += 1
        height, width, _ = arr.shape
        image = QImage(arr.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.last_pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(self.last_pixmap)

    def _on_level(self, level):
        self.level.setValue(int(level * 100))

    def _on_state(self, state):
        self.status.setText(STATE_TEXT.get(state, state))

    def _on_user_text(self, text):
        self.chat.appendHtml(
            f'<span style="color:#b07a1e;font-weight:bold">You:</span> '
            f"{escape(text or '(unclear audio)')}"
        )

    def _on_reply_text(self, text):
        self.chat.appendHtml(
            f'<span style="color:#1a7f37;font-weight:bold">Assistant:</span> '
            f"{escape(text)}"
        )

    def _on_interrupted(self):
        self.frames = None
        self.frame_i = 0

    def _on_failed(self, message):
        self.status.setText(f"Failed: {message}")

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        self.running = False
        self._write_settings()
        self._timer.stop()
        self._load_timer.stop()
        self._capture_timer.stop()
        if self.loop is not None:
            self.loop.stop()
        self.capture.stop()
        event.accept()
        self.hide()
        threading.Thread(target=self._shutdown_worker, daemon=True).start()

    def _shutdown_worker(self):
        try:
            if self.engine is not None:
                self.engine.stop()
                self.engine = None
            if self.avatar is not None:
                self.avatar.stop()
                self.avatar = None
            self.server.stop()
        except Exception:
            logger.exception("Shutdown failed")
        self.shut_done.emit()

    def _quit(self):
        app = QApplication.instance()
        if app is not None:
            app.quit()


def _parse_args():
    parser = argparse.ArgumentParser(description="Speech to Speech Avatar.")
    parser.add_argument("--ckpt_dir", type=str, default=str(ROOT / "models" / "SoulX-FlashHead-1_3B"))
    parser.add_argument("--wav2vec_dir", type=str, default=str(ROOT / "models" / "wav2vec2-base-960h"))
    parser.add_argument("--cond_image", type=str, default=None,
                        help="Overrides the saved settings.json value.")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="Overrides the saved settings.json value.")
    parser.add_argument("--size", type=int, default=None,
                        help="Overrides the saved settings.json value.")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--voice", type=str, default=None,
                        help="Overrides the saved settings.json value.")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Debug: stream this wav through the mic path instead of a live mic.")
    parser.add_argument("--av_delay_ms", type=float, default=0.0,
                        help="Video timing offset in ms (positive delays the video, negative advances it).")
    return parser.parse_args()


def main():
    args = _parse_args()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
