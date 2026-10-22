"""Model Settings tab: llama command, sampling, prompt, VAD, TTS, FlashHead."""
import shlex
import subprocess

import llm as llm_client
import logging

logger = logging.getLogger(__name__)
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from runtime import SETTINGS_FILE


def build_model_tab(win):
    """Create win.server_cmd, spin boxes, prompt editor; return the tab."""
    prompt_edit = QPlainTextEdit(win._prompt_template)
    prompt_edit.setFixedHeight(110)
    prompt_edit.setStyleSheet("font-size:11px;")
    prompt_edit.setTabChangesFocus(True)
    prompt_edit.textChanged.connect(win._on_prompt_text)
    prompt_save_btn = QPushButton("Save system prompt")
    prompt_save_btn.clicked.connect(win._save_prompt)
    prompt_save_btn.setToolTip(
        f"Write the prompt to {SETTINGS_FILE}; it is loaded on next launch."
    )
    prompt_row = QHBoxLayout()
    prompt_row.addWidget(prompt_save_btn)
    prompt_row.addStretch(1)

    defaults = dict(llm_client.DEFAULT_SAMPLING)
    saved_sampling = win._settings.get("sampling")
    if isinstance(saved_sampling, dict):
        for key in defaults:
            if key in saved_sampling:
                defaults[key] = type(defaults[key])(saved_sampling[key])
    temp_spin = QDoubleSpinBox()
    temp_spin.setRange(0.0, 2.0)
    temp_spin.setSingleStep(0.05)
    temp_spin.setDecimals(2)
    temp_spin.setValue(defaults["temperature"])
    temp_spin.setToolTip(
        "0 = greedy/deterministic (stable JSON). Higher = more varied replies."
    )
    top_p_spin = QDoubleSpinBox()
    top_p_spin.setRange(0.0, 1.0)
    top_p_spin.setSingleStep(0.01)
    top_p_spin.setDecimals(2)
    top_p_spin.setValue(defaults["top_p"])
    top_p_spin.setToolTip("Nucleus sampling cutoff; 1.0 disables it.")
    top_k_spin = QSpinBox()
    top_k_spin.setRange(0, 400)
    top_k_spin.setValue(defaults["top_k"])
    top_k_spin.setToolTip("Keep only the k most likely tokens; 0 disables it.")
    min_p_spin = QDoubleSpinBox()
    min_p_spin.setRange(0.0, 1.0)
    min_p_spin.setSingleStep(0.01)
    min_p_spin.setDecimals(2)
    min_p_spin.setValue(defaults["min_p"])
    min_p_spin.setToolTip(
        "Drop tokens below this fraction of the most likely token; 0 disables it."
    )
    repeat_pen_spin = QDoubleSpinBox()
    repeat_pen_spin.setRange(1.0, 2.0)
    repeat_pen_spin.setSingleStep(0.05)
    repeat_pen_spin.setDecimals(2)
    repeat_pen_spin.setValue(defaults["repeat_penalty"])
    repeat_pen_spin.setToolTip("Discourage repeated words; 1.0 disables it.")
    max_tokens_spin = QSpinBox()
    max_tokens_spin.setRange(16, 4096)
    max_tokens_spin.setSingleStep(16)
    max_tokens_spin.setValue(int(defaults["max_tokens"]))
    max_tokens_spin.setToolTip(
        "Cap on generated tokens per reply: shorter answers arrive sooner "
        "(generation speed itself is fixed). llama.cpp cannot cap the token "
        "size of an image - use 'Image megapixels' below for that."
    )
    sampling_spins = (temp_spin, top_p_spin, top_k_spin, min_p_spin,
                      repeat_pen_spin, max_tokens_spin)
    for spin in sampling_spins:
        spin.setFixedWidth(90)
    sampling_form = QFormLayout()
    sampling_form.addRow("Temperature", temp_spin)
    sampling_form.addRow("Top-p", top_p_spin)
    sampling_form.addRow("Top-k", top_k_spin)
    sampling_form.addRow("Min-p", min_p_spin)
    sampling_form.addRow("Repeat penalty", repeat_pen_spin)
    sampling_form.addRow("Max reply tokens", max_tokens_spin)
    win._sampling = dict(defaults)
    for spin in sampling_spins:
        spin.valueChanged.connect(win._on_sampling)

    saved_cmd = str(win._settings.get("llama_command", "") or "").strip()
    if saved_cmd:
        try:
            parts = [p.strip('"') for p in shlex.split(saved_cmd, posix=False)]
            parts = [p for p in parts if p]
            if parts:
                win.server.set_command(parts)
        except ValueError as exc:
            logger.warning(f"Invalid saved llama command ({exc}); using defaults.")
            saved_cmd = ""
    server_cmd = QPlainTextEdit(
        saved_cmd or subprocess.list2cmdline(win.server._args())
    )
    server_cmd.setFixedHeight(90)
    server_cmd.setLineWrapMode(QPlainTextEdit.NoWrap)
    server_cmd.setStyleSheet("font-size:11px;")
    server_cmd.setToolTip(
        "Full llama-server command line; press Restart to apply it."
    )
    llama_restart_btn = QPushButton("Restart llama.cpp server")
    llama_restart_btn.clicked.connect(win._restart_llama)
    llama_restart_btn.setToolTip(
        "Stop the server and launch it again with the command above."
    )

    opts = win.server.opts
    ctx_spin = QSpinBox()
    ctx_spin.setRange(1024, 262144)
    ctx_spin.setSingleStep(1024)
    ctx_spin.setValue(int(opts["context"]))
    ctx_spin.setFixedWidth(120)
    ctx_spin.setToolTip(
        "Total context in tokens (system prompt + history + images + audio "
        "+ reply). Larger uses more VRAM; 8192 fits several image turns."
    )
    ngl_spin = QSpinBox()
    ngl_spin.setRange(0, 999)
    ngl_spin.setValue(int(opts["gpu_layers"]))
    ngl_spin.setFixedWidth(120)
    ngl_spin.setToolTip(
        "Model layers on the GPU; 99 = all. Lower offloads to CPU and makes "
        "generation drastically slower - keep at 99 on a 16 GB card."
    )
    fa_combo = QComboBox()
    fa_combo.addItems(["auto", "on", "off"])
    fa_combo.setCurrentText(str(opts["flash_attn"]))
    fa_combo.setFixedWidth(120)
    fa_combo.setToolTip(
        "Flash attention: saves VRAM and speeds up long prompts. 'auto' "
        "lets llama.cpp decide; try 'on' for speed, 'off' only if the "
        "server misbehaves with it enabled."
    )
    kv_combo = QComboBox()
    kv_combo.addItems(["f16", "q8_0", "q4_0"])
    kv_combo.setCurrentText(str(opts["kv_cache"]))
    kv_combo.setFixedWidth(120)
    kv_combo.setToolTip(
        "KV cache precision. q8_0 roughly halves KV VRAM with little "
        "quality loss (helps when the avatar and LLM share the GPU); "
        "q4_0 saves more but can degrade audio/vision quality."
    )
    batch_spin = QSpinBox()
    batch_spin.setRange(64, 8192)
    batch_spin.setSingleStep(128)
    batch_spin.setValue(int(opts["batch"]))
    batch_spin.setFixedWidth(120)
    batch_spin.setToolTip(
        "Prompt batch size (-b/-ub): affects prefill speed only; the "
        "default 1024 is fine for GPU use."
    )
    llama_opts_form = QFormLayout()
    llama_opts_form.addRow("Context (tokens)", ctx_spin)
    llama_opts_form.addRow("GPU layers", ngl_spin)
    llama_opts_form.addRow("Flash attention", fa_combo)
    llama_opts_form.addRow("KV cache", kv_combo)
    llama_opts_form.addRow("Batch size", batch_spin)
    build_cmd_btn = QPushButton("Build command from options")
    build_cmd_btn.clicked.connect(win._build_llama_cmd)
    build_cmd_btn.setToolTip(
        "Rewrite the command box from the options above; then press "
        "Restart below to apply it to the running server."
    )
    save_llama_btn = QPushButton("Save settings for next run")
    save_llama_btn.clicked.connect(win._save_llama_settings)
    save_llama_btn.setToolTip(
        f"Write these options and the built command to {SETTINGS_FILE}; "
        "they are used when the app starts llama-server."
    )
    llama_btn_row = QHBoxLayout()
    llama_btn_row.addWidget(build_cmd_btn)
    llama_btn_row.addWidget(save_llama_btn)
    llama_btn_row.addStretch(1)
    llama_note = QLabel(
        "Options are applied to a running server via Build + Restart. "
        "Saved settings are loaded when the app launches the server - "
        "you need to restart the application for them to take effect there."
    )
    llama_note.setWordWrap(True)
    llama_note.setStyleSheet("font-size:10px; font-style:italic; color:#666;")

    vad_opts = {
        "threshold": 0.5,
        "min_silence_ms": 500,
        "speech_pad_ms": 100,
    }
    saved_vad = win._settings.get("vad")
    if isinstance(saved_vad, dict):
        for key in list(vad_opts):
            if key in saved_vad:
                vad_opts[key] = type(vad_opts[key])(saved_vad[key])
    vad_thresh_spin = QDoubleSpinBox()
    vad_thresh_spin.setRange(0.05, 0.95)
    vad_thresh_spin.setSingleStep(0.05)
    vad_thresh_spin.setDecimals(2)
    vad_thresh_spin.setValue(vad_opts["threshold"])
    vad_thresh_spin.setFixedWidth(90)
    vad_thresh_spin.setToolTip(
        "Speech probability that opens/closes an utterance; "
        "applies when the conversation starts."
    )
    vad_silence_spin = QSpinBox()
    vad_silence_spin.setRange(200, 3000)
    vad_silence_spin.setSingleStep(50)
    vad_silence_spin.setValue(vad_opts["min_silence_ms"])
    vad_silence_spin.setFixedWidth(90)
    vad_silence_spin.setToolTip(
        "Silence length that closes an utterance; applies on start."
    )
    vad_pad_spin = QSpinBox()
    vad_pad_spin.setRange(0, 500)
    vad_pad_spin.setSingleStep(20)
    vad_pad_spin.setValue(vad_opts["speech_pad_ms"])
    vad_pad_spin.setFixedWidth(90)
    vad_pad_spin.setToolTip(
        "Extra audio kept before/after each utterance; applies on start."
    )
    for spin in (vad_thresh_spin, vad_silence_spin, vad_pad_spin):
        spin.valueChanged.connect(win._on_vad_opts)

    try:
        speed = float(win._settings.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed_spin = QDoubleSpinBox()
    speed_spin.setRange(0.5, 2.0)
    speed_spin.setSingleStep(0.05)
    speed_spin.setDecimals(2)
    speed_spin.setValue(speed)
    speed_spin.setFixedWidth(90)
    speed_spin.setToolTip("Kokoro speech rate (1.0 = normal); applies live.")
    speed_spin.valueChanged.connect(win._on_speed)

    seed_spin = QSpinBox()
    seed_spin.setRange(0, 999999999)
    seed_spin.setValue(win.base_seed)
    seed_spin.setFixedWidth(120)
    seed_spin.setToolTip(
        "FlashHead base seed; takes effect when the avatar restarts "
        "(Start, or after changing resolution)."
    )
    seed_spin.valueChanged.connect(win._on_seed)

    megapixel_spin = QDoubleSpinBox()
    megapixel_spin.setRange(0.0, 16.0)
    megapixel_spin.setSingleStep(0.05)
    megapixel_spin.setDecimals(2)
    megapixel_spin.setValue(float(getattr(win, "_image_megapixels", 0.35)))
    megapixel_spin.setFixedWidth(90)
    megapixel_spin.setToolTip(
        "Images are downscaled to at most this many megapixels before being "
        "sent to the LLM. Fewer pixels means far fewer vision tokens and a "
        "faster first answer. 0 = send the image as-is."
    )
    megapixel_set_btn = QPushButton("Set")
    megapixel_set_btn.clicked.connect(win._on_set_megapixels)
    megapixel_set_btn.setToolTip(
        "Apply this limit now (and remember it for the next run)."
    )
    megapixel_row = QHBoxLayout()
    megapixel_row.addWidget(megapixel_spin)
    megapixel_row.addWidget(megapixel_set_btn)
    megapixel_row.addStretch(1)

    llama_group = QGroupBox("llama.cpp server")
    llama_box = QVBoxLayout()
    llama_box.addLayout(llama_opts_form)
    llama_box.addLayout(llama_btn_row)
    llama_box.addWidget(llama_note)
    llama_box.addWidget(server_cmd)
    llama_box.addWidget(llama_restart_btn)
    llama_group.setLayout(llama_box)

    sampling_group = QGroupBox("LLM sampling")
    sampling_box = QVBoxLayout()
    sampling_box.addLayout(sampling_form)
    sampling_group.setLayout(sampling_box)

    prompt_group = QGroupBox("System prompt")
    prompt_box = QVBoxLayout()
    prompt_box.addWidget(prompt_edit)
    prompt_box.addLayout(prompt_row)
    prompt_group.setLayout(prompt_box)

    vision_group = QGroupBox("Vision input (images sent to the LLM)")
    vision_form = QFormLayout()
    vision_form.addRow("Image megapixels", megapixel_row)
    vision_group.setLayout(vision_form)

    silero_group = QGroupBox("Silero VAD")
    silero_form = QFormLayout()
    silero_form.addRow("Threshold", vad_thresh_spin)
    silero_form.addRow("Min silence (ms)", vad_silence_spin)
    silero_form.addRow("Speech pad (ms)", vad_pad_spin)
    silero_group.setLayout(silero_form)

    kokoro_group = QGroupBox("Kokoro TTS")
    kokoro_form = QFormLayout()
    kokoro_form.addRow("Speed", speed_spin)
    kokoro_group.setLayout(kokoro_form)

    flashhead_group = QGroupBox("FlashHead avatar")
    flashhead_form = QFormLayout()
    flashhead_form.addRow("Base seed", seed_spin)
    flashhead_group.setLayout(flashhead_form)

    win.prompt_edit = prompt_edit
    win.prompt_save_btn = prompt_save_btn
    win.temp_spin = temp_spin
    win.top_p_spin = top_p_spin
    win.top_k_spin = top_k_spin
    win.min_p_spin = min_p_spin
    win.repeat_pen_spin = repeat_pen_spin
    win.max_tokens_spin = max_tokens_spin
    win.megapixel_spin = megapixel_spin
    win.ctx_spin = ctx_spin
    win.ngl_spin = ngl_spin
    win.fa_combo = fa_combo
    win.kv_combo = kv_combo
    win.batch_spin = batch_spin
    win.server_cmd = server_cmd
    win.llama_restart_btn = llama_restart_btn
    win._vad_opts = vad_opts
    win.vad_thresh_spin = vad_thresh_spin
    win.vad_silence_spin = vad_silence_spin
    win.vad_pad_spin = vad_pad_spin
    win._speed = speed
    win.speed_spin = speed_spin
    win.seed_spin = seed_spin

    tab = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(llama_group)
    layout.addWidget(sampling_group)
    layout.addWidget(prompt_group)
    layout.addWidget(silero_group)
    layout.addWidget(kokoro_group)
    layout.addWidget(flashhead_group)
    layout.addStretch(1)
    tab.setLayout(layout)
    return tab
