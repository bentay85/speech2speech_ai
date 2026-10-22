"""Component load-status dots and the overall status line."""
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout

from runtime import COLORS, COMPONENTS, STATE_TEXT


def build_status_panel(win):
    """Create win.overall, win._dots, win._labels; return the layout."""
    overall = QLabel("Loading models...")
    overall.setStyleSheet("font-weight:bold;color:#f5a623;")
    dots = {}
    labels = {}
    box = QVBoxLayout()
    box.addWidget(overall)
    for key, name in COMPONENTS:
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background:{COLORS['pending']};border-radius:5px;")
        label = QLabel(name)
        row = QHBoxLayout()
        row.addWidget(dot)
        row.addWidget(label)
        row.addStretch(1)
        box.addLayout(row)
        dots[key] = dot
        labels[key] = label
    win.overall = overall
    win._dots = dots
    win._labels = labels
    return box


def refresh_status(win):
    """Paint the load states collected on win; start the conversation if queued."""
    states = win._get_loads()
    avatar_state, avatar_detail = win._avatar_state()
    states["avatar"] = (avatar_state, avatar_detail)
    if win._tts is not None:
        win._tts.voice = win.voice_combo.currentData()
    ready = True
    failed = None
    for key, name in COMPONENTS:
        state, detail = states[key]
        win._dots[key].setStyleSheet(f"background:{COLORS[state]};border-radius:5px;")
        text = name if not detail else f"{name} - {detail}"
        win._labels[key].setText(text)
        if state == "failed":
            failed = f"{name}: {detail}"
        if state != "ready":
            ready = False
    if failed:
        win.overall.setText(f"Not ready - {failed}")
        win.overall.setStyleSheet("font-weight:bold;color:#e74c3c;")
    elif ready:
        win.overall.setText("All models loaded - ready")
        win.overall.setStyleSheet("font-weight:bold;color:#2ecc71;")
    else:
        win.overall.setText("Loading models...")
        win.overall.setStyleSheet("font-weight:bold;color:#f5a623;")
    win.start_btn.setEnabled(ready and not win.running)
    win.size_combo.setEnabled(avatar_state in ("ready", "failed"))
    if (
        avatar_state == "ready"
        and not win.running
        and win.status.text().startswith(("Recompiling avatar", "Reloading avatar"))
    ):
        win.status.setText(STATE_TEXT["idle"])
    if ready and win._start_pending:
        win._start_pending = False
        win._open_audio()
