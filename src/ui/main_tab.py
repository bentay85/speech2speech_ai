"""Main tab: avatar settings and audio/video input-output settings."""
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from runtime import AVATAR_SIZES
from voices import VOICE_LABELS, VOICES


def build_main_tab(win):
    """Create win.cond_image, combos and capture widgets; return the tab."""
    cond_image = QLineEdit(win.args.cond_image or "")
    cond_browse = QPushButton("Browse")
    cond_browse.clicked.connect(win._pick_cond_image)
    cond_row = QHBoxLayout()
    cond_row.addWidget(cond_image)
    cond_row.addWidget(cond_browse)

    in_combo = QComboBox()
    out_combo = QComboBox()
    refresh_btn = QPushButton("Refresh audio devices")
    refresh_btn.clicked.connect(win._refresh_devices)

    voice_combo = QComboBox()
    for voice_id in VOICES:
        voice_combo.addItem(VOICE_LABELS[voice_id], voice_id)
    index = voice_combo.findData(win.args.voice)
    voice_combo.setCurrentIndex(index if index >= 0 else 0)
    voice_combo.currentIndexChanged.connect(win._on_voice)

    size_combo = QComboBox()
    for value in sorted(set(AVATAR_SIZES + [win.args.size])):
        size_combo.addItem(f"{value} x {value}", value)
    index = size_combo.findData(win.avatar_size)
    size_combo.setCurrentIndex(index if index >= 0 else 0)
    size_combo.setToolTip(
        "Changing resolution restarts and recompiles the avatar model."
    )
    size_combo.currentIndexChanged.connect(win._on_size)

    capture_combo = QComboBox()
    for text, mode in (
        ("Off", "off"),
        ("Desktop region", "desktop"),
        ("Webcam", "webcam"),
    ):
        capture_combo.addItem(text, mode)
    capture_combo.setToolTip(
        "Video input sent to the LLM as a still image when speech ends."
    )
    capture_combo.currentIndexChanged.connect(win._on_capture_source)

    region_x1 = QLineEdit()
    region_y1 = QLineEdit()
    region_x2 = QLineEdit()
    region_y2 = QLineEdit()
    for edit in (region_x1, region_y1, region_x2, region_y2):
        edit.setValidator(QIntValidator(0, 9999, win))
        edit.setMaxLength(4)
        edit.setFixedWidth(60)
    capture_coords_btn = QPushButton("Set region")
    capture_coords_btn.clicked.connect(win._set_capture_region_coords)
    region_tl_row = QHBoxLayout()
    region_tl_row.addWidget(QLabel("X1"))
    region_tl_row.addWidget(region_x1)
    region_tl_row.addWidget(QLabel("Y1"))
    region_tl_row.addWidget(region_y1)
    region_tl_row.addStretch(1)
    region_tl_field = QWidget()
    region_tl_field.setLayout(region_tl_row)
    region_br_row = QHBoxLayout()
    region_br_row.addWidget(QLabel("X2"))
    region_br_row.addWidget(region_x2)
    region_br_row.addWidget(QLabel("Y2"))
    region_br_row.addWidget(region_y2)
    region_br_row.addWidget(capture_coords_btn)
    region_br_row.addStretch(1)
    region_br_field = QWidget()
    region_br_field.setLayout(region_br_row)
    capture_region_label = QLabel("whole desktop (default)")
    capture_region_label.setStyleSheet("color:#666;")

    capture_index = QSpinBox()
    capture_index.setRange(0, 9)
    capture_index.setToolTip("cv2.VideoCapture device index for the webcam.")
    capture_index.valueChanged.connect(win._on_capture_index)

    avatar_group = QGroupBox("Avatar")
    avatar_form = QFormLayout()
    avatar_form.addRow("Image", cond_row)
    avatar_form.addRow("Resolution", size_combo)
    avatar_form.addRow("Voice", voice_combo)
    name_edit = QLineEdit()
    name_edit.setPlaceholderText("Name that replaces <Name> in the prompt")
    name_edit.setText(win._avatar_name)
    name_set_btn = QPushButton("Set name")
    name_set_btn.clicked.connect(win._on_set_name)
    name_set_btn.setToolTip(
        "Replaces every <Name> in the system prompt; can be repeated."
    )
    name_row = QHBoxLayout()
    name_row.addWidget(name_edit, 1)
    name_row.addWidget(name_set_btn)
    avatar_form.addRow("Name", name_row)
    avatar_group.setLayout(avatar_form)

    input_group = QGroupBox("Input / Output")
    input_form = QFormLayout()
    input_form.addRow("Audio Input", in_combo)
    input_form.addRow("Audio Output", out_combo)
    input_form.addRow("", refresh_btn)
    input_form.addRow("Video source", capture_combo)
    video_rows = {"desktop": [], "webcam": []}

    def add_capture_row(title, field, key):
        label = QLabel(title)
        input_form.addRow(label, field)
        video_rows[key].extend((label, field))

    add_capture_row("Top-left (px)", region_tl_field, "desktop")
    add_capture_row("Bottom-right (px)", region_br_field, "desktop")
    add_capture_row("Sending", capture_region_label, "desktop")
    add_capture_row("Webcam index", capture_index, "webcam")
    input_group.setLayout(input_form)

    win.cond_image = cond_image
    win.in_combo = in_combo
    win.out_combo = out_combo
    win.refresh_btn = refresh_btn
    win.voice_combo = voice_combo
    win.size_combo = size_combo
    win.capture_combo = capture_combo
    win.region_x1 = region_x1
    win.region_y1 = region_y1
    win.region_x2 = region_x2
    win.region_y2 = region_y2
    win.capture_coords_btn = capture_coords_btn
    win.capture_region_label = capture_region_label
    win.capture_index = capture_index
    win.name_edit = name_edit
    win.name_set_btn = name_set_btn
    win._video_rows = video_rows

    tab = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(avatar_group)
    layout.addWidget(input_group)
    layout.addStretch(1)
    tab.setLayout(layout)
    return tab
