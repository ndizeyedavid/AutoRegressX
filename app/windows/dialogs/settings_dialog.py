from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
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


@dataclass(frozen=True)
class AppSettings:
    preview_rows: int
    status_refresh_ms: int
    remember_last_export_dir: bool
    last_export_dir: str
    show_gpu: bool


DEFAULTS = AppSettings(
    preview_rows=15,
    status_refresh_ms=1500,
    remember_last_export_dir=True,
    last_export_dir="",
    show_gpu=True,
)


def load_settings(qsettings: QSettings) -> AppSettings:
    return AppSettings(
        preview_rows=int(qsettings.value("data/preview_rows", DEFAULTS.preview_rows)),
        status_refresh_ms=int(qsettings.value("ui/status_refresh_ms", DEFAULTS.status_refresh_ms)),
        remember_last_export_dir=bool(
            qsettings.value("export/remember_last_dir", DEFAULTS.remember_last_export_dir)
        ),
        last_export_dir=str(qsettings.value("export/last_dir", DEFAULTS.last_export_dir)),
        show_gpu=bool(qsettings.value("ui/show_gpu", DEFAULTS.show_gpu)),
    )


def save_settings(qsettings: QSettings, s: AppSettings) -> None:
    qsettings.setValue("data/preview_rows", int(s.preview_rows))
    qsettings.setValue("ui/status_refresh_ms", int(s.status_refresh_ms))
    qsettings.setValue("export/remember_last_dir", bool(s.remember_last_export_dir))
    qsettings.setValue("export/last_dir", str(s.last_export_dir))
    qsettings.setValue("ui/show_gpu", bool(s.show_gpu))


class SettingsDialog(QDialog):
    settings_applied = Signal(AppSettings)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._qs = QSettings()
        self._current = load_settings(self._qs)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        subtitle = QLabel("Customize AutoRegressX behavior")
        subtitle.setStyleSheet("color: #9bb2db;")

        root.addWidget(title)
        root.addWidget(subtitle)

        data_group = QGroupBox("Data Import")
        data_layout = QFormLayout(data_group)
        data_layout.setContentsMargins(12, 14, 12, 12)
        data_layout.setSpacing(10)

        self.preview_rows = QSpinBox()
        self.preview_rows.setRange(5, 200)
        self.preview_rows.setValue(self._current.preview_rows)
        data_layout.addRow("Preview rows", self.preview_rows)

        export_group = QGroupBox("Export")
        export_layout = QFormLayout(export_group)
        export_layout.setContentsMargins(12, 14, 12, 12)
        export_layout.setSpacing(10)

        self.remember_export_dir = QCheckBox("Remember last export folder")
        self.remember_export_dir.setChecked(self._current.remember_last_export_dir)
        export_layout.addRow(self.remember_export_dir)

        self.last_export_dir = QLineEdit()
        self.last_export_dir.setPlaceholderText("(optional) default export folder")
        self.last_export_dir.setText(self._current.last_export_dir)
        export_layout.addRow("Default export folder", self.last_export_dir)

        ui_group = QGroupBox("Interface")
        ui_layout = QFormLayout(ui_group)
        ui_layout.setContentsMargins(12, 14, 12, 12)
        ui_layout.setSpacing(10)

        self.refresh_ms = QSpinBox()
        self.refresh_ms.setRange(500, 10_000)
        self.refresh_ms.setSingleStep(250)
        self.refresh_ms.setValue(self._current.status_refresh_ms)
        ui_layout.addRow("Status refresh (ms)", self.refresh_ms)

        self.show_gpu = QCheckBox("Show GPU in status bar")
        self.show_gpu.setChecked(self._current.show_gpu)
        ui_layout.addRow(self.show_gpu)

        root.addWidget(data_group)
        root.addWidget(export_group)
        root.addWidget(ui_group)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.reset_btn = QPushButton("Reset to defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.clicked.connect(self._apply)

        buttons.addWidget(self.reset_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.apply_btn)

        root.addLayout(buttons)

    def _reset_defaults(self) -> None:
        self.preview_rows.setValue(DEFAULTS.preview_rows)
        self.refresh_ms.setValue(DEFAULTS.status_refresh_ms)
        self.remember_export_dir.setChecked(DEFAULTS.remember_last_export_dir)
        self.last_export_dir.setText(DEFAULTS.last_export_dir)
        self.show_gpu.setChecked(DEFAULTS.show_gpu)

    def _apply(self) -> None:
        s = AppSettings(
            preview_rows=int(self.preview_rows.value()),
            status_refresh_ms=int(self.refresh_ms.value()),
            remember_last_export_dir=bool(self.remember_export_dir.isChecked()),
            last_export_dir=str(self.last_export_dir.text()).strip(),
            show_gpu=bool(self.show_gpu.isChecked()),
        )
        save_settings(self._qs, s)
        self.settings_applied.emit(s)
        self.accept()
