from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


class ExportPage(QWidget):
    export_state_changed = Signal()

    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QLabel("Export")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Save the best model, metrics, and plots to a backend-friendly folder")
        hint.setStyleSheet("color: #9bb2db;")

        layout.addWidget(header)
        layout.addWidget(hint)

        dest_group = QGroupBox("Destination")
        dest_layout = QHBoxLayout(dest_group)
        dest_layout.setContentsMargins(10, 14, 10, 10)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select export directory")
        self.path_edit.textChanged.connect(self.export_state_changed)

        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        if qta is not None:
            browse.setIcon(qta.icon("fa5s.folder-open", color="#e6eefc"))

        dest_layout.addWidget(self.path_edit, 1)
        dest_layout.addWidget(browse)

        layout.addWidget(dest_group)

        art_group = QGroupBox("Artifacts")
        art_layout = QVBoxLayout(art_group)
        art_layout.setContentsMargins(10, 14, 10, 10)

        self.cb_model = QCheckBox("model.pkl (best estimator)")
        self.cb_pre = QCheckBox("preprocessing_pipeline.pkl")
        self.cb_metrics = QCheckBox("metrics.json")
        self.cb_rank = QCheckBox("rankings.json")
        self.cb_plots = QCheckBox("plots/ (predicted vs actual, residuals)")

        for cb in (self.cb_model, self.cb_pre, self.cb_metrics, self.cb_rank, self.cb_plots):
            cb.setChecked(True)
            cb.stateChanged.connect(self.export_state_changed)
            if qta is not None:
                cb.setIcon(qta.icon("fa5s.check", color="#27d7a3"))
            art_layout.addWidget(cb)

        layout.addWidget(art_group)

        note = QLabel(
            "Export format will be compatible with FastAPI backends and Raspberry Pi deployment. "
            "(Implementation will be added when we start core logic.)"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9bb2db;")
        layout.addWidget(note)

        layout.addStretch(1)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export directory")
        if path:
            self.path_edit.setText(path)

    def perform_export(self) -> None:
        return
