from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


@dataclass(frozen=True)
class Artifact:
    filename: str
    description: str
    size_label: str
    icon: str


ARTIFACTS: list[Artifact] = [
    Artifact("model.pkl", "Trained model (best estimator)", "2.4 MB", "fa5s.file"),
    Artifact("metrics.json", "Evaluation metrics", "1.2 KB", "fa5s.file-alt"),
    Artifact("pipeline.pkl", "Preprocessing pipeline", "156 KB", "fa5s.cubes"),
    Artifact("plots.zip", "Visualization charts", "845 KB", "fa5s.chart-bar"),
]


class ArtifactCard(QFrame):
    def __init__(self, artifact: Artifact) -> None:
        super().__init__()
        self.setObjectName("ArtifactCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)

        icon = QLabel()
        icon.setFixedSize(34, 34)
        icon.setObjectName("ArtifactIcon")
        icon.setAlignment(Qt.AlignCenter)
        if qta is not None:
            icon.setPixmap(qta.icon(artifact.icon, color="#9bb2db").pixmap(16, 16))
        layout.addWidget(icon)

        mid = QVBoxLayout()
        mid.setSpacing(4)
        name = QLabel(artifact.filename)
        name.setObjectName("ArtifactName")
        desc = QLabel(artifact.description)
        desc.setObjectName("ArtifactDesc")
        mid.addWidget(name)
        mid.addWidget(desc)
        layout.addLayout(mid, 1)

        size = QLabel(artifact.size_label)
        size.setObjectName("ArtifactSize")
        layout.addWidget(size, alignment=Qt.AlignRight | Qt.AlignVCenter)


class ExportPage(QWidget):
    export_state_changed = Signal()

    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QLabel("Export Artifacts")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        self.best_model_label = QLabel("Best model: —")
        self.best_model_label.setStyleSheet("color: #9bb2db;")

        layout.addWidget(header)
        layout.addWidget(self.best_model_label)

        self.artifacts_scroll = QScrollArea()
        self.artifacts_scroll.setWidgetResizable(True)
        self.artifacts_scroll.setFrameShape(QFrame.NoFrame)
        self.artifacts_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(12)

        self._artifact_cards: list[ArtifactCard] = []
        for a in ARTIFACTS:
            card = ArtifactCard(a)
            self._artifact_cards.append(card)
            c_layout.addWidget(card)
        c_layout.addStretch(1)

        self.artifacts_scroll.setWidget(container)
        layout.addWidget(self.artifacts_scroll, 1)

        bottom = QFrame()
        bottom.setObjectName("ExportBottomBar")
        b_layout = QHBoxLayout(bottom)
        b_layout.setContentsMargins(14, 12, 14, 12)

        self.download_btn = QPushButton("Download All Artifacts")
        self.download_btn.setObjectName("DownloadButton")
        self.download_btn.clicked.connect(self._download_all)
        if qta is not None:
            self.download_btn.setIcon(qta.icon("fa5s.download", color="#021012"))
        b_layout.addWidget(self.download_btn, 1)

        layout.addWidget(bottom)

    def set_best_model(self, model_name: str | None) -> None:
        if model_name:
            self.best_model_label.setText(f"Best model: <span style='color:#27d7a3; font-weight:700;'>{model_name}</span>")
        else:
            self.best_model_label.setText("Best model: —")

    def _download_all(self) -> None:
        _ = QFileDialog.getExistingDirectory(self, "Select export directory")

    def perform_export(self) -> None:
        self._download_all()
