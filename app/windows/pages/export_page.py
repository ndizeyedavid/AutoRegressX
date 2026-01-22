from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
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
    Artifact("model.joblib", "Best model pipeline (preprocess + estimator)", "—", "fa5s.brain"),
    Artifact("metrics.json", "Evaluation metrics and per-model comparison", "—", "fa5s.file-alt"),
    Artifact("schema.json", "Dataset schema + preprocessing details", "—", "fa5s.file"),
    Artifact("val_predictions.csv", "Validation set predictions", "—", "fa5s.table"),
    Artifact("plots/", "Presentation-ready evaluation charts", "—", "fa5s.chart-bar"),
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
    export_completed = Signal(str)
    export_path_copied = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._remember_last_dir = True
        self._last_dir = ""
        self._run_dir: str | None = None
        self._exported_dir: str | None = None

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

        self.success_card = QFrame()
        self.success_card.setObjectName("ExportSuccessCard")
        sc_layout = QVBoxLayout(self.success_card)
        sc_layout.setContentsMargins(16, 14, 16, 14)
        sc_layout.setSpacing(10)

        self.success_title = QLabel("Export complete")
        self.success_title.setStyleSheet("font-size: 12.5pt; font-weight: 700;")
        self.success_path = QLabel("—")
        self.success_path.setStyleSheet("color: #9bb2db;")

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._open_export_folder)
        if qta is not None:
            self.open_folder_btn.setIcon(qta.icon("fa5s.folder-open", color="#e6eefc"))

        self.copy_path_btn = QPushButton("Copy Path")
        self.copy_path_btn.clicked.connect(self._copy_export_path)
        if qta is not None:
            self.copy_path_btn.setIcon(qta.icon("fa5s.copy", color="#e6eefc"))

        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.copy_path_btn)
        actions.addStretch(1)

        sc_layout.addWidget(self.success_title)
        sc_layout.addWidget(self.success_path)
        sc_layout.addLayout(actions)

        self.success_card.setVisible(False)
        layout.addWidget(self.success_card)

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

    def reset(self) -> None:
        self._run_dir = None
        self._exported_dir = None
        self.best_model_label.setText("Best model: —")
        self.success_path.setText("—")
        self.success_card.setVisible(False)
        self.export_state_changed.emit()

    def set_run_dir(self, run_dir: str | None) -> None:
        self._run_dir = run_dir

    def exported_dir(self) -> str | None:
        return self._exported_dir

    def set_export_preferences(self, remember_last_dir: bool, last_dir: str) -> None:
        self._remember_last_dir = bool(remember_last_dir)
        self._last_dir = str(last_dir or "")

    def _download_all(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export directory", self._last_dir or "")
        if path and self._remember_last_dir:
            self._last_dir = path

        if not path:
            return

        if not self._run_dir:
            return

        src = Path(self._run_dir)
        if not src.exists() or not src.is_dir():
            return

        # Export into a dedicated folder under the chosen directory.
        dest_root = Path(path)
        dest = dest_root / f"AutoRegressX_export_{src.name}"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        except Exception:
            return

        self._exported_dir = str(dest)
        self.success_path.setText(f"Saved to: <span style='color:#27d7a3; font-weight:700;'>{self._exported_dir}</span>")
        self.success_card.setVisible(True)

        self.export_state_changed.emit()
        self.export_completed.emit(str(dest))

    def perform_export(self) -> None:
        self._download_all()

    def _open_export_folder(self) -> None:
        if not self._exported_dir:
            return
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._exported_dir))
        except Exception:
            # Fallback: try opening the parent folder.
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._exported_dir).parent)))
            except Exception:
                pass

    def _copy_export_path(self) -> None:
        if not self._exported_dir:
            return
        try:
            QApplication.clipboard().setText(self._exported_dir)
        except Exception:
            return
        self.export_path_copied.emit(self._exported_dir)
