from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


class DropZone(QWidget):
    file_dropped = Signal(str)
    browse_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()

        self.setAcceptDrops(True)
        self.setObjectName("DropZone")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)

        icon = QLabel()
        icon.setAlignment(Qt.AlignHCenter)
        if qta is not None:
            icon.setPixmap(qta.icon("fa5s.cloud-upload-alt", color="#9bb2db").pixmap(52, 52))
        else:
            icon.setText("⬆")
            icon.setStyleSheet("font-size: 26pt; color: #9bb2db;")

        badge = QLabel()
        badge.setAlignment(Qt.AlignHCenter)
        badge.setFixedSize(84, 84)
        badge.setObjectName("DropZoneBadge")
        badge_layout = QVBoxLayout(badge)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.addWidget(icon, alignment=Qt.AlignCenter)

        title = QLabel("Drop CSV file here")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-size: 12pt; font-weight: 650;")

        sub = QLabel("or click to browse")
        sub.setAlignment(Qt.AlignHCenter)
        sub.setStyleSheet("color: #9bb2db;")

        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setFixedWidth(160)
        if qta is not None:
            self.browse_btn.setIcon(qta.icon("fa5s.folder-open", color="#e6eefc"))
        self.browse_btn.clicked.connect(self.browse_clicked.emit)

        layout.addStretch(1)
        layout.addWidget(badge, alignment=Qt.AlignHCenter)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(8)
        layout.addWidget(self.browse_btn, alignment=Qt.AlignHCenter)
        layout.addStretch(1)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.browse_clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith(".csv"):
            self.file_dropped.emit(path)
