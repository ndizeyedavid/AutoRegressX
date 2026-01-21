from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


class DropZone(QWidget):
    file_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)

        icon = QLabel()
        icon.setAlignment(Qt.AlignHCenter)
        if qta is not None:
            icon.setPixmap(qta.icon("fa5s.cloud-upload-alt", color="#9bb2db").pixmap(44, 44))
        else:
            icon.setText("⬆")
            icon.setStyleSheet("font-size: 26pt; color: #9bb2db;")

        title = QLabel("Drop CSV file here")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-size: 12pt; font-weight: 650;")

        sub = QLabel("or click to browse")
        sub.setAlignment(Qt.AlignHCenter)
        sub.setStyleSheet("color: #9bb2db;")

        layout.addStretch(1)
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addStretch(1)

        self.setStyleSheet(
            "border: 2px dashed #1a2d55; border-radius: 12px; background-color: #0b1327;"
        )

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
