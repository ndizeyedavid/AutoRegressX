from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget


class ValidationBanner(QFrame):
    action_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ValidationBanner")
        self.setVisible(False)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        self.icon = QLabel("!")
        self.icon.setObjectName("ValidationBannerIcon")
        self.text = QLabel("")
        self.text.setObjectName("ValidationBannerText")
        self.text.setWordWrap(True)

        self.action_btn = QPushButton("Fix")
        self.action_btn.setObjectName("ValidationBannerAction")
        self.action_btn.clicked.connect(self.action_clicked.emit)

        root.addWidget(self.icon, 0, Qt.AlignTop)
        root.addWidget(self.text, 1)
        root.addWidget(self.action_btn, 0, Qt.AlignTop)

    def set_message(self, level: str, text: str, action_text: str | None = None) -> None:
        self.setProperty("level", level.lower())
        self.text.setText(text)
        if action_text is None:
            self.action_btn.setVisible(False)
        else:
            self.action_btn.setVisible(True)
            self.action_btn.setText(action_text)

        self.setVisible(bool(text.strip()))
        self.style().unpolish(self)
        self.style().polish(self)
