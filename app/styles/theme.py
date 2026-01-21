from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    qss_path = Path(__file__).with_name("theme.qss")
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
