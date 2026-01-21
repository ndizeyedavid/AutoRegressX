from __future__ import annotations

from PySide6.QtCore import QPropertyAnimation, QTimer, Qt
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class Toast(QFrame):
    def __init__(self, level: str, title: str, message: str) -> None:
        super().__init__()
        self.setObjectName("Toast")
        self.setProperty("level", level.lower())

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(4)

        t = QLabel(title)
        t.setObjectName("ToastTitle")
        m = QLabel(message)
        m.setObjectName("ToastMessage")
        m.setWordWrap(True)

        root.addWidget(t)
        root.addWidget(m)

        eff = QGraphicsOpacityEffect(self)
        eff.setOpacity(0.0)
        self.setGraphicsEffect(eff)
        self._eff = eff

        self._anim = QPropertyAnimation(self._eff, b"opacity", self)
        self._anim.setDuration(160)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)

    def fade_in(self) -> None:
        self._anim.stop()
        self._anim.setDirection(QPropertyAnimation.Forward)
        self._anim.start()

    def fade_out(self) -> None:
        self._anim.stop()
        self._anim.setDirection(QPropertyAnimation.Backward)
        self._anim.start()


class ToastHost(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setObjectName("ToastHost")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)

    def show_toast(self, level: str, title: str, message: str, ms: int = 2600) -> None:
        toast = Toast(level, title, message)
        toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(toast, 0)

        wrap = QWidget()
        wrap.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        wrap.setLayout(row)

        self._layout.insertWidget(self._layout.count() - 1, wrap)
        toast.fade_in()

        def _remove() -> None:
            toast.fade_out()
            QTimer.singleShot(180, lambda: wrap.setParent(None))

        QTimer.singleShot(max(400, int(ms)), _remove)
