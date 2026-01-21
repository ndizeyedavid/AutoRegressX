from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ConfigurePage(QWidget):
    ready_changed = Signal()
    target_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._columns: list[str] = []
        self._auto_suggested: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QLabel("Configure")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Select the target variable (label) for regression")
        hint.setStyleSheet("color: #9bb2db;")

        layout.addWidget(header)
        layout.addWidget(hint)

        group = QGroupBox("Target Selection")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 14, 10, 10)
        group_layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)

        self.auto_btn = QPushButton("Auto-suggest")
        self.auto_btn.clicked.connect(self._apply_auto)
        self.auto_hint = QLabel("Auto-suggestion: —")
        self.auto_hint.setStyleSheet("color: #9bb2db;")

        row.addWidget(QLabel("Target column:"))
        row.addWidget(self.target_combo, 1)
        row.addWidget(self.auto_btn)

        group_layout.addLayout(row)
        group_layout.addWidget(self.auto_hint)

        layout.addWidget(group)
        layout.addStretch(1)

        self._refresh()

    @property
    def is_ready(self) -> bool:
        return bool(self.target_combo.currentText().strip())

    def set_columns(self, columns: list[str]) -> None:
        self._columns = columns
        self.target_combo.clear()
        self.target_combo.addItems(columns)
        self._auto_suggested = columns[-1] if columns else None
        self.auto_hint.setText(f"Auto-suggestion: {self._auto_suggested or '—'}")
        self._refresh()

    def selected_target(self) -> str:
        return self.target_combo.currentText().strip()

    def _apply_auto(self) -> None:
        if self._auto_suggested:
            self.target_combo.setCurrentText(self._auto_suggested)

    def _on_target_changed(self, value: str) -> None:
        self.target_changed.emit(value)
        self._refresh()

    def _refresh(self) -> None:
        self.auto_btn.setEnabled(bool(self._columns))
        self.ready_changed.emit()
