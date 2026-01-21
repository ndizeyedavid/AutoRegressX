from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class LogEntry:
    level: str
    message: str


class _TrainingWorker(QObject):
    progress = Signal(int)
    log = Signal(str, str)
    finished = Signal()

    @Slot()
    def run(self) -> None:
        steps = [
            (5, "INFO", "Validating configuration"),
            (18, "INFO", "Preparing preprocessing pipeline"),
            (35, "INFO", "Training Linear Regression"),
            (52, "INFO", "Training Ridge Regression"),
            (67, "INFO", "Training Random Forest"),
            (80, "INFO", "Training Support Vector Regression"),
            (92, "INFO", "Training KNN Regressor"),
            (100, "SUCCESS", "Training completed. Ranking models by R²"),
        ]

        for pct, level, msg in steps:
            time.sleep(0.6)
            self.progress.emit(pct)
            self.log.emit(level, msg)

        self.finished.emit()


class TrainPage(QWidget):
    training_state_changed = Signal()
    training_completed = Signal()

    def __init__(self) -> None:
        super().__init__()

        self.is_running = False
        self.has_completed = False
        self._thread: QThread | None = None
        self._worker: _TrainingWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QLabel("Train Models")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Run multiple regression algorithms and compare results")
        hint.setStyleSheet("color: #9bb2db;")
        layout.addWidget(header)
        layout.addWidget(hint)

        prog_group = QGroupBox("Training Progress")
        prog_layout = QVBoxLayout(prog_group)
        prog_layout.setContentsMargins(10, 14, 10, 10)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        prog_layout.addWidget(self.progress)
        layout.addWidget(prog_group)

        mid = QHBoxLayout()
        mid.setSpacing(14)

        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)
        logs_layout.setContentsMargins(10, 14, 10, 10)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setAcceptRichText(True)
        self.log_view.setFrameShape(QFrame.NoFrame)
        self.log_view.setStyleSheet(
            "background-color: #0b1327; border: 1px solid #1a2d55; border-radius: 12px; padding: 10px;"
        )
        logs_layout.addWidget(self.log_view)

        mid.addWidget(logs_group, 2)

        rank_group = QGroupBox("Model Rankings")
        rank_layout = QVBoxLayout(rank_group)
        rank_layout.setContentsMargins(10, 14, 10, 10)

        self.rank_table = QTableWidget(0, 4)
        self.rank_table.setHorizontalHeaderLabels(["Model", "R²", "MAE", "RMSE"])
        self.rank_table.horizontalHeader().setStretchLastSection(True)
        self.rank_table.setEditTriggers(QTableWidget.NoEditTriggers)

        rank_layout.addWidget(self.rank_table)
        mid.addWidget(rank_group, 3)

        layout.addLayout(mid, 1)

        self._seed_rankings()

    @property
    def can_start(self) -> bool:
        return not self.is_running

    def start_training(self) -> None:
        if self.is_running:
            return

        self._clear_logs()
        self.progress.setValue(0)
        self.has_completed = False

        self.is_running = True
        self.training_state_changed.emit()

        self._thread = QThread()
        self._worker = _TrainingWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)

        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_finished(self) -> None:
        self.is_running = False
        self.has_completed = True
        self.training_state_changed.emit()
        self._seed_rankings(final=True)
        self.training_completed.emit()

    def _append_log(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        lvl = level.upper()

        color = "#9bb2db"
        if lvl == "SUCCESS":
            color = "#27d7a3"
        elif lvl == "WARN":
            color = "#fbbf24"
        elif lvl == "ERROR":
            color = "#fb7185"

        html = (
            f"<div style='margin: 2px 0;'>"
            f"<span style='color:#6f86b6;'>{ts}</span> "
            f"<span style='color:{color}; font-weight:700;'>[{lvl}]</span> "
            f"<span style='color:#e6eefc;'>{message}</span>"
            f"</div>"
        )

        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html)
        cursor.insertBlock()
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _clear_logs(self) -> None:
        self.log_view.clear()

    def _seed_rankings(self, final: bool = False) -> None:
        rows = [
            ("Linear Regression", "—", "—", "—"),
            ("Ridge Regression", "—", "—", "—"),
            ("Random Forest", "—", "—", "—"),
            ("SVR", "—", "—", "—"),
            ("KNN", "—", "—", "—"),
        ]

        if final:
            rows = [
                ("Random Forest", "0.92", "1.81", "2.45"),
                ("Ridge Regression", "0.88", "2.13", "2.77"),
                ("Linear Regression", "0.86", "2.28", "2.92"),
                ("SVR", "0.83", "2.41", "3.05"),
                ("KNN", "0.79", "2.73", "3.46"),
            ]

        self.rank_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.rank_table.setItem(r, c, QTableWidgetItem(val))
