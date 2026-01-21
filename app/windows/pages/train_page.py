from __future__ import annotations

import random
import time
from datetime import datetime

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


MODELS: list[str] = [
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "SVR",
    "KNN Regression",
]


class _TrainingWorker(QObject):
    log = Signal(str, str)
    model_started = Signal(str)
    model_finished = Signal(str, float, float, float, float)
    finished = Signal()

    @Slot()
    def run(self) -> None:
        self.log.emit("INFO", "Validating configuration")
        time.sleep(0.4)
        self.log.emit("INFO", "Preparing preprocessing pipeline")
        time.sleep(0.5)

        # Deterministic-ish demo results (UI-only)
        demo = {
            "Linear Regression": (0.847, 2.341, 3.102),
            "Ridge Regression": (0.852, 2.298, 3.045),
            "Random Forest": (0.923, 1.456, 1.891),
            "SVR": (0.889, 1.834, 2.312),
            "KNN Regression": (0.812, 2.011, 2.544),
        }

        for name in MODELS:
            self.model_started.emit(name)
            self.log.emit("INFO", f"Training {name}")
            start = time.perf_counter()
            time.sleep(0.45 + random.random() * 0.35)
            elapsed = time.perf_counter() - start
            r2, mae, rmse = demo.get(name, (0.0, 0.0, 0.0))
            self.model_finished.emit(name, r2, mae, rmse, elapsed)

        self.log.emit("SUCCESS", "Training completed. Best model selected by highest R²")
        self.finished.emit()


class MetricTile(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("MetricTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self.title = QLabel(title)
        self.title.setStyleSheet("color: #9bb2db; font-size: 9.5pt;")
        self.value = QLabel("—")
        self.value.setStyleSheet("font-size: 12pt; font-weight: 700;")
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class ModelCard(QFrame):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.setObjectName("ModelCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)

        self.name_label = QLabel(model_name)
        self.name_label.setStyleSheet("font-size: 11pt; font-weight: 650;")

        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #9bb2db;")

        if qta is not None:
            icon_lbl = QLabel()
            icon_lbl.setPixmap(qta.icon("fa5s.chart-line", color="#9bb2db").pixmap(14, 14))
            header.addWidget(icon_lbl)

        header.addWidget(self.name_label)
        header.addStretch(1)
        if qta is not None:
            clock = QLabel()
            clock.setPixmap(qta.icon("fa5s.clock", color="#6f86b6").pixmap(14, 14))
            header.addWidget(clock)
        header.addWidget(self.time_label)
        root.addLayout(header)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tile_r2 = MetricTile("R²")
        self.tile_mae = MetricTile("MAE")
        self.tile_rmse = MetricTile("RMSE")
        tiles.addWidget(self.tile_r2, 1)
        tiles.addWidget(self.tile_mae, 1)
        tiles.addWidget(self.tile_rmse, 1)
        root.addLayout(tiles)

        self.set_running(False)

    def set_running(self, running: bool) -> None:
        self.setProperty("running", running)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_results(self, r2: float, mae: float, rmse: float, seconds: float) -> None:
        self.tile_r2.set_value(f"{r2:.3f}")
        self.tile_mae.set_value(f"{mae:.3f}")
        self.tile_rmse.set_value(f"{rmse:.3f}")
        self.time_label.setText(f"{seconds:.2f}s")


class TrainPage(QWidget):
    training_state_changed = Signal()
    training_completed = Signal()
    best_model_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.is_running = False
        self.has_completed = False
        self._thread: QThread | None = None
        self._worker: _TrainingWorker | None = None

        self._completed_models = 0
        self._results: dict[str, tuple[float, float, float, float]] = {}
        self._current_model: str | None = None
        self.best_model_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        header = QLabel("Model Training")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        title_row.addWidget(header)
        title_row.addStretch(1)

        self.logs_toggle = QPushButton("Logs")
        if qta is not None:
            self.logs_toggle.setIcon(qta.icon("fa5s.stream", color="#e6eefc"))
        self.logs_toggle.clicked.connect(self._toggle_logs)
        title_row.addWidget(self.logs_toggle)
        layout.addLayout(title_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.completed_label = QLabel("0/5 complete")
        self.completed_label.setStyleSheet("color: #9bb2db;")
        self.completed_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.completed_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QFrame.NoFrame)

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)

        self.model_cards: dict[str, ModelCard] = {}
        for name in MODELS:
            card = ModelCard(name)
            self.model_cards[name] = card
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch(1)

        self.cards_scroll.setWidget(self.cards_container)
        left_layout.addWidget(self.cards_scroll, 1)

        best_wrap = QFrame()
        best_wrap.setObjectName("BestModelCard")
        best_layout = QVBoxLayout(best_wrap)
        best_layout.setContentsMargins(16, 12, 16, 12)
        best_layout.setSpacing(6)

        best_title = QLabel("BEST MODEL")
        best_title.setStyleSheet("color: #9bb2db; font-weight: 700; letter-spacing: 1px;")
        self.best_name = QLabel("—")
        self.best_name.setStyleSheet("color: #27d7a3; font-size: 12pt; font-weight: 750;")
        self.best_metrics = QLabel("R² = — | MAE = —")
        self.best_metrics.setStyleSheet("color: #9bb2db;")

        best_layout.addWidget(best_title)
        best_layout.addWidget(self.best_name)
        best_layout.addWidget(self.best_metrics)
        left_layout.addWidget(best_wrap)

        splitter.addWidget(left)

        self.logs_panel = QFrame()
        self.logs_panel.setObjectName("LogsPanel")
        self.logs_panel.setMinimumWidth(320)
        logs_layout = QVBoxLayout(self.logs_panel)
        logs_layout.setContentsMargins(12, 12, 12, 12)
        logs_layout.setSpacing(10)

        logs_header = QHBoxLayout()
        logs_header.setSpacing(8)
        logs_title = QLabel("Logs")
        logs_title.setStyleSheet("font-size: 11pt; font-weight: 650;")
        logs_header.addWidget(logs_title)
        logs_header.addStretch(1)
        self.logs_close = QPushButton("Close")
        self.logs_close.clicked.connect(self._toggle_logs)
        logs_header.addWidget(self.logs_close)
        logs_layout.addLayout(logs_header)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setAcceptRichText(True)
        self.log_view.setFrameShape(QFrame.NoFrame)
        self.log_view.setStyleSheet(
            "background-color: #0b1327; border: 1px solid #1a2d55; border-radius: 12px; padding: 10px;"
        )
        logs_layout.addWidget(self.log_view, 1)

        splitter.addWidget(self.logs_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 0)

        layout.addWidget(splitter, 1)

        self._logs_visible = True
        self._toggle_logs(force_hide=True)

    @property
    def can_start(self) -> bool:
        return not self.is_running

    def start_training(self) -> None:
        if self.is_running:
            return

        self._clear_logs()
        self.progress.setValue(0)
        self.has_completed = False

        self._completed_models = 0
        self._results.clear()
        self._current_model = None
        self.best_model_name = None
        self._refresh_progress()
        self.best_name.setText("—")
        self.best_metrics.setText("R² = — | MAE = —")
        for card in self.model_cards.values():
            card.set_running(False)
            card.tile_r2.set_value("—")
            card.tile_mae.set_value("—")
            card.tile_rmse.set_value("—")
            card.time_label.setText("")

        self.is_running = True
        self.training_state_changed.emit()

        self._thread = QThread()
        self._worker = _TrainingWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.model_started.connect(self._on_model_started)
        self._worker.model_finished.connect(self._on_model_finished)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)

        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_finished(self) -> None:
        self.is_running = False
        self.has_completed = True
        self.training_state_changed.emit()
        self.training_completed.emit()

    def _on_model_started(self, name: str) -> None:
        self._current_model = name
        for m, card in self.model_cards.items():
            card.set_running(m == name)
        self._append_log("INFO", f"Started: {name}")

    def _on_model_finished(self, name: str, r2: float, mae: float, rmse: float, seconds: float) -> None:
        self._results[name] = (r2, mae, rmse, seconds)
        self._completed_models += 1

        card = self.model_cards.get(name)
        if card is not None:
            card.set_running(False)
            card.set_results(r2, mae, rmse, seconds)

        self._append_log("SUCCESS", f"Finished: {name} (R²={r2:.3f}, MAE={mae:.3f})")
        self._refresh_progress()
        self._refresh_best_model()

    def _refresh_progress(self) -> None:
        total = len(MODELS)
        pct = int((self._completed_models / max(total, 1)) * 100)
        self.progress.setValue(pct)
        self.completed_label.setText(f"{self._completed_models}/{total} complete")

    def _refresh_best_model(self) -> None:
        if not self._results:
            return

        best = max(self._results.items(), key=lambda kv: kv[1][0])
        name, (r2, mae, _rmse, _sec) = best
        self.best_name.setText(name)
        self.best_metrics.setText(f"R² = {r2:.3f} | MAE = {mae:.3f}")
        if self.best_model_name != name:
            self.best_model_name = name
            self.best_model_changed.emit(name)

    def _toggle_logs(self, force_hide: bool = False) -> None:
        if force_hide:
            self._logs_visible = False
        else:
            self._logs_visible = not self._logs_visible

        self.logs_panel.setVisible(self._logs_visible)
        self.logs_close.setVisible(self._logs_visible)

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
