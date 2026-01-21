from __future__ import annotations

import json
import sys
import time
from datetime import datetime

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    training_canceled = Signal()
    best_model_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.is_running = False
        self.has_completed = False
        self._process: QProcess | None = None
        self._stdout_buf = ""
        self._run_dir: str | None = None

        self._completed_models = 0
        self._results: dict[str, tuple[float, float, float, float]] = {}
        self._current_model: str | None = None
        self.best_model_name: str | None = None

        self._csv_path: str | None = None
        self._dataset_name: str | None = None
        self._target_name: str | None = None
        self._auto_scroll = True
        self._log_items: list[tuple[str, str]] = []
        self._started_at: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        header = QLabel("Model Training")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        title_row.addWidget(header)
        self.stage_label = QLabel("Stage: —")
        self.stage_label.setStyleSheet("color: #9bb2db;")
        title_row.addWidget(self.stage_label)
        self.eta_label = QLabel("ETA: —")
        self.eta_label.setStyleSheet("color: #9bb2db;")
        title_row.addWidget(self.eta_label)
        title_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel")
        if qta is not None:
            self.cancel_btn.setIcon(qta.icon("fa5s.times", color="#e6eefc"))
        self.cancel_btn.clicked.connect(self.cancel_training)
        self.cancel_btn.setEnabled(False)
        title_row.addWidget(self.cancel_btn)

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

        self.logs_filter = QComboBox()
        self.logs_filter.addItems(["All", "Info", "Success", "Warn", "Error"])
        self.logs_filter.currentTextChanged.connect(self._rebuild_logs)
        logs_header.addWidget(self.logs_filter)

        self.autoscroll_chk = QCheckBox("Auto")
        self.autoscroll_chk.setChecked(True)
        self.autoscroll_chk.stateChanged.connect(self._on_autoscroll_changed)
        logs_header.addWidget(self.autoscroll_chk)

        self.copy_logs_btn = QPushButton("Copy")
        if qta is not None:
            self.copy_logs_btn.setIcon(qta.icon("fa5s.copy", color="#e6eefc"))
        self.copy_logs_btn.clicked.connect(self._copy_logs)
        logs_header.addWidget(self.copy_logs_btn)

        self.clear_logs_btn = QPushButton("Clear")
        if qta is not None:
            self.clear_logs_btn.setIcon(qta.icon("fa5s.trash", color="#e6eefc"))
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        logs_header.addWidget(self.clear_logs_btn)

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

        self._set_stage("Idle")
        self._set_eta(None)

    @property
    def can_start(self) -> bool:
        return (not self.is_running) and bool(self._csv_path) and bool(self._target_name)

    @property
    def run_dir(self) -> str | None:
        return self._run_dir

    def set_context(self, csv_path: str | None, dataset_name: str | None, target_name: str | None) -> None:
        self._csv_path = csv_path
        self._dataset_name = dataset_name
        self._target_name = target_name

    def start_training(self) -> None:
        if self.is_running:
            return

        if not self._csv_path or not self._target_name:
            return

        self._clear_logs()
        self.progress.setValue(0)
        self.has_completed = False
        self._run_dir = None

        self._started_at = time.perf_counter()
        self._set_stage("Starting")
        self._set_eta(len(MODELS))

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

        self.cancel_btn.setEnabled(True)

        self._stdout_buf = ""
        self._process = QProcess(self)
        self._process.setProgram(sys.executable)
        self._process.setArguments(
            [
                "-m",
                "app.ml.train_runner",
                "--csv",
                str(self._csv_path),
                "--target",
                str(self._target_name),
                "--seed",
                "42",
                "--test-size",
                "0.2",
            ]
        )

        self._process.readyReadStandardOutput.connect(self._on_process_stdout)
        self._process.readyReadStandardError.connect(self._on_process_stderr)
        self._process.finished.connect(self._on_process_finished)
        self._process.start()

    def cancel_training(self) -> None:
        if not self.is_running:
            return
        if self._process is not None:
            try:
                self._process.kill()
            except Exception:
                pass

    def _on_finished(self) -> None:
        self.is_running = False
        self.has_completed = True
        self.cancel_btn.setEnabled(False)
        self._process = None
        self._set_stage("Completed")
        self._set_eta(0)
        self.training_state_changed.emit()
        self.training_completed.emit()

    def _on_canceled(self) -> None:
        self.is_running = False
        self.has_completed = False
        self.cancel_btn.setEnabled(False)
        self._process = None
        self._set_stage("Canceled")
        self._set_eta(None)
        for card in self.model_cards.values():
            card.set_running(False)
        self.training_state_changed.emit()
        self.training_canceled.emit()

    def _on_process_stdout(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._stdout_buf += chunk

        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_event_line(line)

    def _on_process_stderr(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if not chunk:
            return
        for raw in chunk.splitlines():
            msg = raw.strip()
            if msg:
                self._append_log("ERROR", msg)

    def _handle_event_line(self, line: str) -> None:
        try:
            payload = json.loads(line)
        except Exception:
            self._append_log("INFO", line)
            return

        event = str(payload.get("event", "")).strip()
        if event == "log":
            self._append_log(str(payload.get("level", "INFO")), str(payload.get("message", "")))
            return

        if event == "run_started":
            self._run_dir = str(payload.get("run_dir", "")) or None
            if self._run_dir:
                self._append_log("INFO", f"Run directory: {self._run_dir}")
            return

        if event == "model_started":
            name = str(payload.get("name", ""))
            if name:
                self._on_model_started(name)
            return

        if event == "model_finished":
            name = str(payload.get("name", ""))
            if name:
                r2 = float(payload.get("r2", 0.0))
                mae = float(payload.get("mae", 0.0))
                rmse = float(payload.get("rmse", 0.0))
                seconds = float(payload.get("seconds", 0.0))
                self._on_model_finished(name, r2, mae, rmse, seconds)
            return

        if event == "run_finished":
            run_dir = str(payload.get("run_dir", ""))
            if run_dir:
                self._run_dir = run_dir
            best = str(payload.get("best_model", ""))
            if best:
                self.best_model_name = best
                self.best_model_changed.emit(best)
            self._on_finished()
            return

        if event == "error":
            self._append_log("ERROR", str(payload.get("message", "Training error")))
            self._on_canceled()
            return

        self._append_log("INFO", line)

    def _on_process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if not self.is_running:
            return
        # If the process exits without emitting run_finished, treat it as canceled.
        if self.has_completed:
            return
        if exit_code == 0:
            # Some outputs may have been buffered without newline.
            if self._stdout_buf.strip():
                self._handle_event_line(self._stdout_buf.strip())
                self._stdout_buf = ""
            if self.has_completed:
                return
        self._on_canceled()

    def _on_model_started(self, name: str) -> None:
        self._current_model = name
        self._set_stage(f"Training: {name}")
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
        self._set_eta(len(MODELS) - self._completed_models)

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

        self._log_items.append((lvl, html))
        self._rebuild_logs()

    def _rebuild_logs(self) -> None:
        want = self.logs_filter.currentText().strip().upper() if hasattr(self, "logs_filter") else "ALL"
        if want == "ALL":
            allowed = None
        else:
            allowed = {want}

        self.log_view.blockSignals(True)
        self.log_view.clear()
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)

        for lvl, html in self._log_items:
            if allowed is None or lvl in allowed:
                cursor.insertHtml(html)
                cursor.insertBlock()

        self.log_view.setTextCursor(cursor)
        if self._auto_scroll:
            self.log_view.ensureCursorVisible()
        self.log_view.blockSignals(False)

    def _on_autoscroll_changed(self, state: int) -> None:
        self._auto_scroll = state == Qt.Checked

    def _copy_logs(self) -> None:
        QApplication.clipboard().setText(self.log_view.toPlainText())

    def _clear_logs(self) -> None:
        self._log_items.clear()
        self.log_view.clear()

    def _set_stage(self, stage: str) -> None:
        self.stage_label.setText(f"Stage: {stage}")

    def _set_eta(self, remaining_models: int | None) -> None:
        if remaining_models is None:
            self.eta_label.setText("ETA: —")
            return
        if remaining_models <= 0:
            self.eta_label.setText("ETA: 0s")
            return
        if self._started_at is None or self._completed_models <= 0:
            self.eta_label.setText("ETA: estimating...")
            return
        elapsed = max(0.001, time.perf_counter() - self._started_at)
        per_model = elapsed / max(1, self._completed_models)
        eta = int(per_model * remaining_models)
        self.eta_label.setText(f"ETA: ~{eta}s")
