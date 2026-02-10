from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.windows.pages.predictions_page import _PlotCard

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


class _MetricTile(QFrame):
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


class ModelEvaluatePage(QWidget):
    evaluation_state_changed = Signal()
    evaluation_completed = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.is_running = False
        self._process: QProcess | None = None
        self._stdout_buf = ""

        self._model_dir: str | None = None
        self._csv_path: str | None = None
        self._run_dir: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QLabel("Evaluate Exported Model")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Import an AutoRegressX exported model folder and evaluate it on a CSV dataset")
        hint.setStyleSheet("color: #9bb2db;")
        layout.addWidget(header)
        layout.addWidget(hint)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        layout.addWidget(self.tabs, 1)

        self.tab_setup = QWidget()
        self.tab_results = QWidget()
        self.tab_charts = QWidget()
        self.tabs.addTab(self.tab_setup, "Setup")
        self.tabs.addTab(self.tab_results, "Results")
        self.tabs.addTab(self.tab_charts, "Charts")

        setup_layout = QVBoxLayout(self.tab_setup)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(14)

        results_layout = QVBoxLayout(self.tab_results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(14)

        charts_layout = QVBoxLayout(self.tab_charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(14)

        pick_card = QFrame()
        pick_card.setObjectName("Card")
        pick_layout = QGridLayout(pick_card)
        pick_layout.setContentsMargins(16, 14, 16, 14)
        pick_layout.setHorizontalSpacing(12)
        pick_layout.setVerticalSpacing(10)

        self.model_path_lbl = QLabel("Model folder: —")
        self.model_path_lbl.setStyleSheet("color: #9bb2db;")
        self.csv_path_lbl = QLabel("Dataset CSV: —")
        self.csv_path_lbl.setStyleSheet("color: #9bb2db;")

        self.pick_model_btn = QPushButton("Select Model Folder")
        self.pick_model_btn.clicked.connect(self._pick_model_dir)
        if qta is not None:
            self.pick_model_btn.setIcon(qta.icon("fa5s.folder-open", color="#e6eefc"))

        self.pick_csv_btn = QPushButton("Select Dataset CSV")
        self.pick_csv_btn.clicked.connect(self._pick_csv)
        if qta is not None:
            self.pick_csv_btn.setIcon(qta.icon("fa5s.file-csv", color="#e6eefc"))

        self.run_btn = QPushButton("Run Evaluation")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.start_evaluation)
        if qta is not None:
            self.run_btn.setIcon(qta.icon("fa5s.play", color="#021012"))

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_evaluation)
        self.cancel_btn.setEnabled(False)
        if qta is not None:
            self.cancel_btn.setIcon(qta.icon("fa5s.stop", color="#e6eefc"))

        self.export_btn = QPushButton("Export Evaluation")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        if qta is not None:
            self.export_btn.setIcon(qta.icon("fa5s.download", color="#e6eefc"))

        pick_layout.addWidget(self.model_path_lbl, 0, 0, 1, 2)
        pick_layout.addWidget(self.pick_model_btn, 0, 2)
        pick_layout.addWidget(self.csv_path_lbl, 1, 0, 1, 2)
        pick_layout.addWidget(self.pick_csv_btn, 1, 2)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.cancel_btn)
        actions.addStretch(1)
        actions.addWidget(self.export_btn)

        pick_layout.addLayout(actions, 2, 0, 1, 3)

        setup_layout.addWidget(pick_card)
        setup_layout.addStretch(1)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self.tile_r2 = _MetricTile("R²")
        self.tile_mae = _MetricTile("MAE")
        self.tile_rmse = _MetricTile("RMSE")
        self.tile_rows = _MetricTile("Rows Evaluated")
        for t in (self.tile_r2, self.tile_mae, self.tile_rmse, self.tile_rows):
            t.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            metrics_row.addWidget(t)
        results_layout.addLayout(metrics_row)

        mid = QHBoxLayout()
        mid.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(10)

        self.pred_table = QTableWidget(0, 0)
        self.pred_table.setAlternatingRowColors(True)
        self.pred_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pred_table.horizontalHeader().setStretchLastSection(True)

        left.addWidget(QLabel("Prediction Preview"))
        left.addWidget(self.pred_table, 1)

        right = QVBoxLayout()
        right.setSpacing(10)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setObjectName("TrainLog")

        right.addWidget(QLabel("Logs"))
        right.addWidget(self.logs, 1)

        mid.addLayout(left, 3)
        mid.addLayout(right, 2)
        results_layout.addLayout(mid, 1)

        self.charts_scroll = QScrollArea()
        self.charts_scroll.setWidgetResizable(True)
        self.charts_scroll.setFrameShape(QFrame.NoFrame)

        charts_container = QWidget()
        c_grid = QGridLayout(charts_container)
        c_grid.setContentsMargins(0, 0, 0, 0)
        c_grid.setHorizontalSpacing(12)
        c_grid.setVerticalSpacing(12)

        self.card_parity = _PlotCard("Parity Plot")
        self.card_residuals = _PlotCard("Residuals vs Predicted")
        self.card_resid_dist = _PlotCard("Residual Distribution")
        self.card_pred_dist = _PlotCard("Prediction Distribution")

        c_grid.addWidget(self.card_parity, 0, 0)
        c_grid.addWidget(self.card_residuals, 0, 1)
        c_grid.addWidget(self.card_resid_dist, 1, 0)
        c_grid.addWidget(self.card_pred_dist, 1, 1)
        c_grid.setColumnStretch(0, 1)
        c_grid.setColumnStretch(1, 1)

        self.charts_scroll.setWidget(charts_container)
        charts_layout.addWidget(self.charts_scroll, 1)

        self._reset_ui()

    def _append_log(self, level: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"{ts} [{level}] {msg}")
        self.logs.moveCursor(QTextCursor.End)

    def _reset_ui(self) -> None:
        self.tile_r2.set_value("—")
        self.tile_mae.set_value("—")
        self.tile_rmse.set_value("—")
        self.tile_rows.set_value("—")

        self.pred_table.clear()
        self.pred_table.setRowCount(0)
        self.pred_table.setColumnCount(0)

        for card in (self.card_parity, self.card_residuals, self.card_resid_dist, self.card_pred_dist):
            card.image.setText("Run evaluation to view charts")
            card.image.setStyleSheet("color: #9bb2db; padding: 18px;")
            card.set_pixmap(None)

        self.export_btn.setEnabled(False)

    def _pick_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select exported model folder", "")
        if path:
            self._model_dir = path
            self.model_path_lbl.setText(f"Model folder: {path}")
            self.evaluation_state_changed.emit()

    def _pick_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv);;All Files (*.*)")
        if path:
            self._csv_path = path
            self.csv_path_lbl.setText(f"Dataset CSV: {path}")
            self.evaluation_state_changed.emit()

    def start_evaluation(self) -> None:
        if self.is_running:
            return
        if not self._model_dir or not self._csv_path:
            return

        try:
            self.tabs.setCurrentWidget(self.tab_results)
        except Exception:
            pass

        self.logs.clear()
        self._reset_ui()
        self._run_dir = None

        self.is_running = True
        self.cancel_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.evaluation_state_changed.emit()

        self._stdout_buf = ""
        self._process = QProcess(self)
        self._process.setProgram(sys.executable)
        self._process.setArguments(
            [
                "-m",
                "app.ml.eval_runner",
                "--model-dir",
                str(self._model_dir),
                "--csv",
                str(self._csv_path),
                "--max-rows",
                "100",
            ]
        )
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.start()

    def cancel_evaluation(self) -> None:
        if not self.is_running:
            return
        if self._process is not None:
            try:
                self._process.kill()
            except Exception:
                pass

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._stdout_buf += chunk
        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            line = line.strip()
            if line:
                self._handle_event_line(line)

    def _on_stderr(self) -> None:
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

        if event == "run_finished":
            metrics = payload.get("metrics", {})
            rows = payload.get("n_rows")
            if isinstance(metrics, dict):
                r2 = metrics.get("r2")
                mae = metrics.get("mae")
                rmse = metrics.get("rmse")
                self.tile_r2.set_value("—" if r2 is None else f"{float(r2):.3f}")
                self.tile_mae.set_value("—" if mae is None else f"{float(mae):,.3f}")
                self.tile_rmse.set_value("—" if rmse is None else f"{float(rmse):,.3f}")
            if rows is not None:
                self.tile_rows.set_value(f"{int(rows):,}")
            self._load_results_views()
            return

        if event == "error":
            self._append_log("ERROR", str(payload.get("message", "Evaluation error")))
            return

        self._append_log("INFO", line)

    def _on_finished(self, _exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.is_running = False
        self.cancel_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self._process = None
        self.evaluation_state_changed.emit()

        if self._run_dir:
            self.export_btn.setEnabled(True)
            self.evaluation_completed.emit(self._run_dir)

    def _load_results_views(self) -> None:
        if not self._run_dir:
            return
        d = Path(self._run_dir)

        # Load preview table
        pred_path = d / "eval_predictions.csv"
        if pred_path.exists():
            try:
                df = pd.read_csv(pred_path)
            except Exception:
                df = None
            if df is not None:
                preview = df.head(20)
                self.pred_table.setColumnCount(len(preview.columns))
                self.pred_table.setRowCount(len(preview))
                self.pred_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
                for r in range(len(preview)):
                    for c, col in enumerate(preview.columns):
                        item = QTableWidgetItem(str(preview.iloc[r, c]))
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        self.pred_table.setItem(r, c, item)
                self.pred_table.resizeColumnsToContents()

        # Load charts
        plots_dir = d / "plots"
        mapping = {
            "parity.png": self.card_parity,
            "residuals.png": self.card_residuals,
            "residual_distribution.png": self.card_resid_dist,
            "pred_distribution.png": self.card_pred_dist,
        }
        for name, card in mapping.items():
            p = plots_dir / name
            if p.exists():
                from PySide6.QtGui import QPixmap

                pix = QPixmap(str(p))
                card.image.setStyleSheet("")
                card.set_pixmap(pix)
            else:
                card.image.setText(f"Missing plot: {name}")
                card.image.setStyleSheet("color: #9bb2db; padding: 18px;")
                card.set_pixmap(None)

    def load_run_dir(self, run_dir: str) -> None:
        d = Path(run_dir).expanduser().resolve()
        if not d.exists() or not d.is_dir():
            return

        if self.is_running:
            try:
                self.cancel_evaluation()
            except Exception:
                pass

        self.is_running = False
        self._process = None
        self._stdout_buf = ""
        self._run_dir = str(d)

        # Load summary
        summary_path = d / "eval_metrics.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

            try:
                self._model_dir = summary.get("model_dir")
                self._csv_path = summary.get("csv_path")
                self.model_path_lbl.setText(f"Model folder: {self._model_dir}" if self._model_dir else "Model folder: —")
                self.csv_path_lbl.setText(f"Dataset CSV: {self._csv_path}" if self._csv_path else "Dataset CSV: —")
            except Exception:
                pass

            metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
            r2 = metrics.get("r2")
            mae = metrics.get("mae")
            rmse = metrics.get("rmse")
            self.tile_r2.set_value("—" if r2 is None else f"{float(r2):.3f}")
            self.tile_mae.set_value("—" if mae is None else f"{float(mae):,.3f}")
            self.tile_rmse.set_value("—" if rmse is None else f"{float(rmse):,.3f}")

            rows = summary.get("n_rows")
            if rows is not None:
                try:
                    self.tile_rows.set_value(f"{int(rows):,}")
                except Exception:
                    pass

        self._load_results_views()
        self.export_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.evaluation_state_changed.emit()

    def export_results(self) -> None:
        if not self._run_dir:
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Select export directory", "")
        if not dest_root:
            return

        src = Path(self._run_dir)
        dest = Path(dest_root) / f"AutoRegressX_eval_export_{src.name}"
        try:
            import shutil

            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        except Exception:
            return

        self._append_log("SUCCESS", f"Exported evaluation to {dest}")

    def reset(self) -> None:
        if self.is_running:
            try:
                self.cancel_evaluation()
            except Exception:
                pass

        self.is_running = False
        self._process = None
        self._stdout_buf = ""

        self._model_dir = None
        self._csv_path = None
        self._run_dir = None

        self.model_path_lbl.setText("Model folder: —")
        self.csv_path_lbl.setText("Dataset CSV: —")
        self.logs.clear()
        self._reset_ui()

        self.cancel_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.evaluation_state_changed.emit()
