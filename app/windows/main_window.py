from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

from PySide6.QtCore import QSettings, QTimer, QSize, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSystemTrayIcon,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.windows.pages.configure_page import ConfigurePage
from app.windows.pages.data_import_page import DataImportPage
from app.windows.pages.export_page import ExportPage
from app.windows.pages.model_evaluate_page import ModelEvaluatePage
from app.windows.pages.predictions_page import PredictionsPage
from app.windows.pages.train_page import TrainPage
from app.windows.dialogs.help_dialog import HelpDialog
from app.windows.dialogs.settings_dialog import AppSettings, SettingsDialog, load_settings
from app.widgets.toast import ToastHost
from app.widgets.validation_banner import ValidationBanner

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _qta_icon(name: str, color: str) -> QIcon | None:
    if qta is None:
        return None
    return qta.icon(name, color=color)


@dataclass(frozen=True)
class Step:
    title: str
    subtitle: str


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AutoRegressX")
        self.setMinimumSize(1200, 700)

        app_icon = _qta_icon("fa5s.chart-line", "#0ea5a4")
        if app_icon is not None:
            self.setWindowIcon(app_icon)

        self._steps: list[Step] = [
            Step("Data Import", "Load CSV dataset"),
            Step("Configure", "Select target variable"),
            Step("Train Models", "Run algorithms"),
            Step("Export", "Save artifacts"),
            Step("Predictions", "Review charts"),
        ]

        self._current_step = 0
        self._completed_step = -1
        self._csv_path: str | None = None
        self._mode: str = "training"

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._sidebar = self._build_sidebar()
        self._content = self._build_content()

        root_layout.addWidget(self._sidebar)
        root_layout.addWidget(self._content, 1)

        self.setCentralWidget(root)

        self._wire_pages()
        self._refresh_navigation()

        self._qs = QSettings()
        self._app_settings = load_settings(self._qs)
        self._apply_settings(self._app_settings)

        self._init_notifications()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)
        quit_action.setShortcut("Ctrl+Q")

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        header = QWidget()
        header.setObjectName("SidebarHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        logo = QLabel()
        logo.setObjectName("SidebarLogo")
        logo.setFixedSize(42, 42)
        logo.setScaledContents(True)

        logo_path = Path(__file__).resolve().parents[1] / "assets" / "logo.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                logo.setPixmap(pix.scaled(42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            if qta is not None:
                logo.setPixmap(qta.icon("fa5s.chart-line", color="#0ea5a4").pixmap(34, 34))

        name_wrap = QVBoxLayout()
        name_wrap.setContentsMargins(0, 0, 0, 0)
        name_wrap.setSpacing(2)

        title = QLabel("AutoRegressX")
        title.setObjectName("AppTitle")

        subtitle = QLabel("v1.0.0")
        subtitle.setObjectName("AppVersion")

        name_wrap.addWidget(title)
        name_wrap.addWidget(subtitle)

        header_layout.addWidget(logo, 0, Qt.AlignLeft | Qt.AlignTop)
        header_layout.addLayout(name_wrap, 1)

        layout.addWidget(header)

        layout.addSpacing(8)

        workflow_label = QLabel("WORKFLOW")
        workflow_label.setStyleSheet("color: #6f86b6; font-weight: 600;")
        layout.addWidget(workflow_label)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepList")
        self.step_list.setFocusPolicy(Qt.NoFocus)
        self.step_list.setSpacing(2)
        self.step_list.itemClicked.connect(self._on_step_clicked)
        self.step_list.setIconSize(QSize(30, 30))

        for idx, step in enumerate(self._steps):
            item = QListWidgetItem(f"{step.title}\n{step.subtitle}")
            item.setData(Qt.UserRole, idx)
            self.step_list.addItem(item)

        layout.addWidget(self.step_list, 1)

        layout.addSpacing(8)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("SidebarLink")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setFlat(True)
        self.settings_btn.setStyleSheet("text-align:left; padding: 8px 10px; color: #9bb2db;")
        if qta is not None:
            self.settings_btn.setIcon(qta.icon("fa5s.cog", color="#9bb2db"))
        self.settings_btn.clicked.connect(self._open_settings)

        self.help_btn = QPushButton("Help")
        self.help_btn.setObjectName("SidebarLink")
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setFlat(True)
        self.help_btn.setStyleSheet("text-align:left; padding: 8px 10px; color: #9bb2db;")
        if qta is not None:
            self.help_btn.setIcon(qta.icon("fa5s.question-circle", color="#9bb2db"))
        self.help_btn.clicked.connect(self._open_help)

        layout.addWidget(self.settings_btn)
        layout.addWidget(self.help_btn)

        self.restart_btn = QPushButton("Restart")
        self.restart_btn.setObjectName("SidebarLink")
        self.restart_btn.setCursor(Qt.PointingHandCursor)
        self.restart_btn.setFlat(True)
        self.restart_btn.setStyleSheet("text-align:left; padding: 8px 10px; color: #9bb2db;")
        if qta is not None:
            self.restart_btn.setIcon(qta.icon("fa5s.redo", color="#9bb2db"))
        self.restart_btn.clicked.connect(self._restart_workflow)

        layout.addWidget(self.restart_btn)

        return sidebar

    def _build_content(self) -> QWidget:
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(12)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_back)
        if qta is not None:
            self.back_button.setIcon(qta.icon("fa5s.arrow-left", color="#e6eefc"))
        top_layout.addWidget(self.back_button)

        self.breadcrumb = QLabel("No file loaded")
        self.breadcrumb.setStyleSheet("color: #9bb2db;")

        top_layout.addWidget(self.breadcrumb)

        # Top-level mode tabs (centered)
        top_layout.addStretch(1)
        self.mode_tabs = QTabBar()
        self.mode_tabs.setObjectName("ModeTabs")
        self.mode_tabs.setDrawBase(False)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.addTab("Training")
        self.mode_tabs.addTab("Evaluate Model")
        self.mode_tabs.setCurrentIndex(0)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)
        top_layout.addWidget(self.mode_tabs)
        top_layout.addStretch(1)

        self.primary_button = QPushButton("Next")
        self.primary_button.setObjectName("PrimaryButton")
        self.primary_button.clicked.connect(self._on_primary_action)
        self.primary_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_layout.addWidget(self.primary_button)

        wrapper_layout.addWidget(self.top_bar)

        self.validation_banner = ValidationBanner()
        self.validation_banner.action_clicked.connect(self._on_validation_action)

        # Wizard container (training flow)
        self.wizard_container = QWidget()
        wizard_layout = QVBoxLayout(self.wizard_container)
        wizard_layout.setContentsMargins(0, 0, 0, 0)
        wizard_layout.setSpacing(0)
        wizard_layout.addWidget(self.validation_banner)

        self.stack = QStackedWidget()

        self.page_data_import = DataImportPage()
        self.page_configure = ConfigurePage()
        self.page_train = TrainPage()
        self.page_export = ExportPage()
        self.page_predictions = PredictionsPage()

        self.stack.addWidget(self.page_data_import)
        self.stack.addWidget(self.page_configure)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_export)
        self.stack.addWidget(self.page_predictions)
        wizard_layout.addWidget(self.stack, 1)

        # Evaluate container
        self.page_model_evaluate = ModelEvaluatePage()

        # Mode stack
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.wizard_container)
        self.mode_stack.addWidget(self.page_model_evaluate)
        wrapper_layout.addWidget(self.mode_stack, 1)

        self.status_bar = self._build_status_bar()
        wrapper_layout.addWidget(self.status_bar)

        self.toast_host = ToastHost(wrapper)
        self.toast_host.raise_()

        self._apply_mode_ui()
        return wrapper

    def _on_mode_changed(self, idx: int) -> None:
        self._mode = "evaluate" if idx == 1 else "training"
        self._apply_mode_ui()

    def _apply_mode_ui(self) -> None:
        is_training = self._mode == "training"
        try:
            self.mode_stack.setCurrentIndex(0 if is_training else 1)
        except Exception:
            pass

        # Toggle wizard navigation controls
        self.step_list.setVisible(is_training)
        self.back_button.setVisible(is_training)
        self.breadcrumb.setVisible(is_training)
        self.primary_button.setVisible(is_training)
        self.validation_banner.setVisible(is_training)

        if is_training:
            self._refresh_navigation()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "toast_host"):
            w = self._content.width()
            h = self._content.height()

            # Keep toasts bottom-right above the status bar
            bottom_reserved = self.status_bar.height() if hasattr(self, "status_bar") else 0
            margin = 14
            host_w = min(480, w)
            host_h = max(160, min(320, h))
            self.toast_host.setGeometry(
                max(0, w - host_w - margin),
                max(0, h - host_h - bottom_reserved - margin),
                host_w,
                host_h,
            )

    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("StatusBar")
        bar.setFixedHeight(28)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(14)

        self.status_cpu = QLabel("CPU: —")
        self.status_cpu.setObjectName("StatusText")
        self.status_mem = QLabel("Memory: —")
        self.status_mem.setObjectName("StatusText")
        self.status_gpu = QLabel("GPU: —")
        self.status_gpu.setObjectName("StatusText")

        if qta is not None:
            cpu_icon = QLabel()
            cpu_icon.setPixmap(qta.icon("fa5s.microchip", color="#9bb2db").pixmap(12, 12))
            mem_icon = QLabel()
            mem_icon.setPixmap(qta.icon("fa5s.memory", color="#9bb2db").pixmap(12, 12))
            gpu_icon = QLabel()
            gpu_icon.setPixmap(qta.icon("fa5s.hdd", color="#9bb2db").pixmap(12, 12))
        else:
            cpu_icon = None
            mem_icon = None
            gpu_icon = None

        ready_container = QWidget()
        ready_wrap = QHBoxLayout(ready_container)
        ready_wrap.setContentsMargins(0, 0, 0, 0)
        ready_wrap.setSpacing(6)

        if qta is not None:
            ready_icon = QLabel()
            ready_icon.setPixmap(qta.icon("fa5s.check", color="#27d7a3").pixmap(12, 12))
            ready_wrap.addWidget(ready_icon)

        ready_text = QLabel("Ready")
        ready_text.setObjectName("StatusText")
        ready_wrap.addWidget(ready_text)

        layout.addWidget(ready_container)
        if cpu_icon is not None:
            layout.addWidget(cpu_icon)
        layout.addWidget(self.status_cpu)
        if mem_icon is not None:
            layout.addWidget(mem_icon)
        layout.addWidget(self.status_mem)
        if gpu_icon is not None:
            layout.addWidget(gpu_icon)
        layout.addWidget(self.status_gpu)

        layout.addStretch(1)

        self.status_right = QLabel("AutoRegressX v1.0.0")
        self.status_right.setObjectName("StatusText")

        if qta is not None:
            time_icon = QLabel()
            time_icon.setPixmap(qta.icon("fa5s.clock", color="#9bb2db").pixmap(12, 12))
        else:
            time_icon = None

        self.status_time = QLabel("—")
        self.status_time.setObjectName("StatusText")
        layout.addWidget(self.status_right)
        if time_icon is not None:
            layout.addWidget(time_icon)
        layout.addWidget(self.status_time)

        return bar

    def _start_status_timer(self) -> None:
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(int(self._app_settings.status_refresh_ms))
        self._update_status()

    def _update_status(self) -> None:
        self.status_time.setText(datetime.now().strftime("%H:%M"))

        if psutil is None:
            self.status_cpu.setText("CPU: —")
            self.status_mem.setText("Memory: —")
            self.status_gpu.setText("GPU: —")
            return

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_mb = mem.used / (1024 * 1024)
        self.status_cpu.setText(f"CPU: {cpu:.0f}%")
        self.status_mem.setText(f"Memory: {mem_mb:.0f} MB")

        if self._app_settings.show_gpu:
            self.status_gpu.setText(f"GPU: {self._get_gpu_usage_text()}")
        else:
            self.status_gpu.setText("GPU: —")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.settings_applied.connect(self._apply_settings)
        dlg.exec()

    def _open_help(self) -> None:
        dlg = HelpDialog(self)
        dlg.exec()

    def _apply_settings(self, s: AppSettings) -> None:
        self._app_settings = s

        # Apply: Data preview rows
        try:
            self.page_data_import.set_preview_rows(int(s.preview_rows))
        except Exception:
            pass

        # Apply: Export preferences
        try:
            self.page_export.set_export_preferences(
                remember_last_dir=bool(s.remember_last_export_dir),
                last_dir=str(s.last_export_dir),
            )
        except Exception:
            pass

        # Apply: Status timer interval
        if hasattr(self, "_status_timer"):
            self._status_timer.setInterval(int(s.status_refresh_ms))
        else:
            self._start_status_timer()

        # Apply: GPU visibility
        self.status_gpu.setVisible(bool(s.show_gpu))

        self.notify("info", "Settings", "Settings applied", desktop=False)

    def _get_gpu_usage_text(self) -> str:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return "—"

        try:
            out = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.8,
            ).strip()
        except Exception:
            return "—"

        if not out:
            return "—"

        # Take first GPU line
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            util, used, total = parts[0], parts[1], parts[2]
            return f"{util}% ({used}/{total} MB)"
        if len(parts) == 1:
            return f"{parts[0]}%"
        return "—"

    def _wire_pages(self) -> None:
        self.page_data_import.ready_changed.connect(self._refresh_navigation)
        self.page_data_import.dataset_loaded.connect(self._on_dataset_loaded)
        self.page_data_import.dataset_reset.connect(self._on_dataset_reset)

        self.page_configure.ready_changed.connect(self._refresh_navigation)
        self.page_configure.target_changed.connect(self._on_target_changed)

        self.page_train.training_state_changed.connect(self._refresh_navigation)
        self.page_train.training_completed.connect(self._on_training_completed)
        self.page_train.training_canceled.connect(self._on_training_canceled)
        self.page_train.best_model_changed.connect(self._on_best_model_changed)

        self.page_export.export_state_changed.connect(self._refresh_navigation)
        self.page_export.export_completed.connect(self._on_export_completed)
        self.page_export.export_path_copied.connect(self._on_export_path_copied)

        try:
            self.page_model_evaluate.evaluation_completed.connect(self._on_evaluation_completed)
        except Exception:
            pass

    def _on_evaluation_completed(self, run_dir: str) -> None:
        self.notify("success", "Evaluation complete", f"Saved to: {run_dir}", desktop=False)

    def _on_export_completed(self, path: str) -> None:
        self.notify("success", "Export complete", f"Saved to: {path}", desktop=False)
        self._completed_step = max(self._completed_step, 3)
        try:
            self.page_predictions.set_export_dir(path)
        except Exception:
            pass
        self._refresh_navigation()

    def _on_export_path_copied(self, path: str) -> None:
        self.notify("info", "Copied", "Export path copied to clipboard", desktop=False)

    def _restart_workflow(self) -> None:
        try:
            if getattr(self.page_train, "is_running", False):
                self.page_train.cancel_training()
        except Exception:
            pass

        self._csv_path = None
        self.breadcrumb.setText("No file loaded")

        try:
            self.page_data_import.reset()
        except Exception:
            pass
        try:
            self.page_configure.reset()
        except Exception:
            pass
        try:
            self.page_train.reset()
        except Exception:
            pass
        try:
            self.page_export.reset()
        except Exception:
            pass
        try:
            self.page_predictions.reset()
        except Exception:
            pass

        try:
            self.page_model_evaluate.reset()
        except Exception:
            pass

        try:
            self.mode_tabs.setCurrentIndex(0)
        except Exception:
            pass

        self._current_step = 0
        self._completed_step = -1
        self.stack.setCurrentIndex(self._current_step)
        self.notify("info", "Restart", "Started a new run", desktop=False)
        self._refresh_navigation()

    def _on_dataset_loaded(self, csv_path: str, filename: str, columns: list[str]) -> None:
        self._csv_path = csv_path
        self.breadcrumb.setText(filename)
        self.page_configure.set_columns(columns)
        self.page_train.set_context(csv_path, filename, self.page_configure.selected_target())
        self.notify("success", "Dataset loaded", f"{filename} is ready", desktop=True)
        self._refresh_navigation()

    def _on_dataset_reset(self) -> None:
        self.breadcrumb.setText("No file loaded")
        self._csv_path = None
        self.page_configure.reset()
        self.page_train.set_context(None, None, None)

        self._current_step = 0
        self._completed_step = -1
        self.stack.setCurrentIndex(self._current_step)
        self.notify("info", "Reset", "Dataset cleared", desktop=False)
        self._refresh_navigation()

    def _on_target_changed(self, _value: str) -> None:
        self.page_train.set_context(self._csv_path, self.breadcrumb.text(), self.page_configure.selected_target())
        self._refresh_navigation()

    def _on_training_completed(self) -> None:
        self.notify("success", "Training completed", "Best model selected", desktop=True)
        self._completed_step = max(self._completed_step, 2)
        self.page_export.set_best_model(self.page_train.best_model_name)
        self.page_export.set_run_dir(self.page_train.run_dir)
        self._refresh_navigation()

    def _on_training_canceled(self) -> None:
        self.notify("warn", "Training canceled", "You can adjust settings and run again", desktop=False)
        self._refresh_navigation()

    def _on_best_model_changed(self, model_name: str) -> None:
        self.page_export.set_best_model(model_name)

    def _on_primary_action(self) -> None:
        if self._current_step == 0:
            self._go_next()
            return
        if self._current_step == 1:
            self._go_next()
            return
        if self._current_step == 2:
            if self.page_train.has_completed:
                self._go_next()
            else:
                self.page_train.start_training()
            return
        if self._current_step == 3:
            if self.page_export.exported_dir():
                self._go_next()
            else:
                self.page_export.perform_export()
            return
        if self._current_step == 4:
            return

    def _on_step_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if not isinstance(idx, int):
            return

        if idx == self._current_step:
            return

        if idx <= self._completed_step:
            self._current_step = idx
            self.stack.setCurrentIndex(self._current_step)
            self._refresh_navigation()

    def _go_back(self) -> None:
        if self._current_step <= 0:
            return
        self._current_step -= 1
        self.stack.setCurrentIndex(self._current_step)
        self._refresh_navigation()

    def _go_next(self) -> None:
        if self._current_step < len(self._steps) - 1:
            self._completed_step = max(self._completed_step, self._current_step)
            self._current_step += 1
            self.stack.setCurrentIndex(self._current_step)
            self._refresh_navigation()

    def _refresh_navigation(self) -> None:
        if getattr(self, "_mode", "training") != "training":
            return

        can_proceed = self._can_proceed_from_step(self._current_step)

        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            enabled = (i == self._current_step) or (i <= self._completed_step)
            flags = item.flags()
            if enabled:
                flags |= Qt.ItemIsEnabled | Qt.ItemIsSelectable
            else:
                flags &= ~Qt.ItemIsEnabled
                flags &= ~Qt.ItemIsSelectable
            item.setFlags(flags)

            icon: QIcon | None = None
            if qta is not None:
                if i <= self._completed_step:
                    icon = qta.icon("fa5s.check-circle", color="#27d7a3")
                else:
                    icon = qta.icon("fa5s.circle", color="#4b5b79")
            if icon is not None:
                item.setIcon(icon)

            if i == self._current_step:
                self.step_list.setCurrentRow(i)

        self.primary_button.setEnabled(can_proceed)

        self._refresh_validation_banner(can_proceed)

        self.back_button.setEnabled(self._current_step > 0)

        if self._current_step in (0, 1):
            self.primary_button.setText("Next")
            if qta is not None:
                self.primary_button.setIcon(qta.icon("fa5s.arrow-right", color="#021012"))
        elif self._current_step == 2:
            if self.page_train.is_running:
                self.primary_button.setText("Training...")
                self.primary_button.setEnabled(False)
                if qta is not None:
                    self.primary_button.setIcon(qta.icon("fa5s.spinner", color="#021012"))
            elif self.page_train.has_completed:
                self.primary_button.setText("Next")
                self.primary_button.setEnabled(True)
                if qta is not None:
                    self.primary_button.setIcon(qta.icon("fa5s.arrow-right", color="#021012"))
            else:
                self.primary_button.setText("Run Training")
                self.primary_button.setEnabled(can_proceed)
                if qta is not None:
                    self.primary_button.setIcon(qta.icon("fa5s.play", color="#021012"))
        elif self._current_step == 3:
            if self.page_export.exported_dir():
                self.primary_button.setText("Next")
                self.primary_button.setEnabled(True)
                if qta is not None:
                    self.primary_button.setIcon(qta.icon("fa5s.arrow-right", color="#021012"))
            else:
                self.primary_button.setText("Export")
                self.primary_button.setEnabled(can_proceed)
                if qta is not None:
                    self.primary_button.setIcon(qta.icon("fa5s.download", color="#021012"))
        elif self._current_step == 4:
            self.primary_button.setText("Done")
            self.primary_button.setEnabled(False)
            if qta is not None:
                self.primary_button.setIcon(qta.icon("fa5s.check", color="#021012"))


    def _refresh_validation_banner(self, can_proceed: bool) -> None:
        # Show a helpful banner if the primary action is blocked.
        if can_proceed:
            self.validation_banner.set_message("info", "", None)
            return

        if self._current_step == 0:
            self.validation_banner.set_message(
                "warn",
                "Load a CSV dataset to continue.",
                None,
            )
        elif self._current_step == 1:
            self.validation_banner.set_message(
                "warn",
                "Select a target column to continue.",
                None,
            )
        elif self._current_step == 2:
            if getattr(self.page_train, "is_running", False):
                self.validation_banner.set_message(
                    "info",
                    "Training is currently running.",
                    None,
                )
            else:
                self.validation_banner.set_message(
                    "warn",
                    "Run training to continue.",
                    "Run",
                )
        elif self._current_step == 3:
            self.validation_banner.set_message(
                "warn",
                "Export artifacts to continue.",
                "Export",
            )
        elif self._current_step == 4:
            self.validation_banner.set_message("info", "", None)
        else:
            self.validation_banner.set_message("warn", "", None)

    def _on_validation_action(self) -> None:
        if self._current_step == 0:
            try:
                self.page_data_import.drop_zone.browse_clicked.emit()
            except Exception:
                pass
        elif self._current_step == 2:
            try:
                self.page_train.start_training()
            except Exception:
                pass
        elif self._current_step == 3:
            try:
                self.page_export.perform_export()
            except Exception:
                pass

    def _init_notifications(self) -> None:
        self._tray: QSystemTrayIcon | None = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = QSystemTrayIcon(self)
            if not self.windowIcon().isNull():
                self._tray.setIcon(self.windowIcon())
            self._tray.setVisible(True)

    def notify(self, level: str, title: str, message: str, desktop: bool = True) -> None:
        if hasattr(self, "toast_host"):
            self.toast_host.show_toast(level, title, message)

        if desktop and self._tray is not None:
            try:
                self._tray.showMessage(title, message)
            except Exception:
                pass


    def _can_proceed_from_step(self, step_index: int) -> bool:
        if step_index == 0:
            return self.page_data_import.is_ready
        if step_index == 1:
            return self.page_configure.is_ready
        if step_index == 2:
            return self.page_train.can_start
        if step_index == 3:
            return bool(self.page_export.exported_dir())
        if step_index == 4:
            return False
        return False
