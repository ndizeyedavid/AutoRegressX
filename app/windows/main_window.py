from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import shutil
import subprocess

from PySide6.QtCore import QTimer, QSize, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.windows.pages.configure_page import ConfigurePage
from app.windows.pages.data_import_page import DataImportPage
from app.windows.pages.export_page import ExportPage
from app.windows.pages.train_page import TrainPage

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
        ]

        self._current_step = 0
        self._completed_step = -1

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

        self._start_status_timer()

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

        title = QLabel("AutoRegressX")
        title.setObjectName("AppTitle")

        subtitle = QLabel("v1.0.0")
        subtitle.setStyleSheet("color: #9bb2db;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

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

        settings = QLabel("Settings")
        settings.setStyleSheet("color: #9bb2db;")
        help_ = QLabel("Help")
        help_.setStyleSheet("color: #9bb2db;")
        layout.addWidget(settings)
        layout.addWidget(help_)

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
        top_layout.addStretch(1)

        self.primary_button = QPushButton("Next")
        self.primary_button.setObjectName("PrimaryButton")
        self.primary_button.clicked.connect(self._on_primary_action)
        self.primary_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_layout.addWidget(self.primary_button)

        wrapper_layout.addWidget(self.top_bar)

        self.stack = QStackedWidget()

        self.page_data_import = DataImportPage()
        self.page_configure = ConfigurePage()
        self.page_train = TrainPage()
        self.page_export = ExportPage()

        self.stack.addWidget(self.page_data_import)
        self.stack.addWidget(self.page_configure)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_export)

        wrapper_layout.addWidget(self.stack, 1)

        self.status_bar = self._build_status_bar()
        wrapper_layout.addWidget(self.status_bar)

        return wrapper

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
            gpu_icon.setPixmap(qta.icon("fa5s.video", color="#9bb2db").pixmap(12, 12))
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
        self._status_timer.start(1500)
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

        self.status_gpu.setText(f"GPU: {self._get_gpu_usage_text()}")

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
        self.page_configure.target_changed.connect(self._refresh_navigation)

        self.page_train.training_state_changed.connect(self._refresh_navigation)
        self.page_train.training_completed.connect(self._on_training_completed)
        self.page_train.best_model_changed.connect(self._on_best_model_changed)

        self.page_export.export_state_changed.connect(self._refresh_navigation)

    def _on_dataset_loaded(self, filename: str, columns: list[str]) -> None:
        self.breadcrumb.setText(filename)
        self.page_configure.set_columns(columns)
        self._refresh_navigation()

    def _on_dataset_reset(self) -> None:
        self.breadcrumb.setText("No file loaded")
        self.page_configure.reset()

        self._current_step = 0
        self._completed_step = -1
        self.stack.setCurrentIndex(self._current_step)
        self._refresh_navigation()

    def _on_training_completed(self) -> None:
        self._completed_step = max(self._completed_step, 2)
        self.page_export.set_best_model(self.page_train.best_model_name)
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
            self.page_export.perform_export()

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
            self.primary_button.setText("Export")
            if qta is not None:
                self.primary_button.setIcon(qta.icon("fa5s.download", color="#021012"))
            self.primary_button.setEnabled(False)

    def _can_proceed_from_step(self, step_index: int) -> bool:
        if step_index == 0:
            return self.page_data_import.is_ready
        if step_index == 1:
            return self.page_configure.is_ready
        if step_index == 2:
            return self.page_train.can_start
        if step_index == 3:
            return True
        return False
