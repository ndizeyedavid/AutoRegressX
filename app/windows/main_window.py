from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
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


@dataclass(frozen=True)
class Step:
    title: str
    subtitle: str


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AutoRegressX")
        self.setMinimumSize(1200, 700)

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

        for idx, step in enumerate(self._steps):
            item = QListWidgetItem(f"{idx + 1}.  {step.title}\n{step.subtitle}")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
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

        return wrapper

    def _wire_pages(self) -> None:
        self.page_data_import.ready_changed.connect(self._refresh_navigation)
        self.page_data_import.dataset_loaded.connect(self._on_dataset_loaded)

        self.page_configure.ready_changed.connect(self._refresh_navigation)
        self.page_configure.target_changed.connect(self._refresh_navigation)

        self.page_train.training_state_changed.connect(self._refresh_navigation)
        self.page_train.training_completed.connect(self._on_training_completed)

        self.page_export.export_state_changed.connect(self._refresh_navigation)

    def _on_dataset_loaded(self, filename: str, columns: list[str]) -> None:
        self.breadcrumb.setText(filename)
        self.page_configure.set_columns(columns)
        self._refresh_navigation()

    def _on_training_completed(self) -> None:
        self._completed_step = max(self._completed_step, 2)
        self._refresh_navigation()

    def _on_primary_action(self) -> None:
        if self._current_step == 0:
            self._go_next()
            return
        if self._current_step == 1:
            self._go_next()
            return
        if self._current_step == 2:
            self.page_train.start_training()
            return
        if self._current_step == 3:
            self.page_export.perform_export()

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
            enabled = i <= self._completed_step + 1
            item.setFlags((item.flags() | Qt.ItemIsEnabled) if enabled else (item.flags() & ~Qt.ItemIsEnabled))
            if i == self._current_step:
                self.step_list.setCurrentRow(i)

        self.primary_button.setEnabled(can_proceed)

        if self._current_step in (0, 1):
            self.primary_button.setText("Next")
        elif self._current_step == 2:
            self.primary_button.setText("Run Training" if not self.page_train.is_running else "Training...")
            self.primary_button.setEnabled(can_proceed and not self.page_train.is_running)
        elif self._current_step == 3:
            self.primary_button.setText("Export")

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
