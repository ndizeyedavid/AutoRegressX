from __future__ import annotations

import platform
import sys

from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


REPO_URL = "https://github.com/ndizeyedavid/AutoRegressX"
DOCS_URL = "https://github.com/ndizeyedavid/AutoRegressX#readme"
ISSUES_URL = "https://github.com/ndizeyedavid/AutoRegressX/issues"


class HelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.setModal(True)
        self.setMinimumSize(720, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("Help & Support")
        title.setStyleSheet("font-size: 14pt; font-weight: 700;")
        title_row.addWidget(title)
        title_row.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        hint = QLabel("Quick guide, useful links, and diagnostics")
        hint.setStyleSheet("color: #9bb2db;")
        root.addWidget(hint)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.repo_btn = QPushButton("GitHub Repo")
        self.docs_btn = QPushButton("Documentation")
        self.bug_btn = QPushButton("Report a Bug")
        self.copy_btn = QPushButton("Copy System Info")

        if qta is not None:
            try:
                self.repo_btn.setIcon(qta.icon("fa5b.github", color="#e6eefc"))
                self.docs_btn.setIcon(qta.icon("fa5s.book", color="#e6eefc"))
                self.bug_btn.setIcon(qta.icon("fa5s.bug", color="#e6eefc"))
                self.copy_btn.setIcon(qta.icon("fa5s.copy", color="#e6eefc"))
            except Exception:
                pass

        self.repo_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(REPO_URL)))
        self.docs_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        self.bug_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ISSUES_URL)))
        self.copy_btn.clicked.connect(self._copy_system_info)

        actions.addWidget(self.repo_btn)
        actions.addWidget(self.docs_btn)
        actions.addWidget(self.bug_btn)
        actions.addStretch(1)
        actions.addWidget(self.copy_btn)
        root.addLayout(actions)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFrameShape(QFrame.NoFrame)
        self.text.setStyleSheet(
            "background-color: #0b1327; border: 1px solid #1a2d55; border-radius: 12px; padding: 12px;"
        )
        self.text.setHtml(
            """
            <h3>Workflow</h3>
            <ol>
              <li><b>Data Import</b>: drag & drop a CSV (or Browse Files). Use Reset to pick a different dataset.</li>
              <li><b>Configure</b>: select your target column (Auto-suggest helps).</li>
              <li><b>Train Models</b>: run training and compare models. Best model is chosen by highest R².</li>
              <li><b>Export</b>: download artifacts to a folder for backend deployment.</li>
            </ol>
            <h3>Troubleshooting</h3>
            <ul>
              <li>If GPU shows <b>—</b>, ensure NVIDIA drivers are installed and <code>nvidia-smi</code> is available.</li>
              <li>If the app fails to start, verify your virtual environment and dependencies.</li>
            </ul>
            """
        )
        root.addWidget(self.text, 1)

    def _copy_system_info(self) -> None:
        qs = QSettings()
        lines: list[str] = []
        lines.append(f"AutoRegressX v1.0.0")
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"Platform: {platform.platform()}")

        if psutil is not None:
            try:
                lines.append(f"CPU cores: {psutil.cpu_count(logical=True)}")
                vm = psutil.virtual_memory()
                lines.append(f"RAM total: {vm.total / (1024**3):.1f} GB")
            except Exception:
                pass

        lines.append(f"Status refresh (ms): {qs.value('ui/status_refresh_ms', '—')}")
        lines.append(f"Preview rows: {qs.value('data/preview_rows', '—')}")
        lines.append(f"Remember export dir: {qs.value('export/remember_last_dir', '—')}")

        QApplication.clipboard().setText("\n".join(lines))
