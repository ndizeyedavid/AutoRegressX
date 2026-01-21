from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.widgets.drop_zone import DropZone

try:
    import qtawesome as qta
except Exception:  # pragma: no cover
    qta = None


class DataImportPage(QWidget):
    ready_changed = Signal()
    dataset_loaded = Signal(str, list)

    def __init__(self) -> None:
        super().__init__()

        self._csv_path: Path | None = None
        self._df: pd.DataFrame | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(14)

        header = QLabel("Data Import")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Load a CSV dataset to begin model training")
        hint.setStyleSheet("color: #9bb2db;")

        left.addWidget(header)
        left.addWidget(hint)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._load_csv)

        browse = QPushButton("Browse Files")
        browse.clicked.connect(self._browse)
        browse.setFixedWidth(140)
        if qta is not None:
            browse.setIcon(qta.icon("fa5s.folder-open", color="#e6eefc"))

        dz_wrap = QFrame()
        dz_wrap.setObjectName("Card")
        dz_layout = QVBoxLayout(dz_wrap)
        dz_layout.setContentsMargins(16, 16, 16, 16)
        dz_layout.setSpacing(12)
        dz_layout.addWidget(self.drop_zone)
        dz_layout.addWidget(browse, alignment=Qt.AlignHCenter)

        left.addWidget(dz_wrap, 1)

        preview_group = QGroupBox("Preview (first 10 rows)")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 14, 10, 10)

        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setStretchLastSection(True)

        preview_layout.addWidget(self.preview_table)
        left.addWidget(preview_group, 1)

        layout.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(12)

        info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(10, 14, 10, 10)

        self.rows_label = QLabel("Rows: —")
        self.cols_label = QLabel("Columns: —")
        self.target_label = QLabel("Target: —")
        for w in (self.rows_label, self.cols_label, self.target_label):
            w.setStyleSheet("color: #9bb2db;")

        info_layout.addWidget(self.rows_label)
        info_layout.addWidget(self.cols_label)
        info_layout.addWidget(self.target_label)
        right.addWidget(info_group)

        prep_group = QGroupBox("Auto Preprocessing")
        prep_layout = QVBoxLayout(prep_group)
        prep_layout.setContentsMargins(10, 14, 10, 10)

        for txt in (
            "Missing value imputation",
            "Categorical encoding",
            "Feature scaling",
            "Train/test split (80/20)",
        ):
            lbl = QLabel(f"- {txt}")
            lbl.setStyleSheet("color: #9bb2db;")
            prep_layout.addWidget(lbl)

        right.addWidget(prep_group)

        alg_group = QGroupBox("Algorithms")
        alg_layout = QVBoxLayout(alg_group)
        alg_layout.setContentsMargins(10, 14, 10, 10)

        for txt in (
            "Linear Regression",
            "Ridge Regression",
            "Random Forest",
            "Support Vector Regression",
            "K-Nearest Neighbors",
        ):
            lbl = QLabel(f"- {txt}")
            lbl.setStyleSheet("color: #9bb2db;")
            alg_layout.addWidget(lbl)

        right.addWidget(alg_group)
        right.addStretch(1)

        layout.addLayout(right, 1)

    @property
    def is_ready(self) -> bool:
        return self._df is not None

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            "",
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if path:
            self._load_csv(path)

    def _load_csv(self, file_path: str) -> None:
        p = Path(file_path)
        if not p.exists() or p.suffix.lower() != ".csv":
            return

        try:
            df = pd.read_csv(p)
        except Exception:
            return

        self._csv_path = p
        self._df = df

        self.rows_label.setText(f"Rows: {len(df):,}")
        self.cols_label.setText(f"Columns: {len(df.columns):,}")

        self._populate_preview(df)

        self.dataset_loaded.emit(p.name, list(map(str, df.columns.tolist())))
        self.ready_changed.emit()

    def _populate_preview(self, df: pd.DataFrame) -> None:
        preview = df.head(10)
        self.preview_table.clear()
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in preview.columns])

        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                value = preview.iloc[r, c]
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(r, c, item)

        self.preview_table.resizeColumnsToContents()
