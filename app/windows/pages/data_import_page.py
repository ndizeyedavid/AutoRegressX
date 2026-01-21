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
    QSizePolicy,
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


PREVIEW_ROWS = 15


class DataImportPage(QWidget):
    ready_changed = Signal()
    dataset_loaded = Signal(str, list)
    dataset_reset = Signal()

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

        self.import_card = QFrame()
        self.import_card.setObjectName("Card")
        self.import_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        dz_layout = QVBoxLayout(self.import_card)
        dz_layout.setContentsMargins(18, 18, 18, 18)
        dz_layout.setSpacing(0)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._load_csv)
        self.drop_zone.browse_clicked.connect(self._browse)
        dz_layout.addWidget(self.drop_zone)

        left.addWidget(self.import_card, 3)

        self.preview_group = QFrame()
        self.preview_group.setObjectName("Card")
        self.preview_group.setVisible(False)

        pg_layout = QVBoxLayout(self.preview_group)
        pg_layout.setContentsMargins(18, 16, 18, 16)
        pg_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        title = QLabel("Data Preview")
        title.setStyleSheet("font-size: 12.5pt; font-weight: 650;")

        self.reset_btn = QPushButton("Reset")
        if qta is not None:
            self.reset_btn.setIcon(qta.icon("fa5s.undo", color="#e6eefc"))
        self.reset_btn.clicked.connect(self.reset)

        self.rows_cols_badge = QLabel("—")
        self.rows_cols_badge.setStyleSheet("color: #9bb2db;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(self.reset_btn)
        header_row.addWidget(self.rows_cols_badge)
        pg_layout.addLayout(header_row)

        self.preview_table = QTableWidget(0, 0)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        pg_layout.addWidget(self.preview_table, 1)

        legend = QHBoxLayout()
        legend.setSpacing(14)
        numeric = QLabel("#  Numeric")
        numeric.setStyleSheet("color: #27d7a3;")
        categorical = QLabel("T  Categorical")
        categorical.setStyleSheet("color: #f59e0b;")
        legend.addWidget(numeric)
        legend.addWidget(categorical)
        legend.addStretch(1)
        pg_layout.addLayout(legend)

        left.addWidget(self.preview_group, 3)

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
        self.import_card.setVisible(False)
        self.preview_group.setVisible(True)

        self.dataset_loaded.emit(p.name, list(map(str, df.columns.tolist())))
        self.ready_changed.emit()

    def reset(self) -> None:
        self._csv_path = None
        self._df = None

        self.import_card.setVisible(True)
        self.preview_group.setVisible(False)

        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        self.rows_cols_badge.setText("—")

        self.rows_label.setText("Rows: —")
        self.cols_label.setText("Columns: —")
        self.target_label.setText("Target: —")

        self.dataset_reset.emit()
        self.ready_changed.emit()

    def _populate_preview(self, df: pd.DataFrame) -> None:
        preview = df.head(PREVIEW_ROWS)
        self.preview_table.clear()
        self.preview_table.setRowCount(len(preview))
        self.preview_table.setColumnCount(len(preview.columns))
        self.rows_cols_badge.setText(f"{len(preview):,} rows × {len(preview.columns):,} cols")

        for c, col in enumerate(preview.columns):
            is_numeric = pd.api.types.is_numeric_dtype(preview[col])
            item = QTableWidgetItem(str(col))
            if qta is not None:
                if is_numeric:
                    item.setIcon(qta.icon("fa5s.hashtag", color="#27d7a3"))
                else:
                    item.setIcon(qta.icon("fa5s.font", color="#f59e0b"))
            else:
                item.setText(("# " if is_numeric else "T ") + str(col))
            self.preview_table.setHorizontalHeaderItem(c, item)

        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                value = preview.iloc[r, c]
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(r, c, item)

        self.preview_table.resizeColumnsToContents()
