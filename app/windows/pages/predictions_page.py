from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _PlotCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("PlotCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.title = QLabel(title)
        self.title.setStyleSheet("font-size: 12pt; font-weight: 650;")

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setMinimumHeight(260)
        self.image.setObjectName("PlotImage")

        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)

        self._pix: QPixmap | None = None

    def set_pixmap(self, pix: QPixmap | None) -> None:
        self._pix = pix
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._pix is None or self._pix.isNull():
            self.image.clear()
            return

        w = max(200, self.image.width())
        h = max(200, self.image.height())
        self.image.setPixmap(self._pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_scale()


class PredictionsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._export_dir: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        header = QLabel("Predictions & Insights")
        header.setStyleSheet("font-size: 16pt; font-weight: 650;")
        hint = QLabel("Review evaluation charts generated from your latest run")
        hint.setStyleSheet("color: #9bb2db;")

        left = QVBoxLayout()
        left.setSpacing(2)
        left.addWidget(header)
        left.addWidget(hint)

        header_row.addLayout(left, 1)
        layout.addLayout(header_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(12)

        self.card_comparison = _PlotCard("Model Comparison (R²)")
        self.card_parity = _PlotCard("Parity Plot (True vs Predicted)")
        self.card_residuals = _PlotCard("Residuals vs Predicted")
        self.card_resid_dist = _PlotCard("Residual Distribution")

        c_layout.addWidget(self.card_comparison)
        c_layout.addWidget(self.card_parity)
        c_layout.addWidget(self.card_residuals)
        c_layout.addWidget(self.card_resid_dist)
        c_layout.addStretch(1)

        self.scroll.setWidget(container)
        layout.addWidget(self.scroll, 1)

        self._set_empty_state()

    def set_export_dir(self, export_dir: str | None) -> None:
        self._export_dir = Path(export_dir).resolve() if export_dir else None
        self._reload_plots()

    def reset(self) -> None:
        self._export_dir = None
        self._set_empty_state()

    def _set_empty_state(self) -> None:
        for card in (self.card_comparison, self.card_parity, self.card_residuals, self.card_resid_dist):
            card.image.setText("No plots yet. Run training and export artifacts to view charts here.")
            card.image.setStyleSheet("color: #9bb2db; padding: 18px;")
            card.set_pixmap(None)

    def _reload_plots(self) -> None:
        if self._export_dir is None:
            self._set_empty_state()
            return

        plots_dir = self._export_dir / "plots"
        if not plots_dir.exists():
            self._set_empty_state()
            return

        mapping: list[tuple[_PlotCard, str]] = [
            (self.card_comparison, "model_comparison_r2.png"),
            (self.card_parity, "best_parity.png"),
            (self.card_residuals, "best_residuals.png"),
            (self.card_resid_dist, "best_residual_distribution.png"),
        ]

        loaded_any = False
        for card, name in mapping:
            p = plots_dir / name
            if p.exists():
                pix = QPixmap(str(p))
                card.image.setStyleSheet("")
                card.set_pixmap(pix)
                loaded_any = True
            else:
                card.image.setText(f"Missing plot: {name}")
                card.image.setStyleSheet("color: #9bb2db; padding: 18px;")
                card.set_pixmap(None)

        if not loaded_any:
            self._set_empty_state()
