from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.ml.paths import get_app_data_dir


def _history_path() -> Path:
    d = get_app_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "eval_history.json"


@dataclass
class EvalHistoryItem:
    id: str
    run_dir: str
    created_at: str
    model_dir: str | None
    csv_path: str | None
    target: str | None
    target_present: bool
    n_rows: int | None
    metrics: dict[str, float | None]
    pinned: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalHistoryItem":
        return cls(
            id=str(d.get("id", "")),
            run_dir=str(d.get("run_dir", "")),
            created_at=str(d.get("created_at", "")),
            model_dir=(None if d.get("model_dir") in (None, "") else str(d.get("model_dir"))),
            csv_path=(None if d.get("csv_path") in (None, "") else str(d.get("csv_path"))),
            target=(None if d.get("target") in (None, "") else str(d.get("target"))),
            target_present=bool(d.get("target_present", False)),
            n_rows=(None if d.get("n_rows") is None else int(d.get("n_rows"))),
            metrics=(d.get("metrics") if isinstance(d.get("metrics"), dict) else {"r2": None, "mae": None, "rmse": None}),
            pinned=bool(d.get("pinned", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "model_dir": self.model_dir,
            "csv_path": self.csv_path,
            "target": self.target,
            "target_present": self.target_present,
            "n_rows": self.n_rows,
            "metrics": self.metrics,
            "pinned": self.pinned,
        }


def load_history() -> list[EvalHistoryItem]:
    p = _history_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    items: list[EvalHistoryItem] = []
    for x in raw:
        if isinstance(x, dict):
            try:
                it = EvalHistoryItem.from_dict(x)
            except Exception:
                continue
            if it.id and it.run_dir:
                items.append(it)

    return items


def save_history(items: list[EvalHistoryItem]) -> None:
    p = _history_path()
    tmp = p.with_suffix(".tmp")
    data = [it.to_dict() for it in items]
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _make_id(run_dir: str) -> str:
    return Path(run_dir).name


def add_from_run_dir(run_dir: str, keep: int = 20) -> list[EvalHistoryItem]:
    items = load_history()
    run_path = Path(run_dir)

    summary_path = run_path / "eval_metrics.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    new_item = EvalHistoryItem(
        id=_make_id(run_dir),
        run_dir=str(run_path),
        created_at=str(summary.get("created_at") or datetime.now().isoformat(timespec="seconds")),
        model_dir=(None if summary.get("model_dir") in (None, "") else str(summary.get("model_dir"))),
        csv_path=(None if summary.get("csv_path") in (None, "") else str(summary.get("csv_path"))),
        target=(None if summary.get("target") in (None, "") else str(summary.get("target"))),
        target_present=bool(summary.get("target_present", False)),
        n_rows=(None if summary.get("n_rows") is None else int(summary.get("n_rows"))),
        metrics=(summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {"r2": None, "mae": None, "rmse": None}),
        pinned=False,
    )

    # Remove any previous entry with same id
    items = [it for it in items if it.id != new_item.id]

    # Add newest to front
    items.insert(0, new_item)

    # Retention: keep pinned always, limit non-pinned
    pinned = [it for it in items if it.pinned]
    non_pinned = [it for it in items if not it.pinned]

    items = pinned + non_pinned
    items = pinned + non_pinned[: max(0, int(keep) - len(pinned))]

    save_history(items)
    return items


def toggle_pin(item_id: str) -> list[EvalHistoryItem]:
    items = load_history()
    for it in items:
        if it.id == item_id:
            it.pinned = not it.pinned
            break
    save_history(items)
    return items


def remove_item(item_id: str) -> list[EvalHistoryItem]:
    items = [it for it in load_history() if it.id != item_id]
    save_history(items)
    return items


def clear_history() -> None:
    save_history([])
