# Changelog

All notable changes to this project will be documented in this file.

## [Alpha 0.0.1] - 2026-01-22

### Added

- Bottom **Status Bar** displaying:
  - CPU usage
  - Memory usage
  - Best-effort GPU usage (NVIDIA via `nvidia-smi`, falls back to `—`)
  - Current time with icon
- **Settings** modal dialog with persisted preferences (via `QSettings`):
  - Data preview row count
  - Status refresh interval
  - Export directory behavior (remember last / default folder)
  - Toggle GPU visibility in the status bar
- **Help** modal dialog:
  - In-app workflow guidance + troubleshooting
  - Quick actions: open GitHub repo, open documentation, report a bug
  - Copy system info to clipboard
- In-app **Toast notifications** system:
  - Animated fade in/out
  - Non-blocking overlay (does not block underlying content)
- **Validation banner** (inline guidance) to explain why the primary action is blocked and how to fix it
- Sidebar **Settings** and **Help** items as clickable actions with icons
- Sidebar header **logo support** (loads from `app/assets/logo.png` with a graceful fallback icon)
- **Real model training** (regression) using scikit-learn:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regressor
  - SVR
  - KNN Regressor
- **Non-blocking training execution** via a separate Python process (Qt `QProcess`):
  - Live streaming logs
  - True mid-fit cancellation (process kill)
- **Run caching** to internal app data directory (`~/.autoregressex/runs/...`) to avoid high RAM usage
- **Training artifacts export**:
  - `model.joblib`
  - `metrics.json`
  - `schema.json`
  - `val_predictions.csv`
  - `plots/` (PNG charts)
- **Export UX** improvements:
  - Export success card
  - Open exported folder
  - Copy exported path
- **Predictions & Insights** page (post-export) that displays generated charts inside the app
- **Restart / New Run** action to reset the entire workflow and cancel any running job
- **Evaluate Model** top-level mode (tab) to validate exported models:
  - Import exported model folder (`model.joblib` + `schema.json`)
  - Evaluate on a CSV with or without the target column
  - Row limit: evaluates up to 100 rows
  - Metrics + prediction preview + charts
  - Export evaluation results

### Changed

- Improved **DropZone** component:
  - Cleaner typography and hints
  - Improved hover + drag-over states
  - Larger badge/icon presentation
- Enhanced **Train** page:
  - Real training results shown per model (R² / MAE / RMSE)
  - Stage + ETA indicators
  - Logs panel with streaming output
- Enhanced **Export** page:
  - Updated artifact list to match real outputs
  - Export now copies the cached run directory contents
- Enhanced **Data Preview**:
  - Column search/filter
  - Sorting enabled on preview table
  - Richer dataset stats: missing values count, numeric/categorical column counts

### Fixed

- Training now validates that the selected regression target is numeric (with actionable error messages)
- Improved robustness for “numeric-looking” CSVs by coercing numeric-like string columns during preprocessing
- Guardrails against very high-cardinality categorical columns that can cause one-hot feature explosions
- `qtawesome` icon crash due to invalid icon name (updated to valid icon names)
- Toast overlay background rendering as a large rectangle (forced transparent host/wrappers)
- Toast card background visibility and text border artifacts (ensured proper paint + removed label borders)
- Notification policy:
  - Dataset load uses toast only
  - Desktop notification reserved for training completion

### Dependencies

- Added: `scikit-learn`, `joblib`, `matplotlib`, `seaborn`
