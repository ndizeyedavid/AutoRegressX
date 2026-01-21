# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - UI/UX Alpha Improvements

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

### Changed

- Improved **DropZone** component:
  - Cleaner typography and hints
  - Improved hover + drag-over states
  - Larger badge/icon presentation
- Enhanced **Train** page:
  - Cancelable demo training worker
  - Stage + ETA indicators
  - Logs panel improvements:
    - Level filter (All/Info/Success/Warn/Error)
    - Auto-scroll toggle
    - Copy logs
    - Clear logs
- Enhanced **Data Preview**:
  - Column search/filter
  - Sorting enabled on preview table
  - Richer dataset stats: missing values count, numeric/categorical column counts
- Enhanced **Export** page:
  - Export summary card (best model + generated timestamp)
  - Export success state with:
    - Open folder
    - Copy export path
    - Export again
  - Toast feedback on export completion and export-path copy

### Fixed

- `qtawesome` icon crash due to invalid icon name (updated to valid icon names)
- Toast overlay background rendering as a large rectangle (forced transparent host/wrappers)
- Toast card background visibility and text border artifacts (ensured proper paint + removed label borders)
- Notification policy:
  - Dataset load uses toast only
  - Desktop notification reserved for training completion
