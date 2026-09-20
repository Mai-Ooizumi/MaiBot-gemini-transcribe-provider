# Changelog

All notable changes to this plugin are documented here. Release versions are
read from `_manifest.json`; the manual release workflow requires a matching
version section before creating `v<version>`.

## [1.1.0] - 2026-09-21

### Added

- Added `auto`, `inline`, and `files` Gemini audio transport strategies.
- Small audio now uses Interactions inline Base64 with a conservative 256 KiB default limit; larger audio keeps the Files API path.
- Added inline-format-error-only fallback to Files API, while preserving Flash-Lite fallback for normal dedicated-transcription empty responses.
- Added lifecycle-scoped async Gemini Client reuse, API-key isolation, safe configuration invalidation, and unload cleanup.
- Added `pilk-nogil`-first SILK decoding to in-memory WAV, with FFmpeg retained as a fallback for unsupported formats and decoder failures.
- Added a conservative local PCM/WAV silence gate that skips only high-confidence silence and does not classify loud noise from RMS alone.
- Added focused unit coverage for transport selection, fallback, client lifecycle, SILK fallback, speech gating, and legacy configuration defaults.

### Changed

- Files uploads, FFmpeg normalization, and fallback inputs now use in-memory streams where possible, avoiding unnecessary temporary files and Base64 re-encoding.
- Raised the plugin/manifest version to `1.1.0` and documented the new audio and decoder configuration.

### Fixed

- Fixed real MaiBot package-style loading by changing plugin-internal imports to package-relative imports.
- Added a regression test that loads `plugin.py` with MaiBot's `spec_from_file_location` contract and verifies all internal modules resolve.

### Validation

- `pytest -q`: 16 passed.
- Python compilation, TOML/JSON metadata parsing, and Git whitespace checks passed.
